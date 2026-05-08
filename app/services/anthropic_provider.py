import asyncio
import base64
import logging

import anthropic
import cv2
import numpy as np

from app.core.config import settings
from app.core.errors import ProviderResponseError, UpstreamServiceError
from app.models.responses import OCRResult, VisionResult
from app.services.provider_common import normalize_media_type, parse_json_response

logger = logging.getLogger(__name__)

_ANTHROPIC_MAX_IMAGE_BYTES = 2 * 1024 * 1024  # 2 MB target


def _compress_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Compress image to stay under Anthropic's 5 MB limit. Returns (bytes, media_type)."""
    if len(image_bytes) <= _ANTHROPIC_MAX_IMAGE_BYTES:
        return image_bytes, None

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    for quality in (85, 70, 55, 40):
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(encoded.tobytes()) <= _ANTHROPIC_MAX_IMAGE_BYTES:
            logger.debug(
                "Compressed image from %d to %d bytes (quality=%d)",
                len(image_bytes),
                len(encoded.tobytes()),
                quality,
            )
            return encoded.tobytes(), "image/jpeg"

    # Last resort: halve resolution
    h, w = img.shape[:2]
    img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    logger.warning("Image required resize to fit 5 MB limit, new size=%d bytes", len(encoded.tobytes()))
    return encoded.tobytes(), "image/jpeg"

_IDENTITY_DOCUMENT_TYPES = {"INE", "INE_REVERSO", "PASAPORTE", "LICENCIA"}

_EXTRACT_PROMPT = """Analyze this document image and extract all visible text.
Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{
  "raw_text": "<all text visible in the document as a single string>",
  "structured_fields": {
    "<field_name>": "<value>"
  },
  "confidence": <float between 0.0 and 1.0 representing extraction confidence>
}

For structured_fields, extract key-value pairs using lowercase English keys such as:
date, total, issuer, receipt_number, full_name, id_number, curp, expiry_date, date_of_birth, address, rfc, folio.
Only include fields that are actually visible in the document."""

_INE_REVERSO_EXTRACT_PROMPT = """This is the back side of a Mexican INE card.
Look for the main printed identifier on the back, such as folio, id number, CIC, OCR, or a similar visible document code.
Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{
  "raw_text": "<visible identifier text or short OCR snippet>",
  "structured_fields": {
    "id_number": "<best identifier found>",
    "label": "<folio|id_number|cic|ocr|unknown>"
  },
  "confidence": <float between 0.0 and 1.0>
}

If nothing is clearly visible, return an empty id_number and label "unknown"."""

_INE_FRONT_EXTRACT_PROMPT = """This is the front side of a Mexican INE card.
Extract only the main operational identity fields needed for validation.
Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{
  "raw_text": "<relevant visible text from the front side>",
  "structured_fields": {
    "full_name": "<full name if visible>",
    "id_number": "<document identifier if visible>",
    "curp": "<CURP if visible>",
    "expiry_date": "<expiry date if visible>",
    "date_of_birth": "<date of birth if visible>"
  },
  "confidence": <float between 0.0 and 1.0>
}

Only include fields that are clearly visible on the front side of the card."""

_ADDRESS_PROOF_EXTRACT_PROMPT = """Este es un comprobante de domicilio mexicano (recibo de luz, agua, teléfono, estado de cuenta, etc.).

El documento puede contener DOS direcciones:
- La dirección del CLIENTE (titular del servicio): es la que nos interesa. Aparece junto al nombre del cliente.
- La dirección de la EMPRESA emisora (CFE, TELMEX, etc.): NO la queremos. Suele aparecer en el encabezado o pie con datos fiscales de la empresa.

Extrae ÚNICAMENTE la dirección del CLIENTE titular, no la de la empresa emisora.

Devuelve ÚNICAMENTE un JSON válido (sin markdown, sin explicaciones) con esta estructura exacta:
{
  "raw_text": "<texto relevante visible del documento>",
  "structured_fields": {
    "issuer": "<nombre de la empresa emisora (CFE, TELMEX, JUMAPAC, BBVA, etc.)>",
    "street": "<calle y número del CLIENTE>",
    "colony": "<colonia o fraccionamiento del CLIENTE>",
    "zip_code": "<código postal del CLIENTE (5 dígitos)>",
    "city": "<municipio o alcaldía del CLIENTE>",
    "state": "<estado de la república del CLIENTE>",
    "issue_date": "<fecha del recibo en formato YYYY-MM-DD>"
  },
  "confidence": <float entre 0.0 y 1.0>
}

Si el documento muestra un periodo de facturación, usa la fecha de fin del periodo como issue_date.
Solo incluye campos que sean claramente visibles."""

_VISION_PROMPT_TEMPLATE = """Analyze this {document_type} document image for authenticity and potential fraud.

Examine:
- Physical integrity (tears, folds, unusual damage)
- Print quality and consistency
- Font uniformity and spacing
- Security features appropriate for this document type
- Signs of digital manipulation or alteration
- Whether the document structure matches the expected format for {document_type}

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{{
  "is_authentic": <true or false>,
  "fraud_indicators": ["<indicator 1>", "<indicator 2>"],
  "authenticity_score": <float between 0.0 and 1.0>,
  "notes": "<brief observation about the document>"
}}

fraud_indicators should be an empty list [] if no issues are found.
Be conservative: only flag clear anomalies as fraud indicators."""

_IDENTITY_VISION_PROMPT_TEMPLATE = """Analyze this {document_type} identity document image for operational validation, not forensic authenticity.

Assess:
- whether the document visually matches the expected type
- whether the image quality is sufficient for review
- whether the key text zones appear legible
- whether there are obvious capture issues such as blur, glare, crop, low contrast, or partial framing
- whether there are basic inconsistencies between the visible document and the expected type

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{{
  "document_matches_expected_type": <true or false>,
  "visual_validation_score": <float between 0.0 and 1.0>,
  "quality_flags": ["<quality issue 1>", "<quality issue 2>"],
  "consistency_flags": ["<consistency issue 1>", "<consistency issue 2>"],
  "notes": "<brief operational observation about usability>"
}}

Use quality_flags for capture problems and consistency_flags for mismatches or uncertainty.
Be conservative: if the image is ambiguous, lower the score and add flags instead of asserting authenticity or fraud."""


class AnthropicOCRService:
    def __init__(self, config=settings) -> None:
        self.settings = config
        self.client = anthropic.AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)

    async def extract_text(
        self,
        image_bytes: bytes,
        media_type: str = "image/jpeg",
        document_type: str | None = None,
    ) -> OCRResult:
        image_bytes, compressed_media_type = _compress_image(image_bytes)
        validated_media_type = normalize_media_type(compressed_media_type or media_type)
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        prompt = (
            _INE_REVERSO_EXTRACT_PROMPT
            if document_type == "INE_REVERSO"
            else _INE_FRONT_EXTRACT_PROMPT
            if document_type == "INE"
            else _ADDRESS_PROOF_EXTRACT_PROMPT
            if document_type in ("ADDRESS_PROOF", "COMPROBANTE_DOMICILIO")
            else _EXTRACT_PROMPT
        )

        logger.debug(
            "Sending image to Anthropic for OCR extraction (size=%d bytes)",
            len(image_bytes),
        )

        response = await self._request(
            max_tokens=2048,
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": validated_media_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
            operation_name="OCR",
        )
        raw_response = self._extract_text_block(
            response, "OCR provider returned an unexpected response shape."
        )
        data = parse_json_response(raw_response, "OCR provider returned invalid JSON.")

        return OCRResult(
            raw_text=data.get("raw_text", ""),
            structured_fields=data.get("structured_fields", {}),
            confidence=float(data.get("confidence", 0.5)),
        )

    async def _request(self, max_tokens: int, content: list[dict], operation_name: str):
        last_error: Exception | None = None
        for attempt in range(self.settings.ANTHROPIC_MAX_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self.client.messages.create(
                        model=self.settings.ANTHROPIC_MODEL,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": content}],
                    ),
                    timeout=self.settings.ANTHROPIC_TIMEOUT_SECONDS,
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logger.warning(
                    "Anthropic %s request failed on attempt %d: %s",
                    operation_name,
                    attempt + 1,
                    exc,
                )
                if attempt < self.settings.ANTHROPIC_MAX_RETRIES:
                    await asyncio.sleep(0.25 * (attempt + 1))
        raise UpstreamServiceError(f"{operation_name} provider is unavailable.") from last_error

    @staticmethod
    def _extract_text_block(response, error_message: str) -> str:
        try:
            return response.content[0].text
        except (AttributeError, IndexError, TypeError) as exc:
            raise ProviderResponseError(error_message) from exc


class AnthropicVisionService:
    def __init__(self, config=settings) -> None:
        self.settings = config
        self.client = anthropic.AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)

    async def analyze_document(
        self,
        image_bytes: bytes,
        document_type: str,
        media_type: str = "image/jpeg",
    ) -> VisionResult:
        image_bytes, compressed_media_type = _compress_image(image_bytes)
        validated_media_type = normalize_media_type(compressed_media_type or media_type)
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        prompt_template = (
            _IDENTITY_VISION_PROMPT_TEMPLATE
            if document_type in _IDENTITY_DOCUMENT_TYPES
            else _VISION_PROMPT_TEMPLATE
        )
        prompt = prompt_template.format(document_type=document_type)

        logger.debug(
            "Sending image to Anthropic for vision analysis (document_type=%s, size=%d bytes)",
            document_type,
            len(image_bytes),
        )

        response = await self._request(
            max_tokens=1024,
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": validated_media_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
            operation_name="Vision",
        )
        raw_response = self._extract_text_block(
            response,
            "Vision provider returned an unexpected response shape.",
        )
        data = parse_json_response(raw_response, "Vision provider returned invalid JSON.")

        if document_type in _IDENTITY_DOCUMENT_TYPES:
            quality_flags = [str(flag) for flag in data.get("quality_flags", [])]
            consistency_flags = [str(flag) for flag in data.get("consistency_flags", [])]
            visual_score = float(data.get("visual_validation_score", 0.5))
            matches_expected = bool(data.get("document_matches_expected_type", True))
            return VisionResult(
                is_authentic=matches_expected,
                fraud_indicators=[*quality_flags, *consistency_flags],
                authenticity_score=visual_score,
                document_matches_expected_type=matches_expected,
                visual_validation_score=visual_score,
                quality_flags=quality_flags,
                consistency_flags=consistency_flags,
                notes=data.get("notes", ""),
            )

        return VisionResult(
            is_authentic=bool(data.get("is_authentic", False)),
            fraud_indicators=data.get("fraud_indicators", []),
            authenticity_score=float(data.get("authenticity_score", 0.5)),
            document_matches_expected_type=bool(data.get("is_authentic", False)),
            visual_validation_score=float(data.get("authenticity_score", 0.5)),
            quality_flags=[str(flag) for flag in data.get("fraud_indicators", [])],
            consistency_flags=[],
            notes=data.get("notes", ""),
        )

    async def _request(self, max_tokens: int, content: list[dict], operation_name: str):
        last_error: Exception | None = None
        for attempt in range(self.settings.ANTHROPIC_MAX_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self.client.messages.create(
                        model=self.settings.ANTHROPIC_MODEL,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": content}],
                    ),
                    timeout=self.settings.ANTHROPIC_TIMEOUT_SECONDS,
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logger.warning(
                    "Anthropic %s request failed on attempt %d: %s",
                    operation_name,
                    attempt + 1,
                    exc,
                )
                if attempt < self.settings.ANTHROPIC_MAX_RETRIES:
                    await asyncio.sleep(0.25 * (attempt + 1))
        raise UpstreamServiceError(f"{operation_name} provider is unavailable.") from last_error

    @staticmethod
    def _extract_text_block(response, error_message: str) -> str:
        try:
            return response.content[0].text
        except (AttributeError, IndexError, TypeError) as exc:
            raise ProviderResponseError(error_message) from exc

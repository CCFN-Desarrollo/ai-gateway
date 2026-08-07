import asyncio
import base64
import logging

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.errors import ProviderResponseError, UpstreamServiceError
from app.models.responses import DocumentTypeClassification, OCRResult, VisionResult
from app.services.document_classifier import (
    CLASSIFY_IDENTITY_DOCUMENT_PROMPT,
    parse_classification_payload,
)
from app.services.provider_common import normalize_media_type, parse_json_response

logger = logging.getLogger(__name__)

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

_ADDRESS_PROOF_EXTRACT_PROMPT = """This is a Mexican proof-of-address document such as a utility bill.
Extract only the address and freshness fields needed for validation.
Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{
  "raw_text": "<relevant visible text from the document>",
  "structured_fields": {
    "issuer": "<issuer or company name if visible>",
    "street": "<street and number if visible>",
    "colony": "<colony or neighborhood if visible>",
    "zip_code": "<postal code if visible>",
    "city": "<city or municipality if visible>",
    "state": "<state if visible>",
    "issue_date": "<document issue date in YYYY-MM-DD if visible>"
  },
  "confidence": <float between 0.0 and 1.0>
}

If the document shows a billing period, use the end date of that period as issue_date.
Only include fields that are clearly visible."""

_CSF_EXTRACT_PROMPT = """Esta es una página de una Constancia de Situación Fiscal (CSF) emitida por el SAT (Servicio de Administración Tributaria de México).

Extrae todos los datos fiscales visibles en esta página.

Devuelve ÚNICAMENTE un JSON válido (sin markdown, sin explicaciones) con esta estructura exacta:
{
  "raw_text": "<texto relevante visible en la página>",
  "structured_fields": {
    "rfc": "<RFC del contribuyente, 12 o 13 caracteres alfanuméricos>",
    "full_name": "<nombre completo o razón social del contribuyente>",
    "curp": "<CURP si está visible, 18 caracteres, solo personas físicas>",
    "zip_code": "<código postal del domicilio fiscal, 5 dígitos>",
    "street": "<tipo de vía, nombre de calle y número del domicilio fiscal>",
    "colony": "<colonia o fraccionamiento del domicilio fiscal>",
    "city": "<municipio o alcaldía del domicilio fiscal>",
    "state": "<entidad federativa del domicilio fiscal>",
    "start_date": "<fecha de inicio de operaciones en formato YYYY-MM-DD>",
    "last_change_date": "<fecha de último cambio de situación en formato YYYY-MM-DD>",
    "fiscal_regimes": ["<régimen fiscal 1>", "<régimen fiscal 2>"],
    "fiscal_obligations": ["<descripción de obligación 1>", "<descripción de obligación 2>"]
  },
  "confidence": <float entre 0.0 y 1.0>
}

Notas: fiscal_regimes y fiscal_obligations son listas ([] si no hay). Solo incluye campos claramente visibles."""

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

Type-matching guidance:
- If expected type is PASAPORTE: U.S. passports, U.S. visas, and Border Crossing Cards (B1/B2 VISA / BCC) issued by the United States Department of State MATCH. English headers such as "UNITED STATES OF AMERICA" are expected. Holder nationality MEXICAN is not a mismatch and does NOT mean the document is an INE.
- If expected type is INE: require Mexican voter-ID branding (Instituto Nacional Electoral / Credencial para Votar). A U.S. visa or BCC is not an INE.

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


class OpenAIOCRService:
    def __init__(self, config=settings) -> None:
        self.settings = config
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)

    async def extract_text(
        self,
        image_bytes: bytes,
        media_type: str = "image/jpeg",
        document_type: str | None = None,
    ) -> OCRResult:
        validated_media_type = normalize_media_type(media_type)
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        prompt = (
            _INE_REVERSO_EXTRACT_PROMPT
            if document_type == "INE_REVERSO"
            else _INE_FRONT_EXTRACT_PROMPT
            if document_type == "INE"
            else _ADDRESS_PROOF_EXTRACT_PROMPT
            if document_type in ("ADDRESS_PROOF", "COMPROBANTE_DOMICILIO")
            else _CSF_EXTRACT_PROMPT
            if document_type == "CSF"
            else _EXTRACT_PROMPT
        )

        logger.debug("Sending image to OpenAI for OCR extraction (size=%d bytes)", len(image_bytes))
        response = await self._request(prompt, image_b64, validated_media_type, "OCR")
        raw_response = self._extract_text_block(
            response, "OCR provider returned an unexpected response shape."
        )
        data = parse_json_response(raw_response, "OCR provider returned invalid JSON.")

        return OCRResult(
            raw_text=data.get("raw_text", ""),
            structured_fields=data.get("structured_fields", {}),
            confidence=float(data.get("confidence", 0.5)),
        )

    async def classify_document(
        self,
        image_bytes: bytes,
        media_type: str = "image/jpeg",
    ) -> DocumentTypeClassification:
        """Alternate flow: infer document_type from image content only (ignore filenames)."""
        validated_media_type = normalize_media_type(media_type)
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        logger.debug(
            "Sending image to OpenAI for document type classification (size=%d bytes)",
            len(image_bytes),
        )
        response = await self._request(
            CLASSIFY_IDENTITY_DOCUMENT_PROMPT,
            image_b64,
            validated_media_type,
            "Classify",
        )
        raw_response = self._extract_text_block(
            response, "Classify provider returned an unexpected response shape."
        )
        data = parse_json_response(raw_response, "Classify provider returned invalid JSON.")
        return parse_classification_payload(data)

    async def _request(
        self,
        prompt: str,
        image_b64: str,
        media_type: str,
        operation_name: str,
    ):
        last_error: Exception | None = None
        for attempt in range(self.settings.OPENAI_MAX_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self.client.responses.create(
                        model=self.settings.OPENAI_MODEL,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt},
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:{media_type};base64,{image_b64}",
                                    },
                                ],
                            }
                        ],
                    ),
                    timeout=self.settings.OPENAI_TIMEOUT_SECONDS,
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logger.warning(
                    "OpenAI %s request failed on attempt %d: %s",
                    operation_name,
                    attempt + 1,
                    exc,
                )
                if attempt < self.settings.OPENAI_MAX_RETRIES:
                    await asyncio.sleep(0.25 * (attempt + 1))
        raise UpstreamServiceError(f"{operation_name} provider is unavailable.") from last_error

    @staticmethod
    def _extract_text_block(response, error_message: str) -> str:
        text = getattr(response, "output_text", "")
        if not text:
            raise ProviderResponseError(error_message)
        return text


class OpenAIVisionService:
    def __init__(self, config=settings) -> None:
        self.settings = config
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)

    async def analyze_document(
        self,
        image_bytes: bytes,
        document_type: str,
        media_type: str = "image/jpeg",
    ) -> VisionResult:
        validated_media_type = normalize_media_type(media_type)
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        prompt_template = (
            _IDENTITY_VISION_PROMPT_TEMPLATE
            if document_type in _IDENTITY_DOCUMENT_TYPES
            else _VISION_PROMPT_TEMPLATE
        )
        prompt = prompt_template.format(document_type=document_type)

        logger.debug(
            "Sending image to OpenAI for vision analysis (document_type=%s, size=%d bytes)",
            document_type,
            len(image_bytes),
        )
        response = await self._request(prompt, image_b64, validated_media_type, "Vision")
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

    async def _request(
        self,
        prompt: str,
        image_b64: str,
        media_type: str,
        operation_name: str,
    ):
        last_error: Exception | None = None
        for attempt in range(self.settings.OPENAI_MAX_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self.client.responses.create(
                        model=self.settings.OPENAI_MODEL,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt},
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:{media_type};base64,{image_b64}",
                                    },
                                ],
                            }
                        ],
                    ),
                    timeout=self.settings.OPENAI_TIMEOUT_SECONDS,
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logger.warning(
                    "OpenAI %s request failed on attempt %d: %s",
                    operation_name,
                    attempt + 1,
                    exc,
                )
                if attempt < self.settings.OPENAI_MAX_RETRIES:
                    await asyncio.sleep(0.25 * (attempt + 1))
        raise UpstreamServiceError(f"{operation_name} provider is unavailable.") from last_error

    @staticmethod
    def _extract_text_block(response, error_message: str) -> str:
        text = getattr(response, "output_text", "")
        if not text:
            raise ProviderResponseError(error_message)
        return text

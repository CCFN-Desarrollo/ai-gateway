import asyncio
import base64
import logging

import httpx

from app.core.config import settings
from app.core.errors import ProviderResponseError, UpstreamServiceError
from app.models.responses import OCRResult, VisionResult
from app.services.provider_common import parse_json_response

logger = logging.getLogger(__name__)
_MAX_DEBUG_PAYLOAD_CHARS = 1200

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


class OllamaOCRService:
    def __init__(self, config=settings) -> None:
        self.settings = config
        self.base_url = self.settings.OLLAMA_HOST.rstrip("/")

    async def extract_text(
        self,
        image_bytes: bytes,
        media_type: str = "image/jpeg",
        document_type: str | None = None,
    ) -> OCRResult:
        del media_type
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

        logger.debug("Sending image to Ollama for OCR extraction (size=%d bytes)", len(image_bytes))

        data = await self._request(prompt, image_b64)
        return OCRResult(
            raw_text=str(data.get("raw_text", "")),
            structured_fields=data.get("structured_fields", {}),
            confidence=float(data.get("confidence", 0.5)),
        )

    async def _request(self, prompt: str, image_b64: str) -> dict:
        payload = {
            "model": self.settings.OLLAMA_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": "json",
        }
        return await _request_ollama_json(
            base_url=self.base_url,
            payload=payload,
            timeout_seconds=self.settings.OLLAMA_TIMEOUT_SECONDS,
            max_retries=self.settings.OLLAMA_MAX_RETRIES,
            operation_name="OCR",
        )


class OllamaVisionService:
    def __init__(self, config=settings) -> None:
        self.settings = config
        self.base_url = self.settings.OLLAMA_HOST.rstrip("/")

    async def analyze_document(
        self,
        image_bytes: bytes,
        document_type: str,
        media_type: str = "image/jpeg",
    ) -> VisionResult:
        del media_type
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        prompt = _IDENTITY_VISION_PROMPT_TEMPLATE.format(document_type=document_type)

        logger.debug(
            "Sending image to Ollama for vision analysis (document_type=%s, size=%d bytes)",
            document_type,
            len(image_bytes),
        )

        data = await self._request(prompt, image_b64)
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

    async def _request(self, prompt: str, image_b64: str) -> dict:
        payload = {
            "model": self.settings.OLLAMA_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": "json",
        }
        return await _request_ollama_json(
            base_url=self.base_url,
            payload=payload,
            timeout_seconds=self.settings.OLLAMA_TIMEOUT_SECONDS,
            max_retries=self.settings.OLLAMA_MAX_RETRIES,
            operation_name="Vision",
        )


async def _request_ollama_json(
    *,
    base_url: str,
    payload: dict,
    timeout_seconds: float,
    max_retries: int,
    operation_name: str,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            body = response.json()
            content = _extract_ollama_content(body, operation_name)
            if not isinstance(content, str) or not content.strip():
                logger.warning(
                    "Ollama %s returned unexpected payload: %s",
                    operation_name,
                    _truncate_for_log(body),
                )
                raise ProviderResponseError(
                    f"{operation_name} provider returned an unexpected response."
                )
            return parse_json_response(content, f"{operation_name} provider returned invalid JSON.")
        except ProviderResponseError:
            raise
        except Exception as exc:  # pragma: no cover
            last_error = exc
            logger.warning(
                "Ollama %s request failed on attempt %d [%s]: %s",
                operation_name,
                attempt + 1,
                type(exc).__name__,
                exc,
            )
            if attempt < max_retries:
                await asyncio.sleep(0.25 * (attempt + 1))
    raise UpstreamServiceError(f"{operation_name} provider is unavailable.") from last_error


def _truncate_for_log(payload: object) -> str:
    text = repr(payload)
    if len(text) <= _MAX_DEBUG_PAYLOAD_CHARS:
        return text
    return f"{text[:_MAX_DEBUG_PAYLOAD_CHARS]}...<truncated>"


def _extract_ollama_content(body: dict, operation_name: str) -> str | None:
    response_content = body.get("response")
    if isinstance(response_content, str) and response_content.strip():
        return response_content

    thinking_content = body.get("thinking")
    if isinstance(thinking_content, str) and thinking_content.strip():
        logger.warning(
            "Ollama %s returned content in 'thinking'; using it as fallback.",
            operation_name,
        )
        return thinking_content

    return None

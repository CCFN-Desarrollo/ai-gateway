import json
import logging
import re

from app.core.errors import ProviderResponseError

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
_REFUSAL_PATTERNS = (
    re.compile(r"\bi(?:'m| am) not able to\b", re.IGNORECASE),
    re.compile(r"\bi (?:cannot|can't|must decline|must refuse)\b", re.IGNORECASE),
    re.compile(r"\bi (?:won't|will not) (?:extract|process|analyze)\b", re.IGNORECASE),
    re.compile(r"\b(?:identity theft|document fraud)\b", re.IGNORECASE),
    re.compile(r"\bofficial government systems\b", re.IGNORECASE),
)


def raise_if_provider_refusal(text: str, operation_name: str = "Document") -> None:
    """Raise when the model refuses to process an identity/KYC document."""
    sample = (text or "").strip()
    if not sample:
        return
    # Refusals are plain prose; valid OCR responses are (or contain) JSON objects.
    if "{" in sample and "}" in sample:
        return
    if any(pattern.search(sample) for pattern in _REFUSAL_PATTERNS):
        preview = sample if len(sample) <= 300 else f"{sample[:300]}..."
        logger.warning("%s provider refused the request. Preview: %r", operation_name, preview)
        raise ProviderResponseError(
            f"{operation_name} provider refused to process this identity document."
        )


def normalize_media_type(media_type: str) -> str:
    if media_type in ALLOWED_IMAGE_CONTENT_TYPES:
        return media_type
    if "png" in media_type:
        return "image/png"
    if "webp" in media_type:
        return "image/webp"
    return "image/jpeg"


def parse_json_response(text: str, error_message: str) -> dict:
    """Parse a provider JSON object, tolerating markdown fences and surrounding prose."""
    text = (text or "").strip()
    if not text:
        logger.warning("%s Empty provider response.", error_message)
        raise ProviderResponseError(error_message)

    raise_if_provider_refusal(text, operation_name="Document AI")

    last_error: Exception | None = None
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(data, dict):
            return data
        last_error = ProviderResponseError(error_message)

    preview = text if len(text) <= 500 else f"{text[:500]}..."
    logger.warning("%s Raw response preview: %r", error_message, preview)
    raise ProviderResponseError(error_message) from last_error


def _json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        if not value:
            return
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            candidates.append(value)

    add(text)

    fence_match = _FENCE_RE.match(text)
    if fence_match:
        add(fence_match.group(1))
    elif text.startswith("```"):
        lines = text.splitlines()
        body = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        add("\n".join(body))

    for source in list(candidates):
        add(_extract_first_json_object(source))

    return candidates


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None

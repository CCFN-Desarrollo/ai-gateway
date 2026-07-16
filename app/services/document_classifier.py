"""Classify identity document type from image content (before any type-specific crop)."""

from app.models.responses import DocumentTypeClassification

IDENTITY_DOCUMENT_TYPES = frozenset({"INE", "INE_REVERSO", "PASAPORTE", "LICENCIA"})
CLASSIFY_CONFIDENCE_THRESHOLD = 0.7

# Normalize common model aliases into gateway document types.
_TYPE_ALIASES = {
    "VISA": "PASAPORTE",
    "BCC": "PASAPORTE",
    "BORDER_CROSSING_CARD": "PASAPORTE",
    "BORDER CROSSING CARD": "PASAPORTE",
    "US_VISA": "PASAPORTE",
    "U.S._VISA": "PASAPORTE",
    "PASSPORT": "PASAPORTE",
}

# Strong English issuer cues for U.S. travel documents (visa / BCC / passport).
_ENGLISH_VISA_STRONG_CUES = (
    "united states of america",
    "united states department of state",
    "department of state",
    "border crossing card",
    "b1/b2",
    "b1/b2 visa",
    "visa / border",
)
# Supporting English field-label cues (need a strong cue, or several together).
_ENGLISH_VISA_SUPPORTING_CUES = (
    "visa",
    "passport",
    "surname",
    "given names",
    "expires on",
    "date of issue",
)

# Spanish cues that identify Mexican INE (issuer branding), not mere nationality.
_SPANISH_INE_CUES = (
    "instituto nacional electoral",
    "credencial para votar",
    "clave de elector",
    "año de registro",
    "ano de registro",
    "espacio para firma",
)

CLASSIFY_IDENTITY_DOCUMENT_PROMPT = """You are classifying an identity document from the IMAGE attached to this request.

CRITICAL — read the IMAGE text carefully:
- Look only at the provided image pixels and the printed/visible text in that image.
- Do NOT use any filename, file path, upload name, or metadata. Filenames may be random or inaccurate.
- FIRST scan for English text. THEN scan for Spanish text. Report both separately.
- Distinguish ISSUER from NATIONALITY. The word "MEXICAN" under Nationality does NOT mean the document is an INE.

English (visa / BCC / passport) cues — if present, type is PASAPORTE:
- "UNITED STATES OF AMERICA"
- "UNITED STATES DEPARTMENT OF STATE"
- "B1/B2 VISA", "BORDER CROSSING CARD", "BCC", "VISA", "PASSPORT"
- English labels: Surname, Given Names, Date of Birth, Date of Issue, Expires On, Nationality, Sex

Spanish (INE) cues — only these issuer phrases mean INE:
- "Instituto Nacional Electoral"
- "Credencial para Votar"
- Related INE labels such as "Clave de elector" on a voter credential
- Do NOT treat Spanish-looking layout alone, plastic ID size, or nationality Mexican as INE.

Classification rules (stop at first match):
1. PASAPORTE — English visa/BCC/passport issuer or labels above are visible (even if Nationality is MEXICAN).
2. INE — Spanish INE issuer phrases above are clearly visible on the front.
3. INE_REVERSO — back of Mexican INE (IDMEX MRZ, barcode/QR, CIC/OCR, fingerprint map).
4. LICENCIA — driver's license / licencia de conducir.

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{
  "document_type": "INE|INE_REVERSO|PASAPORTE|LICENCIA",
  "confidence": <float between 0.0 and 1.0>,
  "english_text_found": ["<exact English phrases read from the image>", "..."],
  "spanish_text_found": ["<exact Spanish phrases read from the image>", "..."],
  "evidence": ["<strongest issuer/header lines read from the image>", "..."],
  "notes": "<brief reason; mention English vs Spanish findings and issuer vs nationality>"
}

Populate english_text_found and spanish_text_found from text you actually read.
For a U.S. B1/B2 Border Crossing Card, english_text_found must include headers such as
"UNITED STATES OF AMERICA" and/or "B1/B2 VISA / BORDER CROSSING CARD", and document_type must be PASAPORTE.
If uncertain, pick the most likely type and lower confidence."""


def _normalize_type(raw_type: str | None) -> str | None:
    if not raw_type:
        return None
    normalized = raw_type.strip().upper().replace("-", "_").replace(" ", "_")
    # Also try spaced form for alias table keys that use spaces.
    spaced = raw_type.strip().upper()
    aliased = _TYPE_ALIASES.get(normalized) or _TYPE_ALIASES.get(spaced) or normalized
    return aliased if aliased in IDENTITY_DOCUMENT_TYPES else None


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def parse_classification_payload(data: dict) -> DocumentTypeClassification:
    document_type = _normalize_type(str(data.get("document_type", "")).strip().upper())
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    result = DocumentTypeClassification(
        document_type=document_type,
        confidence=confidence,
        evidence=_coerce_str_list(data.get("evidence", [])),
        english_text_found=_coerce_str_list(data.get("english_text_found", [])),
        spanish_text_found=_coerce_str_list(data.get("spanish_text_found", [])),
        notes=str(data.get("notes", "") or ""),
    )
    return refine_classification(result)


def _blob_has_cues(blob: str, cues: tuple[str, ...]) -> bool:
    return any(cue in blob for cue in cues)


def _has_english_visa_signals(english_blob: str, evidence_blob: str) -> bool:
    blob = f"{english_blob} {evidence_blob}"
    if _blob_has_cues(blob, _ENGLISH_VISA_STRONG_CUES):
        return True
    supporting_hits = sum(1 for cue in _ENGLISH_VISA_SUPPORTING_CUES if cue in blob)
    return supporting_hits >= 3


def refine_classification(result: DocumentTypeClassification) -> DocumentTypeClassification:
    """
    Prefer bilingual cue lists over a wrong top-level label.

    Protects U.S. visa/BCC cards (English) from being labeled INE just because
    nationality is Mexican, and requires Spanish INE issuer branding for INE.
    """
    english_blob = " ".join(result.english_text_found).lower()
    spanish_blob = " ".join(result.spanish_text_found).lower()
    evidence_blob = " ".join([result.notes, *result.evidence]).lower()
    combined = f"{english_blob} {spanish_blob} {evidence_blob}"

    has_english_visa = _has_english_visa_signals(english_blob, evidence_blob)
    # Require true INE issuer Spanish cues; ignore weak "mexican" in nationality.
    has_spanish_ine = _blob_has_cues(spanish_blob, _SPANISH_INE_CUES) or _blob_has_cues(
        evidence_blob, _SPANISH_INE_CUES
    )

    if has_english_visa and not has_spanish_ine:
        return result.model_copy(
            update={
                "document_type": "PASAPORTE",
                "confidence": max(result.confidence, 0.9),
                "notes": (
                    f"{result.notes} | refined_to_PASAPORTE: English visa/BCC/passport text "
                    "found (not INE despite any Mexican nationality)."
                ).strip(" |"),
            }
        )

    if has_spanish_ine and not has_english_visa and result.document_type not in {
        "INE",
        "INE_REVERSO",
    }:
        return result.model_copy(
            update={
                "document_type": "INE",
                "confidence": max(result.confidence, 0.9),
                "notes": (
                    f"{result.notes} | refined_to_INE: Spanish INE issuer text found "
                    "(Instituto Nacional Electoral / Credencial para Votar)."
                ).strip(" |"),
            }
        )

    # If model returned VISA/BCC alias already normalized elsewhere, keep PASAPORTE.
    if result.document_type == "PASAPORTE" and "mexican" in combined and has_english_visa:
        return result.model_copy(update={"confidence": max(result.confidence, 0.9)})

    return result


def resolve_document_type(
    classification: DocumentTypeClassification | None,
    hinted_type: str | None,
) -> tuple[str | None, list[str]]:
    """
    Resolve final document_type before preprocessing/cropping.

    English visa/BCC cues win over an INE hint so specialized INE crops are not applied
    to U.S. travel cards.
    """
    flags: list[str] = []
    hinted = _normalize_type(hinted_type) if hinted_type else None
    classified = classification.document_type if classification else None

    english_blob = " ".join(classification.english_text_found).lower() if classification else ""
    spanish_blob = " ".join(classification.spanish_text_found).lower() if classification else ""
    evidence_blob = (
        " ".join([classification.notes, *classification.evidence]).lower()
        if classification
        else ""
    )
    has_english_visa = _has_english_visa_signals(english_blob, evidence_blob)
    has_spanish_ine = _blob_has_cues(spanish_blob, _SPANISH_INE_CUES) or _blob_has_cues(
        evidence_blob, _SPANISH_INE_CUES
    )

    if has_english_visa and not has_spanish_ine:
        if hinted and hinted != "PASAPORTE":
            flags.append("document_type_overridden_by_classifier")
        return "PASAPORTE", flags

    if has_spanish_ine and not has_english_visa:
        resolved = classified if classified in {"INE", "INE_REVERSO"} else "INE"
        if hinted and hinted != resolved:
            flags.append("document_type_overridden_by_classifier")
        return resolved, flags

    if (
        classified
        and classification
        and classification.confidence >= CLASSIFY_CONFIDENCE_THRESHOLD
    ):
        if hinted and hinted != classified:
            flags.append("document_type_overridden_by_classifier")
        return classified, flags

    if hinted:
        if classified and classified != hinted or classification is not None and (
            classified is None or classification.confidence < CLASSIFY_CONFIDENCE_THRESHOLD
        ):
            flags.append("document_type_uncertain")
        return hinted, flags

    if classified:
        flags.append("document_type_uncertain")
        return classified, flags

    return None, ["document_type_uncertain"]

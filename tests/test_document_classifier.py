"""Unit tests for bilingual document type classification helpers."""

from app.models.responses import DocumentTypeClassification
from app.services.document_classifier import (
    CLASSIFY_IDENTITY_DOCUMENT_PROMPT,
    parse_classification_payload,
    refine_classification,
    resolve_document_type,
)


class TestClassifyPrompt:
    def test_prompt_requires_bilingual_text_scan(self):
        assert "English" in CLASSIFY_IDENTITY_DOCUMENT_PROMPT
        assert "Spanish" in CLASSIFY_IDENTITY_DOCUMENT_PROMPT
        assert "english_text_found" in CLASSIFY_IDENTITY_DOCUMENT_PROMPT
        assert "spanish_text_found" in CLASSIFY_IDENTITY_DOCUMENT_PROMPT
        assert "UNITED STATES OF AMERICA" in CLASSIFY_IDENTITY_DOCUMENT_PROMPT
        assert "B1/B2 VISA" in CLASSIFY_IDENTITY_DOCUMENT_PROMPT
        assert "Instituto Nacional Electoral" in CLASSIFY_IDENTITY_DOCUMENT_PROMPT
        assert "Credencial para Votar" in CLASSIFY_IDENTITY_DOCUMENT_PROMPT
        assert "filename" in CLASSIFY_IDENTITY_DOCUMENT_PROMPT.lower()


class TestParseAndRefine:
    def test_us_bcc_english_cues_override_ine_label(self):
        result = parse_classification_payload(
            {
                "document_type": "INE",
                "confidence": 0.6,
                "english_text_found": [
                    "UNITED STATES OF AMERICA",
                    "B1/B2 VISA / BORDER CROSSING CARD",
                    "Surname",
                    "Given Names",
                ],
                "spanish_text_found": [],
                "evidence": ["UNITED STATES OF AMERICA", "Nationality: MEXICAN"],
                "notes": "Holder is Mexican",
            }
        )
        assert result.document_type == "PASAPORTE"
        assert result.confidence >= 0.9

    def test_visa_alias_maps_to_pasaporte(self):
        result = parse_classification_payload(
            {
                "document_type": "VISA",
                "confidence": 0.8,
                "english_text_found": ["UNITED STATES OF AMERICA"],
                "spanish_text_found": [],
                "evidence": [],
                "notes": "",
            }
        )
        assert result.document_type == "PASAPORTE"

    def test_true_ine_spanish_cues_stay_ine(self):
        result = refine_classification(
            DocumentTypeClassification(
                document_type="INE",
                confidence=0.9,
                english_text_found=[],
                spanish_text_found=["Instituto Nacional Electoral", "Credencial para Votar"],
                evidence=["Instituto Nacional Electoral"],
                notes="Mexican voter ID",
            )
        )
        assert result.document_type == "INE"


class TestResolveBeforeCrop:
    def test_english_visa_beats_ine_hint(self):
        classification = DocumentTypeClassification(
            document_type="INE",
            confidence=0.55,
            english_text_found=["UNITED STATES OF AMERICA", "BORDER CROSSING CARD"],
            spanish_text_found=[],
            evidence=["UNITED STATES OF AMERICA"],
            notes="",
        )
        resolved, flags = resolve_document_type(classification, "INE")
        assert resolved == "PASAPORTE"
        assert "document_type_overridden_by_classifier" in flags

    def test_spanish_ine_beats_pasaporte_hint(self):
        classification = DocumentTypeClassification(
            document_type="PASAPORTE",
            confidence=0.5,
            english_text_found=[],
            spanish_text_found=["Credencial para Votar"],
            evidence=["Instituto Nacional Electoral"],
            notes="",
        )
        resolved, flags = resolve_document_type(classification, "PASAPORTE")
        assert resolved == "INE"
        assert "document_type_overridden_by_classifier" in flags

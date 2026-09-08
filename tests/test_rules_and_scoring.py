from datetime import datetime, timedelta

from app.models.responses import Decision, OCRResult, VisionResult
from app.services.rules_engine import _parse_date, rules_engine
from app.services.scoring_service import scoring_service


def test_parse_month_year_expiry_format():
    parsed = _parse_date("2030/12")
    assert parsed is not None
    assert parsed.year == 2030
    assert parsed.month == 12
    assert parsed.day == 31


def test_parse_single_year_expiry():
    parsed = _parse_date("2026")
    assert parsed is not None
    assert parsed.year == 2026
    assert parsed.month == 12
    assert parsed.day == 31


def test_parse_year_range_expiry_uses_end_year():
    parsed = _parse_date("2026-2030")
    assert parsed is not None
    assert parsed.year == 2030
    assert parsed.month == 12
    assert parsed.day == 31


def test_expiry_date_str_normalizes_year_range_to_end_year():
    assert rules_engine.get_expiry_date_str({"expiry_date": "2026-2036"}) == "2036"
    assert rules_engine.get_expiry_date_str({"expiry_date": "2026"}) == "2026"


def test_parse_year_range_expiry_with_spaces_uses_end_year():
    parsed = _parse_date("2023 - 2033")
    assert parsed is not None
    assert parsed.year == 2033
    assert parsed.month == 12
    assert parsed.day == 31


def test_expiry_date_str_normalizes_spaced_year_range_to_end_year():
    assert rules_engine.get_expiry_date_str({"vigencia": "2023 - 2033"}) == "2033"


def test_parse_date_with_time_component():
    parsed = _parse_date("25-04-2026 8:41:22")
    assert parsed is not None
    assert parsed.year == 2026
    assert parsed.month == 4
    assert parsed.day == 25


def test_comprobante_domicilio_uses_address_proof_rules():
    recent = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
    result = rules_engine.validate_receipt(
        OCRResult(
            raw_text="CESPT DOMICILIO AVE EMILIANO ZAPATA 5",
            structured_fields={
                "issuer": "CESPT",
                "street": "AVE EMILIANO ZAPATA 5",
                "colony": "EJIDO FRANCISCO VILLA",
                "issue_date": recent,
            },
            confidence=0.9,
        ),
        "COMPROBANTE_DOMICILIO",
    )

    assert "has_street" in result.passed_rules
    assert "has_colony" in result.passed_rules
    assert "has_total" not in result.failed_rules
    assert "has_receipt_number" not in result.failed_rules
    assert "issue_date_within_3_months" in result.passed_rules


def test_comprobante_expired_still_scores_address_rules_and_rejects():
    old_date = (datetime.now() - timedelta(days=140)).strftime("%Y-%m-%d")
    ocr_result = OCRResult(
        raw_text="CESPT",
        structured_fields={
            "issuer": "CESPT",
            "street": "AVE EMILIANO ZAPATA 5",
            "colony": "EJIDO FRANCISCO VILLA",
            "issue_date": old_date,
        },
        confidence=0.95,
    )
    rules_result = rules_engine.validate_receipt(ocr_result, "COMPROBANTE_DOMICILIO")
    score = scoring_service.calculate_score(
        ocr_result,
        VisionResult(
            is_authentic=True,
            fraud_indicators=[],
            authenticity_score=1.0,
            document_matches_expected_type=True,
            visual_validation_score=1.0,
            quality_flags=[],
            consistency_flags=[],
            notes="ok",
        ),
        rules_result,
    )

    assert "expired_document" in rules_result.flags
    assert "has_street" in rules_result.passed_rules
    assert score.decision == Decision.AUTO_REJECTED


def test_unknown_expiry_is_flagged_without_failing_rule():
    result = rules_engine.validate_identity(
        OCRResult(
            raw_text="NOMBRE: JUAN PEREZ",
            structured_fields={
                "full_name": "JUAN PEREZ",
                "id_number": "ABC123",
                "expiry_date": "vigente",
            },
            confidence=0.9,
        ),
        "INE",
    )

    assert "expiry_not_past" not in result.failed_rules
    assert "unknown_expiry" in result.flags


def test_unknown_expiry_caps_decision_at_human_review():
    ocr_result = OCRResult(
        raw_text="NOMBRE: JUAN PEREZ",
        structured_fields={
            "full_name": "JUAN PEREZ",
            "id_number": "ABC123",
            "expiry_date": "vigente",
        },
        confidence=0.98,
    )
    rules_result = rules_engine.validate_identity(ocr_result, "INE")
    score = scoring_service.calculate_score(
        ocr_result,
        VisionResult(
            is_authentic=True,
            fraud_indicators=[],
            authenticity_score=0.99,
            document_matches_expected_type=True,
            visual_validation_score=0.99,
            quality_flags=[],
            consistency_flags=[],
            notes="ok",
        ),
        rules_result,
        document_type="INE",
    )

    assert score.decision == Decision.HUMAN_REVIEW


def test_expired_document_forces_rejection():
    ocr_result = OCRResult(
        raw_text="NOMBRE: ANA",
        structured_fields={"full_name": "ANA", "id_number": "ABC123", "expiry_date": "2018-12-31"},
        confidence=0.99,
    )
    rules_result = rules_engine.validate_identity(ocr_result, "INE")
    score = scoring_service.calculate_score(
        ocr_result,
        VisionResult(
            is_authentic=True,
            fraud_indicators=[],
            authenticity_score=0.99,
            document_matches_expected_type=True,
            visual_validation_score=0.99,
            quality_flags=[],
            consistency_flags=[],
            notes="ok",
        ),
        rules_result,
        document_type="INE",
    )

    assert "expired_document" in rules_result.flags
    assert score.decision == Decision.AUTO_REJECTED


def test_identity_quality_flags_force_human_review():
    ocr_result = OCRResult(
        raw_text="NOMBRE: JUAN PEREZ",
        structured_fields={
            "full_name": "JUAN PEREZ",
            "id_number": "ABC123",
            "expiry_date": "2030-12-31",
        },
        confidence=0.99,
    )
    rules_result = rules_engine.validate_identity(ocr_result, "INE")
    score = scoring_service.calculate_score(
        ocr_result,
        VisionResult(
            is_authentic=True,
            fraud_indicators=["blurry_image"],
            authenticity_score=0.95,
            document_matches_expected_type=True,
            visual_validation_score=0.65,
            quality_flags=["blurry_image"],
            consistency_flags=[],
            notes="image is blurry",
        ),
        rules_result,
        document_type="INE",
    )

    assert score.decision == Decision.HUMAN_REVIEW

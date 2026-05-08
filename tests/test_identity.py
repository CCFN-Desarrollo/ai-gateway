"""Tests for POST /api/v1/validate/identity"""

import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.errors import ProviderResponseError, UpstreamServiceError
from app.models.responses import (
    Decision,
    IdentityExtractedData,
    IdentityValidationResponse,
    OCRResult,
    RulesResult,
    ScoringResult,
    VisionResult,
)

# ---------------------------------------------------------------------------
# Fixtures — mock service responses
# ---------------------------------------------------------------------------


@pytest.fixture()
def ocr_ine_result() -> OCRResult:
    return OCRResult(
        raw_text=(
            "INSTITUTO NACIONAL ELECTORAL\n"
            "CREDENCIAL PARA VOTAR\n"
            "NOMBRE: JUAN PEREZ GARCIA\n"
            "CURP: PEGJ900101HDFRZN01\n"
            "CLAVE ELECTOR: PRGAJN90010100H600\n"
            "VIGENCIA: 2030/12"
        ),
        structured_fields={
            "full_name": "JUAN PEREZ GARCIA",
            "curp": "PEGJ900101HDFRZN01",
            "id_number": "PRGAJN90010100H600",
            "expiry_date": "2030-12-31",
            "date_of_birth": "1990-01-01",
        },
        confidence=0.94,
    )


@pytest.fixture()
def ocr_expired_ine_result() -> OCRResult:
    return OCRResult(
        raw_text="VIGENCIA: 2018/12\nNOMBRE: ANA LOPEZ RUIZ",
        structured_fields={
            "full_name": "ANA LOPEZ RUIZ",
            "id_number": "LPRANA800101MDFPZN01",
            "expiry_date": "2018-12-31",
        },
        confidence=0.85,
    )


@pytest.fixture()
def ocr_ine_reverso_result() -> OCRResult:
    return OCRResult(
        raw_text="CLAVE DE ELECTOR PRGAJN90010100H600\nCURP PEGJ900101HDFRZN01",
        structured_fields={
            "id_number": "PRGAJN90010100H600",
            "curp": "PEGJ900101HDFRZN01",
        },
        confidence=0.91,
    )


@pytest.fixture()
def vision_authentic_result() -> VisionResult:
    return VisionResult(
        is_authentic=True,
        fraud_indicators=[],
        authenticity_score=0.97,
        document_matches_expected_type=True,
        visual_validation_score=0.97,
        quality_flags=[],
        consistency_flags=[],
        notes="INE image is clear and usable.",
    )


@pytest.fixture()
def vision_fraud_result() -> VisionResult:
    return VisionResult(
        is_authentic=False,
        fraud_indicators=["blurry_image", "document_type_uncertain"],
        authenticity_score=0.35,
        document_matches_expected_type=False,
        visual_validation_score=0.35,
        quality_flags=["blurry_image"],
        consistency_flags=["document_type_uncertain"],
        notes="Image quality is insufficient and document type is uncertain.",
    )


@pytest.fixture()
def rules_pass_result() -> RulesResult:
    return RulesResult(
        passed_rules=["has_full_name", "has_id_number", "expiry_not_past", "has_photo"],
        failed_rules=[],
        rules_score=1.0,
    )


@pytest.fixture()
def rules_expired_result() -> RulesResult:
    return RulesResult(
        passed_rules=["has_full_name", "has_id_number"],
        failed_rules=["expiry_not_past", "has_photo"],
        rules_score=0.5,
    )


@pytest.fixture()
def scoring_approved_result() -> ScoringResult:
    return ScoringResult(
        final_score=96.8,
        decision=Decision.AUTO_APPROVED,
        breakdown={"ocr_confidence": 0.94, "vision_authenticity": 0.97, "rules_score": 1.0},
        requires_human_review=False,
    )


@pytest.fixture()
def scoring_rejected_result() -> ScoringResult:
    return ScoringResult(
        final_score=55.2,
        decision=Decision.AUTO_REJECTED,
        breakdown={"ocr_confidence": 0.85, "vision_authenticity": 0.35, "rules_score": 0.5},
        requires_human_review=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _upload_file(dummy_png: bytes) -> dict:
    return {"file": ("ine.png", io.BytesIO(dummy_png), "image/png")}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIdentityValidationAuth:
    def test_missing_api_key_returns_403(self, client: TestClient, dummy_png: bytes):
        resp = client.post(
            "/api/v1/validate/identity",
            data={"client_id": "client-001", "document_type": "INE"},
            files=_upload_file(dummy_png),
        )
        assert resp.status_code == 403

    def test_invalid_document_type_returns_422(
        self, client: TestClient, api_headers: dict, dummy_png: bytes
    ):
        resp = client.post(
            "/api/v1/validate/identity",
            headers=api_headers,
            data={"client_id": "client-001", "document_type": "CEDULA"},
            files=_upload_file(dummy_png),
        )
        assert resp.status_code == 422

    def test_unsupported_file_type_returns_415(
        self, client: TestClient, api_headers: dict, dummy_png: bytes
    ):
        resp = client.post(
            "/api/v1/validate/identity",
            headers=api_headers,
            data={"client_id": "client-001", "document_type": "INE"},
            files={"file": ("doc.pdf", io.BytesIO(dummy_png), "application/pdf")},
        )
        assert resp.status_code == 415

    def test_gif_content_type_returns_415(
        self, client: TestClient, api_headers: dict, dummy_png: bytes
    ):
        resp = client.post(
            "/api/v1/validate/identity",
            headers=api_headers,
            data={"client_id": "client-001", "document_type": "INE"},
            files={"file": ("doc.gif", io.BytesIO(dummy_png), "image/gif")},
        )
        assert resp.status_code == 415

    def test_oversized_file_returns_413(
        self, client: TestClient, api_headers: dict, large_dummy_png: bytes
    ):
        resp = client.post(
            "/api/v1/validate/identity",
            headers=api_headers,
            data={"client_id": "client-001", "document_type": "INE"},
            files={"file": ("doc.png", io.BytesIO(large_dummy_png), "image/png")},
        )
        assert resp.status_code == 413


class TestIdentityValidationSuccess:
    def test_auto_approved_ine(
        self,
        client: TestClient,
        api_headers: dict,
        dummy_png: bytes,
        ocr_ine_result: OCRResult,
        vision_authentic_result: VisionResult,
        rules_pass_result: RulesResult,
        scoring_approved_result: ScoringResult,
    ):
        with (
            patch(
                "app.pipelines.identity_pipeline.document_preprocessor.preprocess_identity_document",
                return_value=type(
                    "PreprocessedStub",
                    (),
                    {
                        "image_bytes": dummy_png,
                        "quality_flags": [],
                        "used_specialized_crop": False,
                        "debug_image_path": None,
                    },
                )(),
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.ocr_service.extract_text",
                new_callable=AsyncMock,
                return_value=ocr_ine_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.vision_service.analyze_document",
                new_callable=AsyncMock,
                return_value=vision_authentic_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.rules_engine.validate_identity",
                return_value=rules_pass_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.scoring_service.calculate_score",
                return_value=scoring_approved_result,
            ),
        ):
            resp = client.post(
                "/api/v1/validate/identity",
                headers=api_headers,
                data={"client_id": "client-001", "document_type": "INE"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] == "AUTO_APPROVED"
        assert body["requires_human_review"] is False
        assert body["document_type"] == "INE"
        assert body["is_expired"] is False
        assert body["extracted_data"]["full_name"] == "JUAN PEREZ GARCIA"
        assert body["extracted_data"]["curp"] == "PEGJ900101HDFRZN01"
        assert body["quality_flags"] == []
        assert body["consistency_flags"] == []

    def test_auto_rejected_with_quality_issues(
        self,
        client: TestClient,
        api_headers: dict,
        dummy_png: bytes,
        ocr_expired_ine_result: OCRResult,
        vision_fraud_result: VisionResult,
        rules_expired_result: RulesResult,
        scoring_rejected_result: ScoringResult,
    ):
        with (
            patch(
                "app.pipelines.identity_pipeline.document_preprocessor.preprocess_identity_document",
                return_value=type(
                    "PreprocessedStub",
                    (),
                    {
                        "image_bytes": dummy_png,
                        "quality_flags": [],
                        "used_specialized_crop": False,
                        "debug_image_path": None,
                    },
                )(),
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.ocr_service.extract_text",
                new_callable=AsyncMock,
                return_value=ocr_expired_ine_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.vision_service.analyze_document",
                new_callable=AsyncMock,
                return_value=vision_fraud_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.rules_engine.validate_identity",
                return_value=rules_expired_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.scoring_service.calculate_score",
                return_value=scoring_rejected_result,
            ),
        ):
            resp = client.post(
                "/api/v1/validate/identity",
                headers=api_headers,
                data={"client_id": "client-002", "document_type": "INE"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] == "AUTO_REJECTED"
        assert body["is_expired"] is True
        assert body["quality_flags"] == []
        assert body["consistency_flags"] == []

    def test_ine_reverso_uses_preprocessed_crop(
        self,
        client: TestClient,
        api_headers: dict,
        dummy_png: bytes,
        ocr_ine_reverso_result: OCRResult,
        vision_authentic_result: VisionResult,
        rules_pass_result: RulesResult,
        scoring_approved_result: ScoringResult,
    ):
        cropped_bytes = b"cropped-image"

        with (
            patch(
                "app.pipelines.identity_pipeline.document_preprocessor.preprocess_identity_document",
                return_value=type(
                    "PreprocessedStub",
                    (),
                    {
                        "image_bytes": cropped_bytes,
                        "quality_flags": [],
                        "used_specialized_crop": True,
                        "debug_image_path": None,
                    },
                )(),
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.ocr_service.extract_text",
                new_callable=AsyncMock,
                return_value=ocr_ine_reverso_result,
            ) as ocr_mock,
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.vision_service.analyze_document",
                new_callable=AsyncMock,
                return_value=vision_authentic_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.rules_engine.validate_identity",
                return_value=rules_pass_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.scoring_service.calculate_score",
                return_value=scoring_approved_result,
            ),
        ):
            resp = client.post(
                "/api/v1/validate/identity",
                headers=api_headers,
                data={"client_id": "client-004", "document_type": "INE_REVERSO"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 200
        ocr_mock.assert_awaited_once_with(
            cropped_bytes,
            "image/png",
            document_type="INE_REVERSO",
        )
        body = resp.json()
        assert body["used_specialized_crop"] is True

    def test_ine_reverso_falls_back_when_preprocessing_fails(
        self,
        client: TestClient,
        api_headers: dict,
        dummy_png: bytes,
        ocr_ine_reverso_result: OCRResult,
        vision_authentic_result: VisionResult,
        rules_pass_result: RulesResult,
        scoring_approved_result: ScoringResult,
    ):
        with (
            patch(
                "app.pipelines.identity_pipeline.document_preprocessor.preprocess_identity_document",
                return_value=type(
                    "PreprocessedStub",
                    (),
                    {
                        "image_bytes": dummy_png,
                        "quality_flags": ["document_alignment_failed"],
                        "used_specialized_crop": False,
                        "debug_image_path": None,
                    },
                )(),
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.ocr_service.extract_text",
                new_callable=AsyncMock,
                return_value=ocr_ine_reverso_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.vision_service.analyze_document",
                new_callable=AsyncMock,
                return_value=vision_authentic_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.rules_engine.validate_identity",
                return_value=rules_pass_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.scoring_service.calculate_score",
                return_value=scoring_approved_result,
            ),
        ):
            resp = client.post(
                "/api/v1/validate/identity",
                headers=api_headers,
                data={"client_id": "client-005", "document_type": "INE_REVERSO"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["used_specialized_crop"] is False
        assert "document_alignment_failed" in body["quality_flags"]

    def test_ine_reverso_returns_only_id_data(
        self,
        client: TestClient,
        api_headers: dict,
        dummy_png: bytes,
        ocr_ine_reverso_result: OCRResult,
        vision_authentic_result: VisionResult,
        scoring_approved_result: ScoringResult,
    ):
        rules_result = RulesResult(
            passed_rules=["has_id_number"],
            failed_rules=[],
            rules_score=1.0,
        )
        with (
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.ocr_service.extract_text",
                new_callable=AsyncMock,
                return_value=ocr_ine_reverso_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.vision_service.analyze_document",
                new_callable=AsyncMock,
                return_value=vision_authentic_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.rules_engine.validate_identity",
                return_value=rules_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.scoring_service.calculate_score",
                return_value=scoring_approved_result,
            ),
        ):
            resp = client.post(
                "/api/v1/validate/identity",
                headers=api_headers,
                data={"client_id": "client-004", "document_type": "INE_REVERSO"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["document_type"] == "INE_REVERSO"
        assert body["extracted_data"]["id_number"] == "PRGAJN90010100H600"
        assert body["extracted_data"]["full_name"] is None
        assert body["extracted_data"]["expiry_date"] is None

    def test_pasaporte_document_type(
        self,
        client: TestClient,
        api_headers: dict,
        dummy_png: bytes,
        ocr_ine_result: OCRResult,
        vision_authentic_result: VisionResult,
        rules_pass_result: RulesResult,
        scoring_approved_result: ScoringResult,
    ):
        with (
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.ocr_service.extract_text",
                new_callable=AsyncMock,
                return_value=ocr_ine_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.vision_service.analyze_document",
                new_callable=AsyncMock,
                return_value=vision_authentic_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.rules_engine.validate_identity",
                return_value=rules_pass_result,
            ),
            patch(
                "app.pipelines.identity_pipeline.identity_pipeline.scoring_service.calculate_score",
                return_value=scoring_approved_result,
            ),
        ):
            resp = client.post(
                "/api/v1/validate/identity",
                headers=api_headers,
                data={"client_id": "client-003", "document_type": "PASAPORTE"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 200
        assert resp.json()["document_type"] == "PASAPORTE"

    def test_provider_response_error_returns_502(
        self, client: TestClient, api_headers: dict, dummy_png: bytes
    ):
        with patch(
            "app.pipelines.identity_pipeline.identity_pipeline.ocr_service.extract_text",
            new_callable=AsyncMock,
            side_effect=ProviderResponseError("bad json"),
        ):
            resp = client.post(
                "/api/v1/validate/identity",
                headers=api_headers,
                data={"client_id": "client-001", "document_type": "INE"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 502

    def test_upstream_error_returns_503(
        self, client: TestClient, api_headers: dict, dummy_png: bytes
    ):
        with patch(
            "app.pipelines.identity_pipeline.identity_pipeline.ocr_service.extract_text",
            new_callable=AsyncMock,
            side_effect=UpstreamServiceError("timeout"),
        ):
            resp = client.post(
                "/api/v1/validate/identity",
                headers=api_headers,
                data={"client_id": "client-001", "document_type": "INE"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 503

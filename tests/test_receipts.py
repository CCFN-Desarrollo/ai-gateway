"""Tests for POST /api/v1/validate/receipt"""

import io
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.errors import ProviderResponseError, UpstreamServiceError
from app.models.responses import (
    Decision,
    OCRResult,
    ReceiptExtractedData,
    ReceiptValidationResponse,
    RulesResult,
    ScoringResult,
    VisionResult,
)

# ---------------------------------------------------------------------------
# Fixtures — mock service responses
# ---------------------------------------------------------------------------


@pytest.fixture()
def ocr_receipt_result() -> OCRResult:
    return OCRResult(
        raw_text="OXXO SA de CV\nFolio: 12345\nFecha: 2024-03-15\nTotal: $150.00",
        structured_fields={
            "issuer": "OXXO SA de CV",
            "receipt_number": "12345",
            "date": "2024-03-15",
            "total": "150.00",
        },
        confidence=0.92,
    )


@pytest.fixture()
def ocr_address_proof_result() -> OCRResult:
    recent_issue_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    return OCRResult(
        raw_text=(
            f"CFE\nCALLE REFORMA 123\nCOL. CENTRO\nCIUDAD DE MEXICO\nCDMX\nCP 06600\n"
            f"FECHA: {recent_issue_date}"
        ),
        structured_fields={
            "street": "CALLE REFORMA 123",
            "colony": "COL. CENTRO",
            "zip_code": "06600",
            "city": "CIUDAD DE MEXICO",
            "state": "CDMX",
            "issuer": "CFE",
            "issue_date": recent_issue_date,
        },
        confidence=0.94,
    )


@pytest.fixture()
def vision_authentic_result() -> VisionResult:
    return VisionResult(
        is_authentic=True,
        fraud_indicators=[],
        authenticity_score=0.95,
        notes="Document appears genuine with no tampering detected.",
    )


@pytest.fixture()
def rules_pass_result() -> RulesResult:
    return RulesResult(
        passed_rules=["has_date", "has_total", "has_issuer", "has_receipt_number"],
        failed_rules=[],
        rules_score=1.0,
    )


@pytest.fixture()
def scoring_approved_result() -> ScoringResult:
    return ScoringResult(
        final_score=96.3,
        decision=Decision.AUTO_APPROVED,
        breakdown={
            "ocr_confidence": 0.92,
            "vision_authenticity": 0.95,
            "rules_score": 1.0,
        },
        requires_human_review=False,
    )


@pytest.fixture()
def scoring_review_result() -> ScoringResult:
    return ScoringResult(
        final_score=78.5,
        decision=Decision.HUMAN_REVIEW,
        breakdown={
            "ocr_confidence": 0.72,
            "vision_authenticity": 0.80,
            "rules_score": 0.75,
        },
        requires_human_review=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _upload_file(dummy_png: bytes) -> dict:
    return {"file": ("receipt.png", io.BytesIO(dummy_png), "image/png")}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReceiptValidationAuth:
    def test_missing_api_key_returns_403(self, client: TestClient, dummy_png: bytes):
        resp = client.post(
            "/api/v1/validate/receipt",
            data={"client_id": "client-001", "source": "manual"},
            files=_upload_file(dummy_png),
        )
        assert resp.status_code == 403

    def test_invalid_api_key_returns_403(self, client: TestClient, dummy_png: bytes):
        resp = client.post(
            "/api/v1/validate/receipt",
            headers={"X-API-Key": "wrong-key"},
            data={"client_id": "client-001", "source": "manual"},
            files=_upload_file(dummy_png),
        )
        assert resp.status_code == 403

    def test_unsupported_content_type_returns_415(
        self, client: TestClient, api_headers: dict, dummy_png: bytes
    ):
        resp = client.post(
            "/api/v1/validate/receipt",
            headers=api_headers,
            data={"client_id": "client-001"},
            files={"file": ("doc.pdf", io.BytesIO(dummy_png), "application/pdf")},
        )
        assert resp.status_code == 415

    def test_gif_content_type_returns_415(
        self, client: TestClient, api_headers: dict, dummy_png: bytes
    ):
        resp = client.post(
            "/api/v1/validate/receipt",
            headers=api_headers,
            data={"client_id": "client-001"},
            files={"file": ("doc.gif", io.BytesIO(dummy_png), "image/gif")},
        )
        assert resp.status_code == 415

    def test_oversized_file_returns_413(
        self, client: TestClient, api_headers: dict, large_dummy_png: bytes
    ):
        resp = client.post(
            "/api/v1/validate/receipt",
            headers=api_headers,
            data={"client_id": "client-001"},
            files={"file": ("doc.png", io.BytesIO(large_dummy_png), "image/png")},
        )
        assert resp.status_code == 413


class TestReceiptValidationSuccess:
    def test_auto_approved_response_shape(
        self,
        client: TestClient,
        api_headers: dict,
        dummy_png: bytes,
        ocr_receipt_result: OCRResult,
        vision_authentic_result: VisionResult,
        rules_pass_result: RulesResult,
        scoring_approved_result: ScoringResult,
    ):
        with (
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.ocr_service.extract_text",
                new_callable=AsyncMock,
                return_value=ocr_receipt_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.vision_service.analyze_document",
                new_callable=AsyncMock,
                return_value=vision_authentic_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.rules_engine.validate_receipt",
                return_value=rules_pass_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.scoring_service.calculate_score",
                return_value=scoring_approved_result,
            ),
        ):
            resp = client.post(
                "/api/v1/validate/receipt",
                headers=api_headers,
                data={"client_id": "client-001", "source": "whatsapp"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] == "AUTO_APPROVED"
        assert body["requires_human_review"] is False
        assert body["document_type"] == "RECEIPT"
        assert "request_id" in body
        assert "timestamp" in body
        assert "processing_time_ms" in body
        assert "extracted_data" in body
        assert body["extracted_data"]["issuer"] == "OXXO SA de CV"
        assert body["extracted_data"]["total"] == "150.00"
        assert body["is_expired"] is False

    def test_human_review_decision(
        self,
        client: TestClient,
        api_headers: dict,
        dummy_png: bytes,
        ocr_receipt_result: OCRResult,
        vision_authentic_result: VisionResult,
        rules_pass_result: RulesResult,
        scoring_review_result: ScoringResult,
    ):
        with (
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.ocr_service.extract_text",
                new_callable=AsyncMock,
                return_value=ocr_receipt_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.vision_service.analyze_document",
                new_callable=AsyncMock,
                return_value=vision_authentic_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.rules_engine.validate_receipt",
                return_value=rules_pass_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.scoring_service.calculate_score",
                return_value=scoring_review_result,
            ),
        ):
            resp = client.post(
                "/api/v1/validate/receipt",
                headers=api_headers,
                data={"client_id": "client-001", "source": "crm"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] == "HUMAN_REVIEW"
        assert body["requires_human_review"] is True

    def test_address_proof_response_shape(
        self,
        client: TestClient,
        api_headers: dict,
        dummy_png: bytes,
        ocr_address_proof_result: OCRResult,
        vision_authentic_result: VisionResult,
        scoring_approved_result: ScoringResult,
    ):
        rules_result = RulesResult(
            passed_rules=[
                "has_issue_date",
                "has_issuer",
                "has_street",
                "has_colony",
                "has_zip_code",
                "has_city",
                "has_state",
                "issue_date_within_3_months",
            ],
            failed_rules=[],
            rules_score=1.0,
        )
        with (
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.ocr_service.extract_text",
                new_callable=AsyncMock,
                return_value=ocr_address_proof_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.vision_service.analyze_document",
                new_callable=AsyncMock,
                return_value=vision_authentic_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.rules_engine.validate_receipt",
                return_value=rules_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.scoring_service.calculate_score",
                return_value=scoring_approved_result,
            ),
        ):
            resp = client.post(
                "/api/v1/validate/receipt",
                headers=api_headers,
                data={
                    "client_id": "client-002",
                    "source": "manual",
                    "document_type": "ADDRESS_PROOF",
                },
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["document_type"] == "ADDRESS_PROOF"
        assert body["extracted_data"]["street"] == "CALLE REFORMA 123"
        assert body["extracted_data"]["colony"] == "COL. CENTRO"
        assert body["extracted_data"]["zip_code"] == "06600"
        assert body["extracted_data"]["city"] == "CIUDAD DE MEXICO"
        assert body["extracted_data"]["state"] == "CDMX"
        assert body["extracted_data"]["issuer"] == "CFE"
        assert body["extracted_data"]["issue_date"] is not None
        assert body["is_expired"] is False

    def test_jpeg_is_accepted(
        self,
        client: TestClient,
        api_headers: dict,
        dummy_png: bytes,
        ocr_receipt_result: OCRResult,
        vision_authentic_result: VisionResult,
        rules_pass_result: RulesResult,
        scoring_approved_result: ScoringResult,
    ):
        with (
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.ocr_service.extract_text",
                new_callable=AsyncMock,
                return_value=ocr_receipt_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.vision_service.analyze_document",
                new_callable=AsyncMock,
                return_value=vision_authentic_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.rules_engine.validate_receipt",
                return_value=rules_pass_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.scoring_service.calculate_score",
                return_value=scoring_approved_result,
            ),
        ):
            resp = client.post(
                "/api/v1/validate/receipt",
                headers=api_headers,
                data={"client_id": "client-001", "source": "manual"},
                files={"file": ("receipt.jpg", io.BytesIO(dummy_png), "image/jpeg")},
            )

        assert resp.status_code == 200

    def test_webp_is_accepted(
        self,
        client: TestClient,
        api_headers: dict,
        dummy_png: bytes,
        ocr_receipt_result: OCRResult,
        vision_authentic_result: VisionResult,
        rules_pass_result: RulesResult,
        scoring_approved_result: ScoringResult,
    ):
        with (
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.ocr_service.extract_text",
                new_callable=AsyncMock,
                return_value=ocr_receipt_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.vision_service.analyze_document",
                new_callable=AsyncMock,
                return_value=vision_authentic_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.rules_engine.validate_receipt",
                return_value=rules_pass_result,
            ),
            patch(
                "app.pipelines.receipt_pipeline.receipt_pipeline.scoring_service.calculate_score",
                return_value=scoring_approved_result,
            ),
        ):
            resp = client.post(
                "/api/v1/validate/receipt",
                headers=api_headers,
                data={"client_id": "client-001", "source": "manual"},
                files={"file": ("receipt.webp", io.BytesIO(dummy_png), "image/webp")},
            )

        assert resp.status_code == 200

    def test_pipeline_error_returns_500(
        self, client: TestClient, api_headers: dict, dummy_png: bytes
    ):
        with patch(
            "app.pipelines.receipt_pipeline.receipt_pipeline.ocr_service.extract_text",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Claude API unavailable"),
        ):
            resp = client.post(
                "/api/v1/validate/receipt",
                headers=api_headers,
                data={"client_id": "client-001"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 500

    def test_provider_response_error_returns_502(
        self, client: TestClient, api_headers: dict, dummy_png: bytes
    ):
        with patch(
            "app.pipelines.receipt_pipeline.receipt_pipeline.ocr_service.extract_text",
            new_callable=AsyncMock,
            side_effect=ProviderResponseError("bad json"),
        ):
            resp = client.post(
                "/api/v1/validate/receipt",
                headers=api_headers,
                data={"client_id": "client-001"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 502

    def test_upstream_error_returns_503(
        self, client: TestClient, api_headers: dict, dummy_png: bytes
    ):
        with patch(
            "app.pipelines.receipt_pipeline.receipt_pipeline.ocr_service.extract_text",
            new_callable=AsyncMock,
            side_effect=UpstreamServiceError("timeout"),
        ):
            resp = client.post(
                "/api/v1/validate/receipt",
                headers=api_headers,
                data={"client_id": "client-001"},
                files=_upload_file(dummy_png),
            )

        assert resp.status_code == 503


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "uptime_seconds" in body

import base64
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


class _PipelineResponseStub:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def model_dump_json(self) -> str:
        return self._payload


def test_create_validation_case_queues_case(client: TestClient, api_headers: dict):
    payload = {
        "client_id": "0001231",
        "channel": "telegram",
        "chat_id": "123456",
        "documents": [
            {
                "document_type": "INE",
                "file_name": "ine.jpg",
                "content_type": "image/jpeg",
                "content_base64": base64.b64encode(b"fake-image").decode(),
            }
        ],
    }

    response = client.post("/api/v1/validation-cases", headers=api_headers, json=payload)

    assert response.status_code == 202
    body = response.json()
    assert body["case_id"].startswith("case_")
    assert body["status"] == "QUEUED"


def test_get_validation_case_returns_processed_status(client: TestClient, api_headers: dict):
    payload = {
        "client_id": "0001231",
        "channel": "telegram",
        "chat_id": "123456",
        "documents": [
            {
                "document_type": "INE",
                "file_name": "ine.jpg",
                "content_type": "image/jpeg",
                "content_base64": base64.b64encode(b"fake-image").decode(),
            },
            {
                "document_type": "ADDRESS_PROOF",
                "file_name": "address.jpg",
                "content_type": "image/jpeg",
                "content_base64": base64.b64encode(b"fake-image-2").decode(),
            },
        ],
    }

    identity_response = _PipelineResponseStub(
        '{"decision":"AUTO_APPROVED","extracted_data":{"full_name":"TEST USER","id_number":"ABC123",'
        '"curp":"CURP123","expiry_date":"2030-12-31","date_of_birth":"1990-01-01"},'
        '"is_expired":false}'
    )
    receipt_response = _PipelineResponseStub(
        '{"decision":"AUTO_APPROVED","extracted_data":{"street":"MAIN 123","colony":"CENTRO",'
        '"zip_code":"12345","city":"MEXICALI","state":"BC","issue_date":"2025-05-26"},'
        '"is_expired":false}'
    )

    with (
        patch(
            "app.services.validation_case_service.identity_pipeline.process",
            new=AsyncMock(return_value=identity_response),
        ),
        patch(
            "app.services.validation_case_service.receipt_pipeline.process",
            new=AsyncMock(return_value=receipt_response),
        ),
    ):
        create_response = client.post(
            "/api/v1/validation-cases",
            headers=api_headers,
            json=payload,
        )
        assert create_response.status_code == 202
        case_id = create_response.json()["case_id"]

        for _ in range(20):
            get_response = client.get(f"/api/v1/validation-cases/{case_id}", headers=api_headers)
            assert get_response.status_code == 200
            body = get_response.json()
            if body["status"] != "QUEUED" and body["status"] != "PROCESSING":
                break

        assert body["status"] == "WAITING_AUTHORIZATION"
        assert body["authorization_status"] == "PENDING"
        assert body["consolidated_data"]["full_name"] == "TEST USER"
        assert body["consolidated_data"]["street"] == "MAIN 123"

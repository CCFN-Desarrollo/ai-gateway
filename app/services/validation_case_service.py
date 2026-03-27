import asyncio
import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.core.errors import ProviderResponseError, UpstreamServiceError
from app.models.requests import (
    AuthorizationStatus,
    ValidationCaseCreateRequest,
    ValidationCaseStatus,
    ValidationDocumentStatus,
)
from app.models.responses import (
    ConsolidatedValidationData,
    ValidationCaseCreateResponse,
    ValidationCaseResponse,
    ValidationDocumentSummary,
)
from app.pipelines.identity_pipeline import identity_pipeline
from app.pipelines.receipt_pipeline import receipt_pipeline

logger = logging.getLogger(__name__)

_IDENTITY_DOCUMENT_TYPES = {"INE", "INE_REVERSO"}
_SUPPORTED_CASE_DOCUMENT_TYPES = {"INE", "INE_REVERSO", "ADDRESS_PROOF"}


class ValidationCaseService:
    def __init__(self, base_dir: str) -> None:
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, asyncio.Task] = {}

    async def create_case(
        self, payload: ValidationCaseCreateRequest
    ) -> ValidationCaseCreateResponse:
        if not payload.documents:
            raise ValueError("At least one document is required.")

        case_id = f"case_{uuid4().hex[:12]}"
        case_dir = self.base_path / case_id
        files_dir = case_dir / "documents"
        files_dir.mkdir(parents=True, exist_ok=True)

        documents = []
        now = self._utc_now_iso()
        for document in payload.documents:
            document_type = document.document_type.strip().upper()
            if document_type not in _SUPPORTED_CASE_DOCUMENT_TYPES:
                raise ValueError(
                    f"Unsupported document_type '{document.document_type}'. "
                    "Use INE, INE_REVERSO or ADDRESS_PROOF."
                )
            document_id = f"doc_{uuid4().hex[:12]}"
            file_path = files_dir / f"{document_id}-{document.file_name}"
            file_path.write_bytes(base64.b64decode(document.content_base64))
            documents.append(
                {
                    "document_id": document_id,
                    "document_type": document_type,
                    "file_name": document.file_name,
                    "content_type": document.content_type,
                    "file_path": str(file_path),
                    "status": ValidationDocumentStatus.PENDING.value,
                    "error": None,
                    "result": None,
                }
            )

        case_record = {
            "case_id": case_id,
            "client_id": payload.client_id,
            "channel": payload.channel.value,
            "chat_id": payload.chat_id,
            "status": ValidationCaseStatus.QUEUED.value,
            "authorization_status": AuthorizationStatus.PENDING.value,
            "rejection_reason_code": None,
            "rejection_reason_text": None,
            "documents": documents,
            "consolidated_data": ConsolidatedValidationData().model_dump(),
            "created_at": now,
            "updated_at": now,
        }
        self._write_case(case_record)
        self._tasks[case_id] = asyncio.create_task(self._process_case(case_id))
        return ValidationCaseCreateResponse(
            case_id=case_id,
            status=ValidationCaseStatus.QUEUED,
        )

    async def get_case(self, case_id: str) -> ValidationCaseResponse | None:
        record = self._read_case(case_id)
        if record is None:
            return None
        return self._to_response(record)

    async def _process_case(self, case_id: str) -> None:
        record = self._read_case(case_id)
        if record is None:
            return
        self._update_case(record, status=ValidationCaseStatus.PROCESSING.value)

        had_failure = False
        for document in record["documents"]:
            document["status"] = ValidationDocumentStatus.PROCESSING.value
            document["error"] = None
            self._touch(record)
            try:
                result = await self._process_document(record, document)
                document["status"] = ValidationDocumentStatus.DONE.value
                document["result"] = result
            except (UpstreamServiceError, ProviderResponseError, ValueError) as exc:
                had_failure = True
                document["status"] = ValidationDocumentStatus.FAILED.value
                document["error"] = str(exc)
                logger.warning(
                    "Validation case %s document %s failed: %s",
                    case_id,
                    document["document_id"],
                    exc,
                )
            self._touch(record)

        self._consolidate_case(record)
        if had_failure and not any(
            doc["status"] == ValidationDocumentStatus.DONE.value for doc in record["documents"]
        ):
            self._update_case(
                record,
                status=ValidationCaseStatus.FAILED.value,
                rejection_reason_code="PROCESSING_FAILED",
                rejection_reason_text="No se pudo procesar ninguno de los documentos del expediente.",
            )
            return

        if self._has_rejection(record):
            self._update_case(
                record,
                status=ValidationCaseStatus.REJECTED.value,
                authorization_status=AuthorizationStatus.REJECTED.value,
            )
            return

        if had_failure:
            self._update_case(
                record,
                status=ValidationCaseStatus.FAILED.value,
                rejection_reason_code="PARTIAL_PROCESSING_FAILED",
                rejection_reason_text="Algunos documentos del expediente no pudieron procesarse.",
            )
            return

        self._update_case(
            record,
            status=ValidationCaseStatus.WAITING_AUTHORIZATION.value,
        )

    async def _process_document(self, record: dict, document: dict) -> dict:
        image_bytes = Path(document["file_path"]).read_bytes()
        metadata = {
            "client_id": record["client_id"],
            "source": record["channel"],
            "document_type": document["document_type"],
        }
        if document["document_type"] in _IDENTITY_DOCUMENT_TYPES:
            response = await identity_pipeline.process(
                image_bytes=image_bytes,
                media_type=document["content_type"],
                metadata=metadata,
            )
        else:
            response = await receipt_pipeline.process(
                image_bytes=image_bytes,
                media_type=document["content_type"],
                metadata=metadata,
            )
        return json.loads(response.model_dump_json())

    def _consolidate_case(self, record: dict) -> None:
        consolidated = ConsolidatedValidationData().model_dump()
        rejection_reason_code = None
        rejection_reason_text = None

        for document in record["documents"]:
            result = document.get("result")
            if not result:
                continue
            extracted = result.get("extracted_data", {})
            if document["document_type"] in _IDENTITY_DOCUMENT_TYPES:
                consolidated["full_name"] = consolidated["full_name"] or extracted.get("full_name")
                consolidated["id_number"] = consolidated["id_number"] or extracted.get("id_number")
                consolidated["curp"] = consolidated["curp"] or extracted.get("curp")
                consolidated["expiry_date"] = consolidated["expiry_date"] or extracted.get(
                    "expiry_date"
                )
                consolidated["date_of_birth"] = consolidated["date_of_birth"] or extracted.get(
                    "date_of_birth"
                )
            elif document["document_type"] == "ADDRESS_PROOF":
                consolidated["street"] = consolidated["street"] or extracted.get("street")
                consolidated["colony"] = consolidated["colony"] or extracted.get("colony")
                consolidated["zip_code"] = consolidated["zip_code"] or extracted.get("zip_code")
                consolidated["city"] = consolidated["city"] or extracted.get("city")
                consolidated["state"] = consolidated["state"] or extracted.get("state")
                consolidated["issue_date"] = consolidated["issue_date"] or extracted.get(
                    "issue_date"
                )
                consolidated["address_is_expired"] = bool(result.get("is_expired", False))
                if result.get("is_expired"):
                    rejection_reason_code = "ADDRESS_PROOF_EXPIRED"
                    rejection_reason_text = (
                        "El comprobante de domicilio tiene una antigüedad mayor a tres meses."
                    )

            if result.get("decision") == "AUTO_REJECTED" and rejection_reason_code is None:
                rejection_reason_code = "DOCUMENT_REJECTED"
                rejection_reason_text = (
                    f"El documento {document['document_type']} fue rechazado por el pipeline."
                )

        record["consolidated_data"] = consolidated
        record["rejection_reason_code"] = rejection_reason_code
        record["rejection_reason_text"] = rejection_reason_text

    @staticmethod
    def _has_rejection(record: dict) -> bool:
        return bool(record.get("rejection_reason_code"))

    def _read_case(self, case_id: str) -> dict | None:
        path = self.base_path / case_id / "case.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _write_case(self, record: dict) -> None:
        case_dir = self.base_path / record["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        path = case_dir / "case.json"
        path.write_text(json.dumps(record, ensure_ascii=True, indent=2))

    def _touch(self, record: dict) -> None:
        record["updated_at"] = self._utc_now_iso()
        self._write_case(record)

    def _update_case(self, record: dict, **changes: str | None) -> None:
        for key, value in changes.items():
            record[key] = value
        self._touch(record)

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_response(record: dict) -> ValidationCaseResponse:
        documents = [
            ValidationDocumentSummary(
                document_id=document["document_id"],
                document_type=document["document_type"],
                file_name=document["file_name"],
                status=ValidationDocumentStatus(document["status"]),
                error=document.get("error"),
                result=document.get("result"),
            )
            for document in record["documents"]
        ]
        return ValidationCaseResponse(
            case_id=record["case_id"],
            client_id=record["client_id"],
            channel=record["channel"],
            chat_id=record.get("chat_id"),
            status=ValidationCaseStatus(record["status"]),
            authorization_status=AuthorizationStatus(record["authorization_status"]),
            rejection_reason_code=record.get("rejection_reason_code"),
            rejection_reason_text=record.get("rejection_reason_text"),
            documents=documents,
            consolidated_data=ConsolidatedValidationData(**record.get("consolidated_data", {})),
            created_at=datetime.fromisoformat(record["created_at"]),
            updated_at=datetime.fromisoformat(record["updated_at"]),
        )


validation_case_service = ValidationCaseService(settings.VALIDATION_CASES_DIR)

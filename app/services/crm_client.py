import base64
import logging
from pathlib import Path

import httpx

from app.core.config import settings
from app.core.errors import CRMClientError, CRMConflictError, CRMValidationError

logger = logging.getLogger(__name__)

_DOCUMENT_TYPE_TO_FILE_TYPE = {
    "INE": "ID1",
    "INE_REVERSO": "ID2",
    "ADDRESS_PROOF": "ADR",
}


class CRMClient:
    def __init__(self, config=settings) -> None:
        self.settings = config
        self.base_url = self.settings.CRM_BASE_URL.rstrip("/")
        self.enabled = bool(
            self.settings.CRM_ENABLED and self.base_url and self.settings.CRM_API_KEY.strip()
        )

    async def sync_case(self, case_record: dict) -> None:
        if not self.enabled:
            logger.info(
                "CRM integration disabled; skipping sync for case_id=%s",
                case_record["case_id"],
            )
            return

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.settings.CRM_TIMEOUT_SECONDS,
        ) as client:
            await self._update_client(client, case_record)
            await self._upload_documents(client, case_record)
            await self._submit_verification(client, case_record["client_id"])

    async def _update_client(self, client: httpx.AsyncClient, case_record: dict) -> None:
        name_parts = _split_name(case_record["consolidated_data"].get("full_name"))
        payload = {
            "CardFName": case_record["consolidated_data"].get("full_name") or "",
            "Phone1": case_record.get("phone") or "",
            "E_Mail": case_record.get("email") or "",
            "Contact": {
                "Name": case_record["consolidated_data"].get("full_name") or "",
                "FirstName": name_parts["first_name"],
                "MiddleName": name_parts["middle_name"],
                "LastName": name_parts["last_name"],
                "identityDocument": case_record["consolidated_data"].get("id_number")
                or case_record["consolidated_data"].get("curp")
                or "",
                "E_Mail": case_record.get("email") or "",
            },
            "Adress": {
                "Street": case_record["consolidated_data"].get("street") or "",
                "Block": case_record["consolidated_data"].get("colony") or "",
                "City": case_record["consolidated_data"].get("city") or "",
                "State": case_record["consolidated_data"].get("state") or "",
                "ZipCode": case_record["consolidated_data"].get("zip_code") or "",
                "Country": "MX",
            },
        }
        response = await client.put(
            f"/api/Contact/Client/{case_record['client_id']}",
            json=payload,
            headers=self._headers(),
        )
        self._raise_for_status(response, "update client")

    async def _upload_documents(self, client: httpx.AsyncClient, case_record: dict) -> None:
        documents_payload = []
        for document in case_record["documents"]:
            if document.get("status") != "DONE":
                continue
            file_type = _DOCUMENT_TYPE_TO_FILE_TYPE.get(document["document_type"])
            if file_type is None:
                continue
            extension = Path(document["file_name"]).suffix.lower().lstrip(".") or "jpg"
            documents_payload.append(
                {
                    "CardCode": case_record["client_id"],
                    "CntctCode": 1,
                    "fileType": file_type,
                    "fileExtension": extension,
                    "Base64Image": base64.b64encode(
                        Path(document["file_path"]).read_bytes()
                    ).decode("ascii"),
                }
            )

        if not documents_payload:
            raise CRMValidationError("No successful documents available to upload to CRM.")

        response = await client.post(
            "/api/Contact/upload",
            json=documents_payload,
            headers=self._headers(),
        )
        self._raise_for_status(response, "upload documents")

    async def _submit_verification(self, client: httpx.AsyncClient, card_code: str) -> None:
        response = await client.post(
            f"/api/Contact/SubmitVerification/{card_code}",
            headers=self._headers(),
        )
        if response.status_code == 409:
            logger.info("CRM verification already in progress for cardCode=%s", card_code)
            return
        if response.status_code == 422:
            body = _safe_json(response)
            missing_fields = body.get("missingFields", [])
            detail = body.get("error") or "CRM rejected the verification payload."
            if missing_fields:
                detail = f"{detail} Missing fields: {', '.join(missing_fields)}"
            raise CRMValidationError(detail)
        self._raise_for_status(response, "submit verification")

    def _headers(self) -> dict[str, str]:
        value = self.settings.CRM_API_KEY.strip()
        if self.settings.CRM_API_KEY_PREFIX.strip():
            value = f"{self.settings.CRM_API_KEY_PREFIX.strip()} {value}"
        return {
            self.settings.CRM_API_KEY_HEADER: value,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _raise_for_status(response: httpx.Response, operation: str) -> None:
        if response.is_success:
            return
        body = _safe_json(response)
        detail = body.get("error") or body.get("message") or response.text
        if response.status_code == 409:
            raise CRMConflictError(detail)
        if response.status_code == 422:
            raise CRMValidationError(detail)
        raise CRMClientError(f"CRM {operation} failed with status {response.status_code}: {detail}")


def _safe_json(response: httpx.Response) -> dict:
    try:
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except ValueError:
        return {}


def _split_name(full_name: str | None) -> dict[str, str]:
    if not full_name:
        return {"first_name": "", "middle_name": "", "last_name": ""}
    parts = [part for part in full_name.split() if part]
    if not parts:
        return {"first_name": "", "middle_name": "", "last_name": ""}
    if len(parts) == 1:
        return {"first_name": parts[0], "middle_name": "", "last_name": ""}
    if len(parts) == 2:
        return {"first_name": parts[0], "middle_name": "", "last_name": parts[1]}
    return {
        "first_name": parts[0],
        "middle_name": " ".join(parts[1:-1]),
        "last_name": parts[-1],
    }


crm_client = CRMClient(settings)

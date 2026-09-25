from enum import Enum

from pydantic import BaseModel


class DocumentSource(str, Enum):
    WHATSAPP = "whatsapp"
    CRM = "crm"
    WEB = "web"
    MANUAL = "manual"


class DocumentType(str, Enum):
    INE = "INE"
    INE_REVERSO = "INE_REVERSO"
    PASAPORTE = "PASAPORTE"
    LICENCIA = "LICENCIA"
    # Temporary request alias only. Not a pipeline document type.
    # Accepted so callers that send OTRO do not 422; identity maps it to INE.
    # Replace this when OTRO gets its own handling.
    OTRO = "OTRO"


class ReceiptDocumentType(str, Enum):
    RECEIPT = "RECEIPT"
    ADDRESS_PROOF = "ADDRESS_PROOF"
    COMPROBANTE_DOMICILIO = "COMPROBANTE_DOMICILIO"


ADDRESS_PROOF_TYPES = {
    ReceiptDocumentType.ADDRESS_PROOF.value,
    ReceiptDocumentType.COMPROBANTE_DOMICILIO.value,
}


def is_address_proof(document_type: str | None) -> bool:
    return document_type in ADDRESS_PROOF_TYPES


class ReceiptValidationRequest(BaseModel):
    client_id: str
    source: DocumentSource = DocumentSource.MANUAL
    document_type: ReceiptDocumentType = ReceiptDocumentType.RECEIPT


class IdentityValidationRequest(BaseModel):
    client_id: str
    document_type: DocumentType | None = None


class ValidationCaseStatus(str, Enum):
    COLLECTING = "COLLECTING"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    WAITING_AUTHORIZATION = "WAITING_AUTHORIZATION"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class ValidationDocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


class AuthorizationStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ValidationChannel(str, Enum):
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    WEB = "web"
    CRM = "crm"
    MANUAL = "manual"


class ValidationCaseDocumentInput(BaseModel):
    document_type: str
    file_name: str
    content_type: str
    content_base64: str


class ValidationCaseCreateRequest(BaseModel):
    client_id: str
    channel: ValidationChannel = ValidationChannel.MANUAL
    chat_id: str | None = None
    phone: str | None = None
    email: str | None = None
    documents: list[ValidationCaseDocumentInput]

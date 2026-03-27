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


class ReceiptDocumentType(str, Enum):
    RECEIPT = "RECEIPT"
    ADDRESS_PROOF = "ADDRESS_PROOF"
    COMPROBANTE_DOMICILIO = "ADDRESS_PROOF"


class ReceiptValidationRequest(BaseModel):
    client_id: str
    source: DocumentSource = DocumentSource.MANUAL
    document_type: ReceiptDocumentType = ReceiptDocumentType.RECEIPT


class IdentityValidationRequest(BaseModel):
    client_id: str
    document_type: DocumentType

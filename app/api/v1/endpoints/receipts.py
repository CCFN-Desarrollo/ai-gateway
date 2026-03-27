import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.v1.uploads import read_limited_upload, validate_image_file
from app.core.config import settings
from app.core.errors import ProviderResponseError, UpstreamServiceError
from app.core.security import verify_api_key
from app.models.requests import DocumentSource, ReceiptDocumentType
from app.models.responses import ReceiptValidationResponse
from app.pipelines.receipt_pipeline import receipt_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_FILE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


@router.post(
    "/receipt",
    response_model=ReceiptValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate a receipt document",
    description=(
        "Upload a receipt or address-proof image (JPEG, PNG, WebP) along with optional metadata. "
        "The gateway runs OCR → Vision AI → Rules Engine → Scoring and returns "
        "a structured validation result with a routing decision."
    ),
    tags=["validation"],
)
async def validate_receipt(
    file: UploadFile = File(..., description="Receipt image file (JPEG, PNG, WebP)"),
    client_id: str = Form(..., description="Identifier of the submitting client"),
    source: DocumentSource = Form(
        DocumentSource.MANUAL, description="Origin channel of the document"
    ),
    document_type: ReceiptDocumentType = Form(
        ReceiptDocumentType.RECEIPT,
        description="Type of receipt document: RECEIPT or ADDRESS_PROOF",
    ),
    _api_key: str = Depends(verify_api_key),
) -> ReceiptValidationResponse:
    """
    Validate a receipt document through the full AI pipeline.

    - **file**: Multipart image file (JPEG / PNG / WebP, max configured MB)
    - **client_id**: Client identifier for traceability
    - **source**: Channel that submitted the document (whatsapp, crm, web, manual)
    - **document_type**: RECEIPT | ADDRESS_PROOF
    """
    validate_image_file(file)

    try:
        image_bytes = await read_limited_upload(file, _MAX_FILE_BYTES, settings.MAX_FILE_SIZE_MB)
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        logger.error("Failed to read uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file.",
        ) from exc

    media_type = file.content_type or "image/jpeg"

    try:
        result = await receipt_pipeline.process(
            image_bytes=image_bytes,
            media_type=media_type,
            metadata={
                "client_id": client_id,
                "source": source.value,
                "document_type": document_type.value,
            },
        )
    except ProviderResponseError as exc:
        logger.warning(
            "Receipt provider returned invalid payload for client_id=%s source=%s document_type=%s: %s",
            client_id,
            source.value,
            document_type.value,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document AI provider returned an invalid response.",
        ) from exc
    except UpstreamServiceError as exc:
        logger.warning(
            "Receipt provider unavailable for client_id=%s source=%s document_type=%s: %s",
            client_id,
            source.value,
            document_type.value,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document AI provider is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Receipt pipeline failed for client_id=%s source=%s document_type=%s",
            client_id,
            source.value,
            document_type.value,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed. Please try again later.",
        ) from exc

    return result

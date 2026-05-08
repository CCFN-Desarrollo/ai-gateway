import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.v1.uploads import read_limited_upload, validate_image_file
from app.core.config import settings
from app.core.errors import ProviderResponseError, UpstreamServiceError
from app.core.security import verify_api_key
from app.models.requests import DocumentType
from app.models.responses import IdentityValidationResponse
from app.pipelines.identity_pipeline import identity_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_FILE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


@router.post(
    "/identity",
    response_model=IdentityValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate an identity document",
    description=(
        "Upload an identity document image (INE, INE_REVERSO, PASAPORTE, LICENCIA) along with its type. "
        "The gateway runs OCR → Vision AI → Rules Engine → Scoring and returns "
        "a structured validation result with routing, expiry status, and image quality checks."
    ),
    tags=["validation"],
)
async def validate_identity(
    file: UploadFile = File(..., description="Identity document image (JPEG, PNG, WebP)"),  # noqa: B008
    client_id: str = Form(..., description="Identifier of the submitting client"),  # noqa: B008
    document_type: DocumentType = Form(  # noqa: B008
        ..., description="Type of identity document: INE, INE_REVERSO, PASAPORTE, LICENCIA"
    ),
    _api_key: str = Depends(verify_api_key),
) -> IdentityValidationResponse:
    """
    Validate an identity document through the full AI pipeline.

    - **file**: Multipart image file (JPEG / PNG / WebP, max configured MB)
    - **client_id**: Client identifier for traceability
    - **document_type**: INE | INE_REVERSO | PASAPORTE | LICENCIA
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
        result = await identity_pipeline.process(
            image_bytes=image_bytes,
            media_type=media_type,
            metadata={"client_id": client_id, "document_type": document_type.value},
        )
    except ProviderResponseError as exc:
        logger.warning(
            "Identity provider returned invalid payload for client_id=%s document_type=%s: %s",
            client_id,
            document_type.value,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document AI provider returned an invalid response.",
        ) from exc
    except UpstreamServiceError as exc:
        logger.warning(
            "Identity provider unavailable for client_id=%s document_type=%s: %s",
            client_id,
            document_type.value,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document AI provider is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Identity pipeline failed for client_id=%s document_type=%s",
            client_id,
            document_type.value,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed. Please try again later.",
        ) from exc

    return result

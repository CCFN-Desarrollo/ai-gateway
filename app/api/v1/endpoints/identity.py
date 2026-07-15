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
        "Upload an identity document image (INE, INE_REVERSO, PASAPORTE, LICENCIA). "
        "When document_type is provided, the existing typed pipeline runs unchanged. "
        "When omitted, an alternate flow classifies the type from image content only "
        "(filenames are ignored), then continues with the same pipeline."
    ),
    tags=["validation"],
)
async def validate_identity(
    file: UploadFile = File(..., description="Identity document image (JPEG, PNG, WebP)"),  # noqa: B008
    client_id: str = Form(..., description="Identifier of the submitting client"),  # noqa: B008
    document_type: DocumentType | None = Form(  # noqa: B008
        None,
        description=(
            "Optional. When set: INE | INE_REVERSO | PASAPORTE | LICENCIA (existing flow). "
            "When omitted: classify from image content (alternate flow; ignore filename)."
        ),
    ),
    _api_key: str = Depends(verify_api_key),
) -> IdentityValidationResponse:
    """
    Validate an identity document through the full AI pipeline.

    - **file**: Multipart image file (JPEG / PNG / WebP, max configured MB)
    - **client_id**: Client identifier for traceability
    - **document_type**: optional; omit to auto-detect from the image
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
    hinted_type = document_type.value if document_type is not None else None

    try:
        if hinted_type is None:
            # Alternate flow: classify from image, then run the existing typed pipeline.
            result = await identity_pipeline.process_with_auto_detect(
                image_bytes=image_bytes,
                media_type=media_type,
                metadata={"client_id": client_id},
            )
        else:
            result = await identity_pipeline.process(
                image_bytes=image_bytes,
                media_type=media_type,
                metadata={"client_id": client_id, "document_type": hinted_type},
            )
    except ProviderResponseError as exc:
        logger.warning(
            "Identity provider returned invalid payload for client_id=%s document_type=%s: %s",
            client_id,
            hinted_type,
            exc,
        )
        detail = (
            "Document AI provider refused to process this identity document."
            if "refused" in str(exc).lower()
            else "Document AI provider returned an invalid response."
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        ) from exc
    except UpstreamServiceError as exc:
        logger.warning(
            "Identity provider unavailable for client_id=%s document_type=%s: %s",
            client_id,
            hinted_type,
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
            hinted_type,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed. Please try again later.",
        ) from exc

    return result

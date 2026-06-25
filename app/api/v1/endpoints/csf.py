import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.v1.uploads import read_limited_upload, validate_image_file
from app.core.config import settings
from app.core.errors import ProviderResponseError, UpstreamServiceError
from app.core.security import verify_api_key
from app.models.requests import DocumentSource
from app.models.responses import CsfValidationResponse
from app.pipelines.csf_pipeline import csf_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_FILE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


@router.post(
    "/csf",
    response_model=CsfValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate a Constancia de Situación Fiscal",
    description=(
        "Upload one or more JPG/PNG images of a Constancia de Situación Fiscal (SAT). "
        "The gateway runs OCR on all pages in parallel, merges the extracted fields, "
        "applies RFC/fiscal rules, and returns a structured validation result."
    ),
    tags=["validation"],
)
async def validate_csf(
    files: list[UploadFile] = File(..., description="One or more CSF page images (JPEG, PNG, WebP)"),  # noqa: B008
    client_id: str = Form(..., description="Identifier of the submitting client"),  # noqa: B008
    source: DocumentSource = Form(  # noqa: B008
        DocumentSource.MANUAL, description="Origin channel of the document"
    ),
    _api_key: str = Depends(verify_api_key),
) -> CsfValidationResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file is required.",
        )

    for file in files:
        validate_image_file(file)

    pages: list[tuple[bytes, str]] = []
    for file in files:
        try:
            image_bytes = await read_limited_upload(file, _MAX_FILE_BYTES, settings.MAX_FILE_SIZE_MB)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to read uploaded file %s: %s", file.filename, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not read one of the uploaded files.",
            ) from exc
        pages.append((image_bytes, file.content_type or "image/jpeg"))

    try:
        result = await csf_pipeline.process(
            pages=pages,
            metadata={"client_id": client_id, "source": source.value},
        )
    except ProviderResponseError as exc:
        logger.warning("CSF provider invalid payload for client_id=%s: %s", client_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document AI provider returned an invalid response.",
        ) from exc
    except UpstreamServiceError as exc:
        logger.warning("CSF provider unavailable for client_id=%s: %s", client_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document AI provider is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("CSF pipeline failed for client_id=%s", client_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed. Please try again later.",
        ) from exc

    return result

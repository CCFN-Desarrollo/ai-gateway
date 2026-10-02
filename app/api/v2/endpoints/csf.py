import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.v2.uploads import expand_csf_upload
from app.core.config import settings
from app.core.errors import ProviderResponseError, UpstreamServiceError
from app.core.security import verify_api_key
from app.models.requests import DocumentSource
from app.models.responses import CsfV2ValidationResponse
from app.pipelines.csf_v2_pipeline import csf_v2_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_FILE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


@router.post(
    "/csf",
    response_model=CsfV2ValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate a Constancia de Situación Fiscal (v2)",
    description=(
        "Upload a CSF page 1 file, a page 2 file, or both. "
        "A multi-page PDF is rasterized in full (capped): the first page uses the "
        "existing CSF extraction and later pages extract Régimen and Actividad Económica. "
        "An image or single-page PDF keeps the slot it was uploaded in. "
        "The response includes page1, page2, or both; the missing section is null."
    ),
    tags=["validation"],
)
async def validate_csf_v2(
    page1: UploadFile | None = File(None, description="CSF page 1 image, or a multi-page CSF PDF"),  # noqa: B008
    page2: UploadFile | None = File(None, description="CSF page 2 image (Régimen / Actividad Económica)"),  # noqa: B008
    client_id: str = Form(..., description="Identifier of the submitting client"),  # noqa: B008
    source: DocumentSource = Form(  # noqa: B008
        DocumentSource.MANUAL, description="Origin channel of the document"
    ),
    _api_key: str = Depends(verify_api_key),
) -> CsfV2ValidationResponse:
    if page1 is None and page2 is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of page1 or page2 is required.",
        )

    pages: list[tuple[bytes, str, str]] = []
    for slot, upload in (("page1", page1), ("page2", page2)):
        if upload is None:
            continue
        try:
            pages.extend(
                await expand_csf_upload(
                    upload, slot, _MAX_FILE_BYTES, settings.MAX_FILE_SIZE_MB
                )
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to read CSF v2 upload %s: %s", upload.filename, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not read one of the uploaded files.",
            ) from exc

    if not pages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of page1 or page2 is required.",
        )

    try:
        return await csf_v2_pipeline.process(
            pages=pages,
            metadata={"client_id": client_id, "source": source.value},
        )
    except ProviderResponseError as exc:
        logger.warning("CSF v2 provider invalid payload for client_id=%s: %s", client_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document AI provider returned an invalid response.",
        ) from exc
    except UpstreamServiceError as exc:
        logger.warning("CSF v2 provider unavailable for client_id=%s: %s", client_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document AI provider is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("CSF v2 pipeline failed for client_id=%s", client_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed. Please try again later.",
        ) from exc

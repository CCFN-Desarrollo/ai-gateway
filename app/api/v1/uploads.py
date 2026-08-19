import logging
from typing import cast

import fitz
from fastapi import HTTPException, UploadFile, status

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
PDF_CONTENT_TYPE = "application/pdf"
READ_CHUNK_SIZE = 1024 * 1024

# Address-proof and CSF PDFs routinely carry a second page (blank back, terms, etc.)
# with no data we need. Only the first page is sent to the LLM.
PDF_RENDER_DPI = 200


def validate_image_file(file: UploadFile) -> None:
    """Raise HTTPException if the uploaded file has an unsupported content type."""
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES and content_type != PDF_CONTENT_TYPE:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{content_type}'. Allowed: JPEG, PNG, WebP, PDF.",
        )


async def read_limited_upload(
    file: UploadFile, max_file_bytes: int, max_file_size_mb: int
) -> bytes:
    """
    Read an uploaded file in bounded chunks so oversized payloads are rejected early.

    The downstream AI provider still needs the full image bytes, but this avoids reading
    arbitrarily large uploads into memory before enforcing the configured size limit.
    """
    total_bytes = 0
    chunks: list[bytes] = []

    while True:
        chunk = await file.read(READ_CHUNK_SIZE)
        if not chunk:
            break

        total_bytes += len(chunk)
        if total_bytes > max_file_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {max_file_size_mb} MB.",
            )
        chunks.append(chunk)

    return b"".join(chunks)


def render_pdf_first_page(pdf_bytes: bytes) -> bytes:
    """
    Rasterize only the first page of a PDF to PNG bytes.

    Address-proof and CSF documents are scanned as multi-page PDFs, but the data
    we validate only ever appears on the first page — later pages are intentionally
    discarded instead of being sent to the LLM.
    """
    if pdf_bytes.lstrip()[:5] != b"%PDF-":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid PDF.",
        )

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not open the uploaded PDF.",
        ) from exc

    try:
        if doc.page_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF has no pages.",
            )
        page = doc.load_page(0)
        zoom = PDF_RENDER_DPI / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return cast(bytes, pixmap.tobytes("png"))
    finally:
        doc.close()


async def read_upload_as_image(
    file: UploadFile, max_file_bytes: int, max_file_size_mb: int
) -> tuple[bytes, str]:
    """
    Read an upload and return (image_bytes, media_type).

    PDFs are rasterized to a PNG of their first page only; images pass through unchanged.
    Callers must call validate_image_file(file) first to reject unsupported content types.
    """
    raw_bytes = await read_limited_upload(file, max_file_bytes, max_file_size_mb)
    content_type = file.content_type or ""

    if content_type == PDF_CONTENT_TYPE:
        logger.debug("Rendering first page of uploaded PDF (%d bytes)", len(raw_bytes))
        return render_pdf_first_page(raw_bytes), "image/png"

    return raw_bytes, content_type or "image/jpeg"

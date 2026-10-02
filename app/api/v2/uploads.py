import logging
from typing import cast

import fitz
from fastapi import HTTPException, UploadFile, status

from app.api.v1.uploads import (
    PDF_CONTENT_TYPE,
    read_limited_upload,
    validate_image_file,
)

logger = logging.getLogger(__name__)

# A SAT CSF is typically two or three pages. Cap rasterization so a long PDF
# cannot fan out into an unbounded number of LLM calls.
CSF_PDF_PAGE_CAP = 4

CsfPage = tuple[bytes, str, str]


def render_pdf_pages(pdf_bytes: bytes, max_pages: int = CSF_PDF_PAGE_CAP) -> list[bytes]:
    """Rasterize up to `max_pages` of a PDF to PNG bytes, in order."""
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
        page_count = min(doc.page_count, max_pages)
        if doc.page_count > max_pages:
            logger.info(
                "CSF PDF has %d pages; rasterizing the first %d",
                doc.page_count,
                max_pages,
            )
        zoom = 200 / 72
        matrix = fitz.Matrix(zoom, zoom)
        pages: list[bytes] = []
        for index in range(page_count):
            pixmap = doc.load_page(index).get_pixmap(matrix=matrix)
            pages.append(cast(bytes, pixmap.tobytes("png")))
        return pages
    finally:
        doc.close()


async def expand_csf_upload(
    file: UploadFile,
    slot: str,
    max_file_bytes: int,
    max_file_size_mb: int,
) -> list[CsfPage]:
    """
    Turn one uploaded CSF file into labeled page images.

    A multi-page PDF is split: the first page uses the existing CSF prompt and
    every later page uses the page-2 prompt. An image, or a one-page PDF, keeps
    the slot the caller assigned (`page1` or `page2`).
    """
    validate_image_file(file)
    raw_bytes = await read_limited_upload(file, max_file_bytes, max_file_size_mb)
    content_type = file.content_type or ""
    slot_document_type = "CSF" if slot == "page1" else "CSF_PAGE2"

    if content_type == PDF_CONTENT_TYPE:
        rendered = render_pdf_pages(raw_bytes)
        if len(rendered) > 1:
            return [
                (png, "image/png", "CSF" if index == 0 else "CSF_PAGE2")
                for index, png in enumerate(rendered)
            ]
        return [(rendered[0], "image/png", slot_document_type)]

    return [(raw_bytes, content_type or "image/jpeg", slot_document_type)]

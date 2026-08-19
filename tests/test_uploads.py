"""Tests for app.api.v1.uploads — PDF-to-image rasterization helpers."""

import fitz
import pytest
from fastapi import HTTPException

from app.api.v1.uploads import render_pdf_first_page


def _make_pdf(num_pages: int) -> bytes:
    doc = fitz.open()
    for _ in range(num_pages):
        doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestRenderPdfFirstPage:
    def test_returns_png_bytes_for_single_page_pdf(self):
        result = render_pdf_first_page(_make_pdf(1))
        assert result.startswith(b"\x89PNG")

    def test_multi_page_pdf_only_loads_first_page(self, monkeypatch):
        """Address-proof/CSF PDFs often carry an irrelevant second page — it must
        never be rasterized or sent anywhere."""
        pdf_bytes = _make_pdf(2)
        calls: list[int] = []
        original_load_page = fitz.Document.load_page

        def spy_load_page(self, page_id, *args, **kwargs):
            calls.append(page_id)
            return original_load_page(self, page_id, *args, **kwargs)

        monkeypatch.setattr(fitz.Document, "load_page", spy_load_page)

        result = render_pdf_first_page(pdf_bytes)

        assert calls == [0]
        assert result.startswith(b"\x89PNG")

    def test_rejects_non_pdf_bytes(self):
        with pytest.raises(HTTPException) as exc_info:
            render_pdf_first_page(b"not a real pdf")
        assert exc_info.value.status_code == 400

    def test_rejects_zero_page_pdf(self, monkeypatch):
        """PyMuPDF itself refuses to create a real zero-page PDF, so the guard is
        exercised by simulating what fitz.open() would return for a corrupt/empty one."""

        class _ZeroPageDoc:
            page_count = 0

            def close(self):
                pass

        monkeypatch.setattr(fitz, "open", lambda **kwargs: _ZeroPageDoc())

        with pytest.raises(HTTPException) as exc_info:
            render_pdf_first_page(b"%PDF-1.4\n%%EOF")
        assert exc_info.value.status_code == 400

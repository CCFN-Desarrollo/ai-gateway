"""Tests for POST /api/v2/validate/csf and the v2 page splitter."""

import io

import fitz
import pytest
from fastapi.testclient import TestClient

from app.api.v2.uploads import expand_csf_upload, render_pdf_pages
from app.models.responses import OCRResult
from app.pipelines.csf_v2_pipeline import CsfV2Pipeline


def _make_pdf(num_pages: int) -> bytes:
    doc = fitz.open()
    for _ in range(num_pages):
        doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class _FakeOcr:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def extract_text(self, image_bytes, media_type="image/jpeg", document_type=None):
        del image_bytes, media_type
        self.calls.append(document_type)
        if document_type == "CSF_PAGE2":
            return OCRResult(
                raw_text="regimenes",
                structured_fields={
                    "regimenes": [
                        {
                            "regimen": "Régimen de Sueldos y Salarios",
                            "fecha_inicio": "2005-01-01",
                            "fecha_fin": "",
                        }
                    ],
                    "actividades_economicas": [
                        {
                            "actividad": "Comercio al por menor",
                            "porcentaje": "100%",
                            "fecha_inicio": "2010-05-24",
                            "fecha_fin": None,
                        }
                    ],
                },
                confidence=0.9,
            )
        return OCRResult(
            raw_text="rfc",
            structured_fields={
                "rfc": "XAXX010101000",
                "full_name": "JUAN PEREZ",
                "zip_code": "22000",
                "fiscal_regimes": ["Sueldos y Salarios"],
                "fiscal_obligations": [],
            },
            confidence=0.95,
        )


class TestRenderPdfPages:
    def test_multi_page_pdf_rasterizes_every_page_up_to_cap(self, monkeypatch):
        pdf_bytes = _make_pdf(5)
        calls: list[int] = []
        original_load_page = fitz.Document.load_page

        def spy_load_page(self, page_id, *args, **kwargs):
            calls.append(page_id)
            return original_load_page(self, page_id, *args, **kwargs)

        monkeypatch.setattr(fitz.Document, "load_page", spy_load_page)

        pages = render_pdf_pages(pdf_bytes)

        assert calls == [0, 1, 2, 3]
        assert len(pages) == 4
        assert all(page.startswith(b"\x89PNG") for page in pages)


class TestExpandCsfUpload:
    @pytest.mark.asyncio
    async def test_multi_page_pdf_labels_first_page_as_csf_and_rest_as_page2(self):
        upload = _upload("csf.pdf", _make_pdf(2), "application/pdf")

        pages = await expand_csf_upload(upload, "page2", max_file_bytes=5_000_000, max_file_size_mb=5)

        assert [document_type for _, _, document_type in pages] == ["CSF", "CSF_PAGE2"]

    @pytest.mark.asyncio
    async def test_single_page_pdf_keeps_the_slot(self):
        upload = _upload("regimen.pdf", _make_pdf(1), "application/pdf")

        pages = await expand_csf_upload(upload, "page2", max_file_bytes=5_000_000, max_file_size_mb=5)

        assert [document_type for _, _, document_type in pages] == ["CSF_PAGE2"]


class TestCsfV2Pipeline:
    @pytest.mark.asyncio
    async def test_both_pages_return_both_sections(self):
        pipeline = CsfV2Pipeline(ocr=_FakeOcr())

        result = await pipeline.process(
            pages=[(b"p1", "image/png", "CSF"), (b"p2", "image/png", "CSF_PAGE2")],
            metadata={"client_id": "C1"},
        )

        assert result.page1 is not None
        assert result.page1.rfc == "XAXX010101000"
        assert result.page2 is not None
        assert result.page2.regimenes[0].regimen == "Régimen de Sueldos y Salarios"
        assert result.page2.regimenes[0].fecha_inicio == "2005-01-01"
        assert result.page2.regimenes[0].fecha_fin is None
        assert result.page2.actividades_economicas[0].actividad == "Comercio al por menor"
        assert result.page2.actividades_economicas[0].porcentaje == 100.0
        assert result.decision is not None
        assert result.pages_processed == 2

    @pytest.mark.asyncio
    async def test_page1_only_leaves_page2_null(self):
        pipeline = CsfV2Pipeline(ocr=_FakeOcr())

        result = await pipeline.process(
            pages=[(b"p1", "image/jpeg", "CSF")],
            metadata={"client_id": "C1"},
        )

        assert result.page1 is not None
        assert result.page1.full_name == "JUAN PEREZ"
        assert result.page2 is None
        assert result.decision is not None

    @pytest.mark.asyncio
    async def test_page2_only_leaves_page1_null_and_skips_rfc_rules(self):
        pipeline = CsfV2Pipeline(ocr=_FakeOcr())

        result = await pipeline.process(
            pages=[(b"p2", "image/jpeg", "CSF_PAGE2")],
            metadata={"client_id": "C1"},
        )

        assert result.page1 is None
        assert result.decision is None
        assert result.final_score is None
        assert result.requires_human_review is None
        assert result.page2 is not None
        assert len(result.page2.regimenes) == 1
        assert result.breakdown == {}


class TestValidateCsfV2Endpoint:
    def test_requires_a_file(self, client: TestClient, api_headers: dict[str, str]):
        response = client.post(
            "/api/v2/validate/csf",
            headers=api_headers,
            data={"client_id": "C1", "source": "crm"},
        )
        assert response.status_code == 422

    def test_page1_image_returns_null_page2(
        self, client: TestClient, api_headers: dict[str, str], dummy_png: bytes, monkeypatch
    ):
        fake = _FakeOcr()
        monkeypatch.setattr(
            "app.pipelines.csf_v2_pipeline.csf_v2_pipeline.ocr_service.extract_text",
            fake.extract_text,
        )

        response = client.post(
            "/api/v2/validate/csf",
            headers=api_headers,
            data={"client_id": "C1", "source": "crm"},
            files={"page1": ("csf.png", io.BytesIO(dummy_png), "image/png")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["page1"]["rfc"] == "XAXX010101000"
        assert body["page2"] is None
        assert body["decision"] is not None
        assert fake.calls == ["CSF"]

    def test_page2_image_returns_null_page1(
        self, client: TestClient, api_headers: dict[str, str], dummy_png: bytes, monkeypatch
    ):
        fake = _FakeOcr()
        monkeypatch.setattr(
            "app.pipelines.csf_v2_pipeline.csf_v2_pipeline.ocr_service.extract_text",
            fake.extract_text,
        )

        response = client.post(
            "/api/v2/validate/csf",
            headers=api_headers,
            data={"client_id": "C1", "source": "crm"},
            files={"page2": ("csf-p2.png", io.BytesIO(dummy_png), "image/png")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["page1"] is None
        assert body["decision"] is None
        assert body["page2"]["regimenes"][0]["regimen"] == "Régimen de Sueldos y Salarios"
        assert body["page2"]["actividades_economicas"][0]["actividad"] == "Comercio al por menor"
        assert fake.calls == ["CSF_PAGE2"]

    def test_multi_page_pdf_returns_both_sections(
        self, client: TestClient, api_headers: dict[str, str], monkeypatch
    ):
        fake = _FakeOcr()
        monkeypatch.setattr(
            "app.pipelines.csf_v2_pipeline.csf_v2_pipeline.ocr_service.extract_text",
            fake.extract_text,
        )

        response = client.post(
            "/api/v2/validate/csf",
            headers=api_headers,
            data={"client_id": "C1", "source": "crm"},
            files={"page1": ("csf.pdf", io.BytesIO(_make_pdf(2)), "application/pdf")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["page1"]["full_name"] == "JUAN PEREZ"
        assert body["page2"]["regimenes"]
        assert body["pages_processed"] == 2
        assert fake.calls == ["CSF", "CSF_PAGE2"]


def _upload(filename: str, payload: bytes, content_type: str):
    from fastapi import UploadFile

    return UploadFile(file=io.BytesIO(payload), filename=filename, headers={"content-type": content_type})

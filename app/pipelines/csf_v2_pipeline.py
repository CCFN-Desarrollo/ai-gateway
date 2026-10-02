import asyncio
import logging
from datetime import UTC, datetime
from uuid import uuid4

from app.core.config import settings
from app.models.responses import (
    CsfActividadEconomica,
    CsfExtractedData,
    CsfPage2Data,
    CsfRegimen,
    CsfV2ValidationResponse,
    Decision,
    ScoringResult,
)
from app.pipelines.base_pipeline import BasePipeline
from app.pipelines.csf_pipeline import _apply_rules, _merge_pages
from app.services.ai_interfaces import OCRProvider
from app.services.ocr_service import csf_ocr_service

logger = logging.getLogger(__name__)

CsfLabeledPage = tuple[bytes, str, str]


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_percent(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _merge_page2(results: list) -> CsfPage2Data:
    regimenes: list[CsfRegimen] = []
    actividades: list[CsfActividadEconomica] = []
    seen_regimenes: set[tuple[str, str]] = set()
    seen_actividades: set[tuple[str, str]] = set()

    for result in results:
        fields = result.structured_fields or {}
        for row in fields.get("regimenes") or []:
            if not isinstance(row, dict):
                continue
            regimen = _optional_text(row.get("regimen"))
            if not regimen:
                continue
            fecha_inicio = _optional_text(row.get("fecha_inicio"))
            key = (regimen.casefold(), fecha_inicio or "")
            if key in seen_regimenes:
                continue
            seen_regimenes.add(key)
            regimenes.append(
                CsfRegimen(
                    regimen=regimen,
                    fecha_inicio=fecha_inicio,
                    fecha_fin=_optional_text(row.get("fecha_fin")),
                )
            )

        for row in fields.get("actividades_economicas") or []:
            if not isinstance(row, dict):
                continue
            actividad = _optional_text(row.get("actividad"))
            if not actividad:
                continue
            fecha_inicio = _optional_text(row.get("fecha_inicio"))
            key = (actividad.casefold(), fecha_inicio or "")
            if key in seen_actividades:
                continue
            seen_actividades.add(key)
            actividades.append(
                CsfActividadEconomica(
                    actividad=actividad,
                    porcentaje=_optional_percent(row.get("porcentaje")),
                    fecha_inicio=fecha_inicio,
                    fecha_fin=_optional_text(row.get("fecha_fin")),
                )
            )

    return CsfPage2Data(regimenes=regimenes, actividades_economicas=actividades)


def _page1_data(fields: dict) -> CsfExtractedData:
    return CsfExtractedData(
        rfc=fields.get("rfc"),
        full_name=fields.get("full_name"),
        curp=fields.get("curp"),
        zip_code=fields.get("zip_code"),
        street=fields.get("street"),
        colony=fields.get("colony"),
        city=fields.get("city"),
        state=fields.get("state"),
        start_date=fields.get("start_date"),
        last_change_date=fields.get("last_change_date"),
        fiscal_regimes=fields.get("fiscal_regimes", []),
        fiscal_obligations=fields.get("fiscal_obligations", []),
    )


class CsfV2Pipeline(BasePipeline):
    def __init__(self, ocr: OCRProvider) -> None:
        self.ocr_service = ocr

    async def process(
        self,
        pages: list[CsfLabeledPage],
        metadata: dict,
    ) -> CsfV2ValidationResponse:
        request_id = uuid4()
        start = self._start_timer()

        logger.info(
            "Starting CSF v2 pipeline | request_id=%s client_id=%s pages=%d",
            request_id,
            metadata.get("client_id"),
            len(pages),
        )

        ocr_results = await asyncio.gather(
            *[
                self.ocr_service.extract_text(image_bytes, media_type, document_type)
                for image_bytes, media_type, document_type in pages
            ]
        )

        page1_results = [
            result
            for result, (_, _, document_type) in zip(ocr_results, pages, strict=True)
            if document_type == "CSF"
        ]
        page2_results = [
            result
            for result, (_, _, document_type) in zip(ocr_results, pages, strict=True)
            if document_type == "CSF_PAGE2"
        ]

        page1: CsfExtractedData | None = None
        decision: Decision | None = None
        final_score: float | None = None
        requires_human_review: bool | None = None
        breakdown: dict = {}

        if page1_results:
            merged_fields, ocr_confidence = _merge_pages(page1_results)
            rules_result = _apply_rules(merged_fields)
            final_score = (ocr_confidence * 50.0) + (rules_result.rules_score * 50.0)
            if final_score >= settings.SCORE_AUTO_APPROVE:
                decision = Decision.AUTO_APPROVED
            elif final_score >= settings.SCORE_HUMAN_REVIEW:
                decision = Decision.HUMAN_REVIEW
            else:
                decision = Decision.AUTO_REJECTED
            scoring = ScoringResult(
                final_score=final_score,
                decision=decision,
                requires_human_review=decision == Decision.HUMAN_REVIEW,
                breakdown={
                    "ocr_score": round(ocr_confidence * 50.0, 2),
                    "rules_score": round(rules_result.rules_score * 50.0, 2),
                    "passed_rules": rules_result.passed_rules,
                    "failed_rules": rules_result.failed_rules,
                },
            )
            requires_human_review = scoring.requires_human_review
            breakdown = scoring.breakdown
            page1 = _page1_data(merged_fields)

        page2 = _merge_page2(page2_results) if page2_results else None

        elapsed_ms = self._elapsed_ms(start)
        logger.info(
            "CSF v2 pipeline complete | request_id=%s page1=%s page2=%s elapsed_ms=%.0f",
            request_id,
            page1 is not None,
            page2 is not None,
            elapsed_ms,
        )

        return CsfV2ValidationResponse(
            request_id=request_id,
            timestamp=datetime.now(UTC),
            processing_time_ms=elapsed_ms,
            document_type="CSF",
            pages_processed=len(pages),
            page1=page1,
            page2=page2,
            final_score=final_score,
            decision=decision,
            requires_human_review=requires_human_review,
            breakdown=breakdown,
        )


csf_v2_pipeline = CsfV2Pipeline(ocr=csf_ocr_service)

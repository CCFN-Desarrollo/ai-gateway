import asyncio
import logging
import re
from datetime import UTC, datetime
from uuid import uuid4

from app.core.config import settings
from app.models.responses import (
    CsfExtractedData,
    CsfValidationResponse,
    Decision,
    RulesResult,
    ScoringResult,
)
from app.pipelines.base_pipeline import BasePipeline
from app.services.ai_interfaces import OCRProvider
from app.services.ocr_service import csf_ocr_service

logger = logging.getLogger(__name__)

_RFC_PATTERN = re.compile(r"^[A-Z&Ñ]{3,4}[0-9]{6}[A-Z0-9]{3}$")
_ZIP_PATTERN = re.compile(r"^\d{5}$")


def _merge_pages(results: list) -> tuple[dict, float]:
    """Merge OCR results from multiple pages, highest confidence fields win."""
    sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)
    merged: dict = {}
    all_regimes: list[str] = []
    all_obligations: list[str] = []

    for result in sorted_results:
        fields = result.structured_fields
        for key, value in fields.items():
            if key in ("fiscal_regimes", "fiscal_obligations"):
                continue
            if key not in merged and value:
                merged[key] = value

        regimes = fields.get("fiscal_regimes") or []
        obligations = fields.get("fiscal_obligations") or []
        if isinstance(regimes, list):
            all_regimes.extend(r for r in regimes if r)
        if isinstance(obligations, list):
            all_obligations.extend(o for o in obligations if o)

    merged["fiscal_regimes"] = list(dict.fromkeys(all_regimes))
    merged["fiscal_obligations"] = list(dict.fromkeys(all_obligations))
    best_confidence = sorted_results[0].confidence if sorted_results else 0.0
    return merged, best_confidence


def _apply_rules(fields: dict) -> RulesResult:
    passed: list[str] = []
    failed: list[str] = []
    flags: list[str] = []
    rules_score = 0.0

    rfc = (fields.get("rfc") or "").strip().upper()
    if _RFC_PATTERN.match(rfc):
        rules_score += 0.35
        passed.append("rfc_valid")
    elif rfc:
        rules_score += 0.10
        passed.append("rfc_present")
        failed.append("rfc_invalid_format")
        flags.append("rfc_invalid_format")
    else:
        failed.append("rfc_missing")
        flags.append("rfc_missing")

    if (fields.get("full_name") or "").strip():
        rules_score += 0.30
        passed.append("name_present")
    else:
        failed.append("name_missing")
        flags.append("name_missing")

    if fields.get("fiscal_regimes"):
        rules_score += 0.20
        passed.append("has_fiscal_regime")
    else:
        failed.append("no_fiscal_regime")
        flags.append("no_fiscal_regime")

    zip_code = (fields.get("zip_code") or "").strip()
    if _ZIP_PATTERN.match(zip_code):
        rules_score += 0.15
        passed.append("zip_code_valid")
    elif zip_code:
        rules_score += 0.05
        failed.append("zip_code_invalid_format")
        flags.append("zip_code_invalid_format")
    else:
        failed.append("zip_code_missing")
        flags.append("zip_code_missing")

    return RulesResult(
        passed_rules=passed,
        failed_rules=failed,
        flags=flags,
        rules_score=rules_score,
    )


class CsfPipeline(BasePipeline):
    def __init__(self, ocr: OCRProvider) -> None:
        self.ocr_service = ocr

    async def process(
        self,
        pages: list[tuple[bytes, str]],
        metadata: dict,
    ) -> CsfValidationResponse:
        request_id = uuid4()
        start = self._start_timer()

        logger.info(
            "Starting CSF pipeline | request_id=%s client_id=%s pages=%d",
            request_id,
            metadata.get("client_id"),
            len(pages),
        )

        ocr_tasks = [
            self.ocr_service.extract_text(img_bytes, media_type, "CSF")
            for img_bytes, media_type in pages
        ]
        ocr_results = await asyncio.gather(*ocr_tasks)

        merged_fields, ocr_confidence = _merge_pages(list(ocr_results))
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

        extracted = CsfExtractedData(
            rfc=merged_fields.get("rfc"),
            full_name=merged_fields.get("full_name"),
            curp=merged_fields.get("curp"),
            zip_code=merged_fields.get("zip_code"),
            street=merged_fields.get("street"),
            colony=merged_fields.get("colony"),
            city=merged_fields.get("city"),
            state=merged_fields.get("state"),
            start_date=merged_fields.get("start_date"),
            last_change_date=merged_fields.get("last_change_date"),
            fiscal_regimes=merged_fields.get("fiscal_regimes", []),
            fiscal_obligations=merged_fields.get("fiscal_obligations", []),
        )

        elapsed_ms = self._elapsed_ms(start)
        logger.info(
            "CSF pipeline complete | request_id=%s score=%.1f decision=%s elapsed_ms=%.0f",
            request_id,
            final_score,
            decision,
            elapsed_ms,
        )

        return CsfValidationResponse(
            request_id=request_id,
            timestamp=datetime.now(UTC),
            processing_time_ms=elapsed_ms,
            document_type="CSF",
            final_score=final_score,
            decision=decision,
            requires_human_review=scoring.requires_human_review,
            extracted_data=extracted,
            pages_processed=len(pages),
            breakdown=scoring.breakdown,
        )


csf_pipeline = CsfPipeline(ocr=csf_ocr_service)

import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.models.responses import ReceiptExtractedData, ReceiptValidationResponse, VisionResult
from app.pipelines.base_pipeline import BasePipeline
from app.services.ai_interfaces import OCRProvider, VisionProvider
from app.services.document_preprocessor import document_preprocessor
from app.services.ocr_service import receipt_ocr_service
from app.services.rules_engine import _parse_date, rules_engine
from app.services.scoring_service import ScoringService, scoring_service
from app.services.vision_service import receipt_vision_service

logger = logging.getLogger(__name__)


class ReceiptPipeline(BasePipeline):
    """
    Orchestrates the full validation pipeline for receipt documents.

    Steps:
        1. OCR  — extract text and fields from the image
        2. Vision — assess document quality and consistency
        3. Rules  — validate required receipt fields
        4. Score  — compute weighted confidence and routing decision
    """

    def __init__(
        self,
        ocr: OCRProvider,
        vision: VisionProvider,
        scoring: ScoringService,
    ) -> None:
        self.ocr_service = ocr
        self.vision_service = vision
        self.scoring_service = scoring

    async def process(
        self,
        image_bytes: bytes,
        media_type: str,
        metadata: dict,
    ) -> ReceiptValidationResponse:
        """
        Run OCR → Vision → Rules → Scoring for a receipt image.

        Args:
            image_bytes: Raw bytes of the receipt image.
            media_type:  MIME type of the image.
            metadata:    Must contain 'client_id' and optionally 'source'.

        Returns:
            ReceiptValidationResponse with all validation data.
        """
        request_id = uuid4()
        start = self._start_timer()

        document_type = metadata.get("document_type", "RECEIPT")
        logger.info(
            "Starting receipt pipeline | request_id=%s client_id=%s source=%s document_type=%s",
            request_id,
            metadata.get("client_id"),
            metadata.get("source"),
            document_type,
        )

        preprocessed = document_preprocessor.preprocess_identity_document(
            image_bytes=image_bytes,
            document_type=document_type,
        )
        logger.info(
            "Receipt preprocessing | request_id=%s used_specialized_crop=%s quality_flags=%s debug_image_path=%s",
            request_id,
            preprocessed.used_specialized_crop,
            preprocessed.quality_flags,
            preprocessed.debug_image_path,
        )

        # Step 1 — OCR
        ocr_result = await self.ocr_service.extract_text(
            preprocessed.image_bytes,
            media_type,
            document_type=document_type,
        )
        logger.debug("OCR done | request_id=%s confidence=%.2f", request_id, ocr_result.confidence)

        # Step 2 — Vision AI
        if document_type in ("ADDRESS_PROOF", "COMPROBANTE_DOMICILIO"):
            vision_result = self._build_neutral_vision_result()
            logger.info(
                "Skipping vision stage for request_id=%s document_type=%s",
                request_id,
                document_type,
            )
        else:
            vision_result = await self.vision_service.analyze_document(
                image_bytes,
                document_type,
                media_type,
            )
            logger.debug(
                "Vision done | request_id=%s authentic=%s score=%.2f",
                request_id,
                vision_result.is_authentic,
                vision_result.authenticity_score,
            )

        # Step 3 — Rules engine
        rules_result = rules_engine.validate_receipt(ocr_result, document_type)
        logger.debug(
            "Rules done | request_id=%s passed=%s failed=%s",
            request_id,
            rules_result.passed_rules,
            rules_result.failed_rules,
        )

        # Step 4 — Scoring
        scoring_result = self.scoring_service.calculate_score(
            ocr_result, vision_result, rules_result
        )

        elapsed_ms = self._elapsed_ms(start)
        self._log_result(
            request_id,
            document_type,
            scoring_result.final_score,
            scoring_result.decision.value,
            elapsed_ms,
        )

        # Build extracted data from OCR structured fields
        fields = ocr_result.structured_fields
        extracted = ReceiptExtractedData(
            date=fields.get("date") or fields.get("fecha"),
            total=fields.get("total") or fields.get("importe") or fields.get("amount"),
            issuer=fields.get("issuer") or fields.get("emisor") or fields.get("company"),
            receipt_number=(
                fields.get("receipt_number")
                or fields.get("folio")
                or fields.get("numero")
                or fields.get("ticket")
            ),
            street=fields.get("street") or fields.get("calle") or fields.get("domicilio"),
            colony=fields.get("colony") or fields.get("colonia"),
            zip_code=fields.get("zip_code") or fields.get("codigo_postal") or fields.get("cp"),
            city=fields.get("city") or fields.get("ciudad") or fields.get("municipio"),
            state=fields.get("state") or fields.get("estado"),
            issue_date=rules_engine.get_issue_date_str(fields),
        )
        issue_date_str = rules_engine.get_issue_date_str(fields)
        is_expired = (
            self._compute_is_expired(issue_date_str) if document_type in ("ADDRESS_PROOF", "COMPROBANTE_DOMICILIO") else False
        )

        return ReceiptValidationResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=elapsed_ms,
            document_type=document_type,
            final_score=scoring_result.final_score,
            decision=scoring_result.decision,
            requires_human_review=scoring_result.requires_human_review,
            extracted_data=extracted,
            is_expired=is_expired,
            fraud_indicators=vision_result.fraud_indicators,
            breakdown=scoring_result.breakdown,
        )

    @staticmethod
    def _compute_is_expired(issue_date_str: str | None) -> bool:
        """Return True if the issue date could be parsed and is older than 3 months."""
        if not issue_date_str:
            return False
        parsed = _parse_date(issue_date_str)
        if parsed is None:
            return False
        return parsed < (datetime.now() - timedelta(days=92))

    @staticmethod
    def _build_neutral_vision_result() -> VisionResult:
        return VisionResult(
            is_authentic=True,
            fraud_indicators=[],
            authenticity_score=1.0,
            document_matches_expected_type=True,
            visual_validation_score=1.0,
            quality_flags=[],
            consistency_flags=[],
            notes="Vision skipped for address-proof OCR-only flow.",
        )


receipt_pipeline = ReceiptPipeline(
    ocr=receipt_ocr_service,
    vision=receipt_vision_service,
    scoring=scoring_service,
)

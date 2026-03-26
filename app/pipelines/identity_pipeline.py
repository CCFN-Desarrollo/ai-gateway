import logging
import re
from datetime import datetime, timezone
from uuid import uuid4

from app.models.responses import IdentityExtractedData, IdentityValidationResponse, VisionResult
from app.pipelines.base_pipeline import BasePipeline
from app.services.ai_interfaces import OCRProvider, VisionProvider
from app.services.document_preprocessor import document_preprocessor
from app.services.ocr_service import identity_ocr_service
from app.services.rules_engine import _parse_date, rules_engine
from app.services.scoring_service import ScoringService, scoring_service
from app.services.vision_service import identity_vision_service

logger = logging.getLogger(__name__)
_INE_REVERSO_ID_PATTERNS = (
    re.compile(r"IDMEX\d+", re.IGNORECASE),
    re.compile(r"\b[A-Z]{4,}\d{6,}\b"),
)


class IdentityPipeline(BasePipeline):
    """
    Orchestrates the full validation pipeline for identity documents (INE, PASAPORTE, LICENCIA).

    Steps:
        1. OCR  — extract text and fields from the image
        2. Vision — evaluate document type, capture quality, and basic consistency
        3. Rules  — validate required identity fields
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
    ) -> IdentityValidationResponse:
        """
        Run OCR → Vision → Rules → Scoring for an identity document image.

        Args:
            image_bytes: Raw bytes of the identity document image.
            media_type:  MIME type of the image.
            metadata:    Must contain 'client_id' and 'document_type'.

        Returns:
            IdentityValidationResponse with all validation data.
        """
        request_id = uuid4()
        document_type = metadata.get("document_type", "IDENTITY")
        start = self._start_timer()

        logger.info(
            "Starting identity pipeline | request_id=%s client_id=%s document_type=%s",
            request_id,
            metadata.get("client_id"),
            document_type,
        )

        preprocessed = document_preprocessor.preprocess_identity_document(
            image_bytes=image_bytes,
            document_type=document_type,
        )
        logger.info(
            "Identity preprocessing | request_id=%s used_specialized_crop=%s quality_flags=%s debug_image_path=%s",
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
        if document_type == "INE_REVERSO":
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
        rules_result = rules_engine.validate_identity(ocr_result, document_type)
        logger.debug(
            "Rules done | request_id=%s passed=%s failed=%s",
            request_id,
            rules_result.passed_rules,
            rules_result.failed_rules,
        )

        # Step 4 — Scoring
        scoring_result = self.scoring_service.calculate_score(
            ocr_result,
            vision_result,
            rules_result,
            document_type=document_type,
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
        if document_type == "INE_REVERSO":
            fields = dict(fields)
            normalized_id = self._normalize_ine_reverso_id(
                fields.get("id_number"), ocr_result.raw_text
            )
            if normalized_id:
                fields["id_number"] = normalized_id
        expiry_date_str = rules_engine.get_expiry_date_str(fields)
        is_expired = self._compute_is_expired(expiry_date_str)

        extracted = IdentityExtractedData(
            full_name=(
                None
                if document_type == "INE_REVERSO"
                else (
                    fields.get("full_name") or fields.get("nombre_completo") or fields.get("nombre")
                )
            ),
            id_number=(
                fields.get("id_number")
                or fields.get("clave_elector")
                or fields.get("numero_identificacion")
                or fields.get("folio")
            ),
            curp=None if document_type == "INE_REVERSO" else fields.get("curp"),
            expiry_date=None if document_type == "INE_REVERSO" else expiry_date_str,
            date_of_birth=(
                None
                if document_type == "INE_REVERSO"
                else (
                    fields.get("date_of_birth")
                    or fields.get("fecha_nacimiento")
                    or fields.get("dob")
                )
            ),
        )

        return IdentityValidationResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=elapsed_ms,
            document_type=document_type,
            final_score=scoring_result.final_score,
            decision=scoring_result.decision,
            requires_human_review=scoring_result.requires_human_review,
            extracted_data=extracted,
            is_expired=is_expired,
            quality_flags=[*preprocessed.quality_flags, *vision_result.quality_flags],
            consistency_flags=vision_result.consistency_flags,
            breakdown=scoring_result.breakdown,
            used_specialized_crop=preprocessed.used_specialized_crop,
        )

    @staticmethod
    def _compute_is_expired(expiry_date_str: str | None) -> bool:
        """Return True if the expiry date could be parsed and is in the past."""
        if not expiry_date_str:
            return False
        parsed = _parse_date(expiry_date_str)
        if parsed is None:
            return False
        return parsed <= datetime.now()

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
            notes="Vision skipped for INE reverso OCR-only flow.",
        )

    @staticmethod
    def _normalize_ine_reverso_id(
        candidate_id: object,
        raw_text: str,
    ) -> str | None:
        texts_to_scan = []
        if isinstance(candidate_id, str) and candidate_id.strip():
            texts_to_scan.append(candidate_id)
        if raw_text.strip():
            texts_to_scan.append(raw_text)

        for text in texts_to_scan:
            normalized = " ".join(text.split())
            for pattern in _INE_REVERSO_ID_PATTERNS:
                match = pattern.search(normalized)
                if match:
                    return match.group(0).upper()
        return str(candidate_id).strip() if isinstance(candidate_id, str) else None


identity_pipeline = IdentityPipeline(
    ocr=identity_ocr_service,
    vision=identity_vision_service,
    scoring=scoring_service,
)

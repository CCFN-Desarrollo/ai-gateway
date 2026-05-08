import logging

from app.core.config import settings
from app.models.responses import Decision, OCRResult, RulesResult, ScoringResult, VisionResult

logger = logging.getLogger(__name__)
_IDENTITY_DOCUMENT_TYPES = {"INE", "INE_REVERSO", "PASAPORTE", "LICENCIA"}

# Weighted contributions to the final score (must sum to 1.0)
_WEIGHT_OCR = 0.25
_WEIGHT_VISION = 0.40
_WEIGHT_RULES = 0.35
_IDENTITY_WEIGHT_OCR = 0.40
_IDENTITY_WEIGHT_VISION = 0.20
_IDENTITY_WEIGHT_RULES = 0.40


class ScoringService:
    """Aggregates OCR, vision, and rules results into a final confidence score and routing decision."""

    def calculate_score(
        self,
        ocr_result: OCRResult,
        vision_result: VisionResult,
        rules_result: RulesResult,
        document_type: str | None = None,
    ) -> ScoringResult:
        """
        Compute a weighted confidence score and determine the routing decision.

        Weights:
            - OCR confidence:          25%
            - Vision authenticity:     40%
            - Rules compliance:        35%

        Thresholds:
            - score > SCORE_AUTO_APPROVE  → AUTO_APPROVED
            - score >= SCORE_HUMAN_REVIEW → HUMAN_REVIEW
            - score < SCORE_HUMAN_REVIEW  → AUTO_REJECTED

        Args:
            ocr_result:    Result from OCRService.
            vision_result: Result from VisionService.
            rules_result:  Result from RulesEngine.

        Returns:
            ScoringResult with final_score (0-100), decision, breakdown, and requires_human_review flag.
        """
        ocr_score_100 = ocr_result.confidence * 100.0
        vision_metric = self._vision_score_for_document(vision_result, document_type)
        vision_score_100 = vision_metric * 100.0
        rules_score_100 = rules_result.rules_score * 100.0
        weights = self._weights_for_document(document_type)

        final_score = (
            ocr_score_100 * weights["ocr"]
            + vision_score_100 * weights["vision"]
            + rules_score_100 * weights["rules"]
        )
        final_score = round(final_score, 2)

        decision = self._apply_thresholds(final_score)
        if "expired_document" in rules_result.flags:
            decision = Decision.AUTO_REJECTED
        elif (
            "unknown_expiry" in rules_result.flags and decision == Decision.AUTO_APPROVED
        ) or self._requires_identity_review(vision_result, document_type):
            decision = Decision.HUMAN_REVIEW
        requires_human_review = decision == Decision.HUMAN_REVIEW

        breakdown = {
            "ocr_confidence": round(ocr_result.confidence, 4),
            "ocr_contribution": round(ocr_score_100 * weights["ocr"], 2),
            "vision_authenticity": round(vision_result.authenticity_score, 4),
            "visual_validation": round(vision_metric, 4),
            "vision_contribution": round(vision_score_100 * weights["vision"], 2),
            "rules_score": round(rules_result.rules_score, 4),
            "rules_contribution": round(rules_score_100 * weights["rules"], 2),
            "weights": weights,
            "thresholds": {
                "auto_approve": settings.SCORE_AUTO_APPROVE,
                "human_review": settings.SCORE_HUMAN_REVIEW,
            },
        }
        if rules_result.flags:
            breakdown["policy_flags"] = list(rules_result.flags)
        if document_type in _IDENTITY_DOCUMENT_TYPES:
            breakdown["quality_flags"] = list(vision_result.quality_flags)
            breakdown["consistency_flags"] = list(vision_result.consistency_flags)
            breakdown["document_matches_expected_type"] = (
                vision_result.document_matches_expected_type
            )

        logger.info(
            "Score calculated: final=%.2f decision=%s (ocr=%.1f vision=%.1f rules=%.1f)",
            final_score,
            decision.value,
            ocr_score_100,
            vision_score_100,
            rules_score_100,
        )

        return ScoringResult(
            final_score=final_score,
            decision=decision,
            breakdown=breakdown,
            requires_human_review=requires_human_review,
        )

    @staticmethod
    def _apply_thresholds(score: float) -> Decision:
        """Map a numeric score to a routing decision using configured thresholds."""
        if score > settings.SCORE_AUTO_APPROVE:
            return Decision.AUTO_APPROVED
        if score >= settings.SCORE_HUMAN_REVIEW:
            return Decision.HUMAN_REVIEW
        return Decision.AUTO_REJECTED

    @staticmethod
    def _weights_for_document(document_type: str | None) -> dict[str, float]:
        if document_type in _IDENTITY_DOCUMENT_TYPES:
            return {
                "ocr": _IDENTITY_WEIGHT_OCR,
                "vision": _IDENTITY_WEIGHT_VISION,
                "rules": _IDENTITY_WEIGHT_RULES,
            }
        return {"ocr": _WEIGHT_OCR, "vision": _WEIGHT_VISION, "rules": _WEIGHT_RULES}

    @staticmethod
    def _vision_score_for_document(vision_result: VisionResult, document_type: str | None) -> float:
        if document_type in _IDENTITY_DOCUMENT_TYPES:
            return vision_result.visual_validation_score
        return vision_result.authenticity_score

    @staticmethod
    def _requires_identity_review(vision_result: VisionResult, document_type: str | None) -> bool:
        if document_type not in _IDENTITY_DOCUMENT_TYPES:
            return False
        return (
            not vision_result.document_matches_expected_type
            or bool(vision_result.quality_flags)
            or bool(vision_result.consistency_flags)
        )


scoring_service = ScoringService()

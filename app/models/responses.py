from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


class Decision(str, Enum):
    AUTO_APPROVED = "AUTO_APPROVED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    AUTO_REJECTED = "AUTO_REJECTED"


# ---------------------------------------------------------------------------
# Internal pipeline result models
# ---------------------------------------------------------------------------


class OCRResult(BaseModel):
    """Result from the OCR service."""

    raw_text: str
    structured_fields: dict = Field(default_factory=dict)
    confidence: float  # 0.0 – 1.0


class VisionResult(BaseModel):
    """Result from the Vision AI service."""

    is_authentic: bool = True
    fraud_indicators: list[str] = Field(default_factory=list)
    authenticity_score: float = 0.5  # 0.0 – 1.0
    document_matches_expected_type: bool = True
    visual_validation_score: float = 0.5  # 0.0 – 1.0
    quality_flags: list[str] = Field(default_factory=list)
    consistency_flags: list[str] = Field(default_factory=list)
    notes: str = ""


class RulesResult(BaseModel):
    """Result from the Rules Engine."""

    passed_rules: list[str] = Field(default_factory=list)
    failed_rules: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    rules_score: float  # 0.0 – 1.0


class ScoringResult(BaseModel):
    """Final scoring and routing decision."""

    final_score: float  # 0.0 – 100.0
    decision: Decision
    breakdown: dict = Field(default_factory=dict)
    requires_human_review: bool


# ---------------------------------------------------------------------------
# Extracted data sub-models
# ---------------------------------------------------------------------------


class ReceiptExtractedData(BaseModel):
    date: str | None = None
    total: str | None = None
    issuer: str | None = None
    receipt_number: str | None = None
    street: str | None = None
    colony: str | None = None
    zip_code: str | None = None
    city: str | None = None
    state: str | None = None
    issue_date: str | None = None


class IdentityExtractedData(BaseModel):
    full_name: str | None = None
    id_number: str | None = None
    curp: str | None = None
    expiry_date: str | None = None
    date_of_birth: str | None = None


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------


class BaseValidationResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    request_id: UUID
    timestamp: datetime
    processing_time_ms: float
    document_type: str
    final_score: float
    decision: Decision
    requires_human_review: bool


class ReceiptValidationResponse(BaseValidationResponse):
    extracted_data: ReceiptExtractedData
    is_expired: bool = False
    fraud_indicators: list[str] = Field(default_factory=list)
    breakdown: dict = Field(default_factory=dict)


class IdentityValidationResponse(BaseValidationResponse):
    extracted_data: IdentityExtractedData
    is_expired: bool
    quality_flags: list[str] = Field(default_factory=list)
    consistency_flags: list[str] = Field(default_factory=list)
    breakdown: dict = Field(default_factory=dict)
    used_specialized_crop: bool = False


# ---------------------------------------------------------------------------
# Health response
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    uptime_seconds: float

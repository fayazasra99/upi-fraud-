from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

Decision = Literal["allow", "review", "block"]
CaseStatus = Literal[
    "allow",
    "review",
    "block",
    "investigating",
    "resolved_legit",
    "confirmed_fraud",
]
UserRole = Literal["analyst", "merchant_admin"]


class UserProfile(BaseModel):
    username: str
    role: UserRole
    merchant_id: str | None = None


class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=3, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: UserProfile


class TransactionIn(BaseModel):
    merchant_id: str = Field(min_length=2, max_length=64)
    payer_id: str = Field(min_length=2, max_length=64)
    amount: float = Field(gt=0)
    currency: str = Field(default="INR", min_length=3, max_length=6)
    device_id: str = Field(min_length=2, max_length=64)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    payer_account_age_days: int = Field(ge=0, le=36500)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RuleCondition(BaseModel):
    min_amount: float | None = Field(default=None, ge=0)
    max_amount: float | None = Field(default=None, ge=0)
    velocity_10m_gt: int | None = Field(default=None, ge=0)
    amount_spike_gt: float | None = Field(default=None, ge=0)
    require_new_device: bool = False
    geo_distance_km_gt: float | None = Field(default=None, ge=0)
    account_age_days_lt: int | None = Field(default=None, ge=0)
    night_hours_only: bool = False


class RuleCreate(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    merchant_id: str | None = Field(default=None, max_length=64)
    action: Decision
    condition: RuleCondition
    enabled: bool = True


class RuleRecord(BaseModel):
    id: str
    name: str
    merchant_id: str | None
    action: Decision
    condition: RuleCondition
    enabled: bool
    created_at: datetime


class ToggleRuleRequest(BaseModel):
    enabled: bool


class TransactionRecord(BaseModel):
    id: str
    merchant_id: str
    payer_id: str
    amount: float
    currency: str
    device_id: str
    lat: float
    lon: float
    payer_account_age_days: int
    timestamp: datetime
    decision: Decision
    status: CaseStatus
    risk_score: float
    heuristic_score: float | None = None
    model_probability: float | None = None
    reasons: list[str]
    features: dict[str, Any]
    case_note: str | None = None


class NoteUpdate(BaseModel):
    note: str = Field(min_length=1, max_length=500)


class StatusUpdate(BaseModel):
    status: CaseStatus
    note: str | None = Field(default=None, max_length=500)


class Metrics(BaseModel):
    merchant_id: str | None = None
    total_24h: int
    allow_24h: int
    review_24h: int
    block_24h: int
    avg_risk_score_24h: float


class MerchantSummary(BaseModel):
    merchant_id: str
    total_24h: int
    allow_24h: int
    review_24h: int
    block_24h: int
    avg_risk_score_24h: float
    block_rate_pct: float
    review_rate_pct: float


class ModelStatus(BaseModel):
    loaded: bool
    path: str
    feature_order: list[str]
    last_loaded_at: datetime | None = None
    trained_at: datetime | None = None
    metrics: dict[str, Any] | None = None


class CsvUploadRowError(BaseModel):
    row: int
    message: str


class CsvUploadResult(BaseModel):
    total_rows: int
    inserted: int
    failed: int
    errors: list[CsvUploadRowError]


class ModelTrainRequest(BaseModel):
    merchant_id: str | None = Field(default=None, max_length=64)
    min_samples: int = Field(default=300, ge=40, le=50000)
    max_cases: int = Field(default=20000, ge=10, le=200000)
    allow_synthetic: bool = True


class ModelTrainResult(BaseModel):
    merchant_id: str | None = None
    model_path: str
    trained_at: datetime
    confirmed_cases_used: int
    synthetic_added: int
    trained_samples: int
    metrics: dict[str, Any]

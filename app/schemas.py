"""Pydantic v2 request/response contracts for the inference API.

Validation is strict at the edge: unknown fields, wrong JSON types, out-of-range values,
and unknown category levels are rejected with HTTP 422 before they reach the model.
(The pipeline itself is also defensive, for batch jobs that bypass the API.)
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import CATEGORY_LEVELS

MAX_BATCH_SIZE = int(os.getenv("CHURN_MAX_BATCH_SIZE", "1000"))

# Literal types generated from the single feature contract in src/config.py.
ContractType = Literal[tuple(CATEGORY_LEVELS["contract_type"])]
PaymentMethod = Literal[tuple(CATEGORY_LEVELS["payment_method"])]
InternetService = Literal[tuple(CATEGORY_LEVELS["internet_service"])]
PlanTier = Literal[tuple(CATEGORY_LEVELS["plan_tier"])]
Region = Literal[tuple(CATEGORY_LEVELS["region"])]

EXAMPLE_CUSTOMER: dict[str, Any] = {
    "customer_id": "CUST-004242",
    "age": 29,
    "tenure_months": 4,
    "monthly_charges": 94.5,
    "total_charges": 310.2,
    "num_support_tickets": 3,
    "avg_monthly_usage_gb": 120.0,
    "days_since_last_login": 21,
    "num_addon_services": 0,
    "satisfaction_score": 2,
    "late_payments_12m": 1,
    "contract_type": "month_to_month",
    "payment_method": "electronic_check",
    "internet_service": "fiber_optic",
    "plan_tier": "standard",
    "region": "west",
    "paperless_billing": True,
    "has_partner": False,
}


class ResponseModel(BaseModel):
    # Allow field names such as `model_version` without pydantic namespace warnings.
    model_config = ConfigDict(protected_namespaces=())


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ChurnPredictionRequest(BaseModel):
    """One customer's current account snapshot."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={"examples": [EXAMPLE_CUSTOMER]},
    )

    customer_id: str | None = Field(default=None, max_length=64, description="Echoed back.")
    age: int = Field(ge=18, le=100)
    tenure_months: int = Field(ge=0, le=120, description="Months since signup.")
    monthly_charges: float = Field(gt=0, le=500, description="Current monthly bill (USD).")
    total_charges: float | None = Field(
        default=None, ge=0, le=60_000, description="Lifetime billed amount; null if unknown."
    )
    num_support_tickets: int = Field(ge=0, le=50, description="Tickets in the last 90 days.")
    avg_monthly_usage_gb: float | None = Field(default=None, ge=0, le=5_000)
    days_since_last_login: int = Field(ge=0, le=365)
    num_addon_services: int = Field(ge=0, le=10)
    satisfaction_score: int | None = Field(
        default=None, ge=1, le=5, description="Latest CSAT survey (1-5); null if unanswered."
    )
    late_payments_12m: int = Field(ge=0, le=12)
    contract_type: ContractType
    payment_method: PaymentMethod
    internet_service: InternetService
    plan_tier: PlanTier
    region: Region
    paperless_billing: bool
    has_partner: bool

    def feature_record(self) -> dict[str, Any]:
        return self.model_dump(exclude={"customer_id"})


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[ChurnPredictionRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class PredictionResponse(ResponseModel):
    customer_id: str | None
    churn_probability: float = Field(ge=0, le=1, description="Calibrated P(churn).")
    churn_prediction: bool = Field(description="probability >= decision_threshold")
    risk_tier: RiskTier
    decision_threshold: float
    model_version: str


class BatchPredictionResponse(ResponseModel):
    predictions: list[PredictionResponse]
    count: int
    n_predicted_churners: int
    model_version: str


class FeatureContribution(ResponseModel):
    feature: str
    value: float | str | None
    shap_value: float = Field(description="Contribution to calibrated churn log-odds.")


class ExplanationResponse(ResponseModel):
    customer_id: str | None
    churn_probability: float
    base_value: float = Field(description="Log-odds of the average customer.")
    base_probability: float
    contributions: list[FeatureContribution] = Field(description="Sorted by |shap_value|.")
    model_version: str


class HealthResponse(ResponseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None
    model_type: str | None
    decision_threshold: float | None
    uptime_seconds: float
    api_version: str


class RouteLatency(ResponseModel):
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class LatencyMetricsResponse(ResponseModel):
    window_size: int
    routes: dict[str, RouteLatency]


class ModelInfoResponse(ResponseModel):
    model_version: str
    model_type: str
    trained_at: str
    decision_threshold: float
    thresholds: dict[str, float]
    calibration: dict[str, Any]
    hyperparameters: dict[str, Any]
    features: dict[str, Any]
    holdout_metrics: dict[str, Any] | None = None
    inference_path: Literal["compiled", "sklearn"]

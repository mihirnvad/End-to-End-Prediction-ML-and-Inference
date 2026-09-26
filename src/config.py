"""Central configuration: paths, feature contract, hyperparameter spaces, business costs.

Everything that the training job, the inference service, the dashboard, and the tests
must agree on lives here, so the feature contract has exactly one source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import loguniform, randint, uniform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "churn_dataset.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --------------------------------------------------------------------------------------
# Feature contract
# --------------------------------------------------------------------------------------
TARGET = "churned"
ID_COLUMN = "customer_id"

NUMERIC_FEATURES: list[str] = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "avg_monthly_usage_gb",
    "days_since_last_login",
    "num_addon_services",
    "satisfaction_score",
    "late_payments_12m",
]
CATEGORICAL_FEATURES: list[str] = [
    "contract_type",
    "payment_method",
    "internet_service",
    "plan_tier",
    "region",
]
BOOLEAN_FEATURES: list[str] = ["paperless_billing", "has_partner"]

# Raw columns a caller must supply (values may be null; the columns may not be absent).
RAW_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

# Stateless derived features added by `FeatureEngineer` before preprocessing.
ENGINEERED_FEATURES: list[str] = ["avg_monthly_spend", "charge_increase_pct", "is_early_tenure"]

CATEGORY_LEVELS: dict[str, list[str]] = {
    "contract_type": ["month_to_month", "one_year", "two_year"],
    "payment_method": ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
    "internet_service": ["fiber_optic", "dsl", "no_internet"],
    "plan_tier": ["basic", "standard", "premium"],
    "region": ["northeast", "southeast", "midwest", "southwest", "west"],
}

# Physically plausible ranges. Values outside are treated as data errors -> NaN -> imputed.
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (18, 100),
    "tenure_months": (0, 120),
    "monthly_charges": (0, 500),
    "total_charges": (0, 60_000),
    "num_support_tickets": (0, 50),
    "avg_monthly_usage_gb": (0, 5_000),
    "days_since_last_login": (0, 365),
    "num_addon_services": (0, 10),
    "satisfaction_score": (1, 5),
    "late_payments_12m": (0, 12),
}

# --------------------------------------------------------------------------------------
# Model search spaces (sampled with RandomizedSearchCV over StratifiedKFold)
# --------------------------------------------------------------------------------------
LOGREG_PARAM_GRID: dict[str, list] = {
    "model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
}

XGB_BASE_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": 1,  # parallelism lives at the CV level; n_jobs=1 also minimises p95 latency
    "random_state": RANDOM_STATE,
}

XGB_PARAM_DISTRIBUTIONS: dict = {
    "model__n_estimators": randint(200, 700),
    "model__learning_rate": loguniform(0.02, 0.15),
    "model__max_depth": randint(3, 7),
    "model__min_child_weight": randint(1, 12),
    "model__subsample": uniform(0.65, 0.35),
    "model__colsample_bytree": uniform(0.5, 0.5),
    "model__reg_lambda": loguniform(0.3, 20.0),
    "model__gamma": uniform(0.0, 1.0),
}
XGB_SEARCH_ITERATIONS = 30

# --------------------------------------------------------------------------------------
# Business cost model for threshold tuning
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Economics of a retention campaign.

    Every flagged customer receives an offer costing `offer_cost`. A true churner who
    receives the offer is retained with probability `retention_success_rate`, recovering
    `customer_lifetime_value`. Unflagged customers cost nothing extra.
    """

    offer_cost: float = 50.0
    customer_lifetime_value: float = 500.0
    retention_success_rate: float = 0.40

    @property
    def true_positive_value(self) -> float:
        return self.retention_success_rate * self.customer_lifetime_value - self.offer_cost

    @property
    def false_positive_cost(self) -> float:
        return self.offer_cost

    @property
    def bayes_optimal_threshold(self) -> float:
        """Threshold minimising expected cost for a perfectly calibrated model.

        Flag when p * TP_value - (1 - p) * FP_cost > 0  =>  p > FP / (TP + FP).
        """
        denom = self.true_positive_value + self.false_positive_cost
        return self.false_positive_cost / denom if denom > 0 else 1.0


COST_MODEL = CostModel()

# Risk tiers expressed as calibrated churn probabilities. Anything at or above the
# decision threshold is at least "medium" so the tier never contradicts the prediction.
HIGH_RISK_PROBABILITY = 0.60


# --------------------------------------------------------------------------------------
# Artifact layout
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def model(self) -> Path:
        return self.root / "model_pipeline.joblib"

    @property
    def baseline_model(self) -> Path:
        return self.root / "baseline_pipeline.joblib"

    @property
    def metadata(self) -> Path:
        return self.root / "model_metadata.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"

    @property
    def test_predictions(self) -> Path:
        return self.root / "test_predictions.csv"

    @property
    def oof_predictions(self) -> Path:
        return self.root / "oof_predictions.csv"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @classmethod
    def from_env(cls) -> "ArtifactPaths":
        return cls(Path(os.getenv("CHURN_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)))

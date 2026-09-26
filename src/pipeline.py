"""Leakage-free preprocessing + model pipeline.

The serialized artifact is a single scikit-learn ``Pipeline``:

    SchemaEnforcer -> FeatureEngineer -> ColumnTransformer -> PlattCalibratedClassifier

Every stateful step (imputation medians, scaler moments, one-hot vocabularies, the
calibration map) is fit inside ``Pipeline.fit``. During cross-validation each fold refits
the whole chain on its own training rows, so no statistic from a validation fold can leak
into training. Serving uses the exact same object, which removes training/serving skew.
"""

from __future__ import annotations

import re
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from src.config import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    ENGINEERED_FEATURES,
    NUMERIC_BOUNDS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    XGB_BASE_PARAMS,
)

ModelName = Literal["logistic_regression", "xgboost"]

_BOOL_MAP = {
    "true": 1.0,
    "t": 1.0,
    "yes": 1.0,
    "y": 1.0,
    "1": 1.0,
    "1.0": 1.0,
    "false": 0.0,
    "f": 0.0,
    "no": 0.0,
    "n": 0.0,
    "0": 0.0,
    "0.0": 0.0,
}


class SchemaError(ValueError):
    """Raised when inference input violates the feature contract in a non-recoverable way."""


class SchemaEnforcer(TransformerMixin, BaseEstimator):
    """Validate and normalise raw input against the feature contract.

    Policy for schema drift:
      * missing required column  -> ``SchemaError`` (fail loudly; silent imputation of an
        entire feature would hide an upstream outage),
      * unexpected extra columns -> dropped,
      * column order             -> irrelevant (output is always in contract order),
      * numeric dtype drift      -> coerced (``"12"`` -> 12.0; unparseable -> NaN),
      * impossible values        -> NaN (e.g. negative tenure), then imputed downstream,
      * category formatting      -> normalised (``" Month-To-Month"`` -> ``"month_to_month"``),
      * unseen categories        -> passed through; the encoder maps them to all-zeros.
    """

    def __init__(
        self,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        boolean_features: list[str] | None = None,
        numeric_bounds: dict[str, tuple[float, float]] | None = None,
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.boolean_features = boolean_features
        self.numeric_bounds = numeric_bounds

    def fit(self, X, y=None):
        self.numeric_features_ = list(self.numeric_features or NUMERIC_FEATURES)
        self.categorical_features_ = list(self.categorical_features or CATEGORICAL_FEATURES)
        self.boolean_features_ = list(self.boolean_features or BOOLEAN_FEATURES)
        self.numeric_bounds_ = dict(
            NUMERIC_BOUNDS if self.numeric_bounds is None else self.numeric_bounds
        )
        self.required_columns_ = (
            self.numeric_features_ + self.categorical_features_ + self.boolean_features_
        )
        self.n_features_in_ = len(self.required_columns_)
        self._check_columns(self._as_frame(X))
        return self

    def transform(self, X) -> pd.DataFrame:
        check_is_fitted(self, "required_columns_")
        X = self._as_frame(X)
        self._check_columns(X)
        raw = {col: X[col].to_numpy(dtype=object) for col in self.required_columns_}
        return pd.DataFrame(self.normalise_columns(raw), index=X.index)

    def normalise_columns(self, columns: Mapping[str, Sequence]) -> dict[str, np.ndarray]:
        """Column-wise normalisation shared by ``transform`` and the compiled serving path.

        Implemented with NumPy / plain Python rather than per-column pandas operations,
        whose fixed per-call overhead dominated single-row latency.
        """
        out: dict[str, np.ndarray] = {}
        for col in self.numeric_features_:
            lo, hi = self.numeric_bounds_.get(col, (-np.inf, np.inf))
            out[col] = coerce_numeric(columns[col], lo, hi)
        for col in self.categorical_features_:
            out[col] = np.array([normalise_category(v) for v in columns[col]], dtype=object)
        for col in self.boolean_features_:
            values = columns[col]
            out[col] = np.fromiter((parse_bool(v) for v in values), dtype=float, count=len(values))
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "required_columns_")
        return np.asarray(self.required_columns_, dtype=object)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, dict):
            return pd.DataFrame([X])
        if isinstance(X, list) and (not X or isinstance(X[0], dict)):
            return pd.DataFrame.from_records(X)
        raise SchemaError(
            f"Expected a pandas DataFrame or list of records, got {type(X).__name__}."
        )

    def _check_columns(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns_ if c not in X.columns]
        if missing:
            raise SchemaError(f"Input is missing required feature column(s): {missing}")


_WHITESPACE_OR_DASH = re.compile(r"[\s\-]+")
_NULL_TOKENS = frozenset({"", "nan", "none", "null"})


def coerce_numeric(values: Sequence, lo: float = -np.inf, hi: float = np.inf) -> np.ndarray:
    """To float; unparseable, non-finite, or out-of-bounds values become NaN."""
    values = list(values)
    try:  # numbers, numeric strings, None
        out = np.array(values, dtype=float)
    except (TypeError, ValueError):  # e.g. "n/a" or pd.NA
        series = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce")
        out = series.to_numpy(dtype=float, na_value=np.nan, copy=True)  # CoW: writable copy
    with np.errstate(invalid="ignore"):
        out[~np.isfinite(out) | (out < lo) | (out > hi)] = np.nan
    return out


def normalise_category(value) -> object:
    """``" Month-To-Month"`` -> ``"month_to_month"``; missing -> ``np.nan`` (sklearn's sentinel)."""
    if value is None or value is pd.NA or (isinstance(value, float) and value != value):
        return np.nan
    text = _WHITESPACE_OR_DASH.sub("_", str(value).strip().lower())
    return np.nan if text in _NULL_TOKENS else text


def parse_bool(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if value in (0, 1) else np.nan
    if isinstance(value, str):
        return _BOOL_MAP.get(value.strip().lower(), np.nan)
    return np.nan


def derive_features(
    tenure: np.ndarray, monthly: np.ndarray, total: np.ndarray
) -> dict[str, np.ndarray]:
    """Engineered features, shared by ``FeatureEngineer`` and the compiled serving path."""
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_spend = np.where(tenure > 0, total / tenure, np.where(tenure == 0, monthly, np.nan))
        increase = np.where(avg_spend > 0, monthly / avg_spend - 1.0, np.nan)
    early = np.where(np.isnan(tenure), np.nan, (tenure <= 6).astype(float))
    return {
        "avg_monthly_spend": avg_spend,
        "charge_increase_pct": increase,
        "is_early_tenure": early,
    }


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Stateless domain features. Holds no fitted statistics, so it cannot leak.

    * ``avg_monthly_spend``   - lifetime billing / tenure (historical price level)
    * ``charge_increase_pct`` - current bill vs. historical average; surfaces recent price
      increases that are otherwise invisible in any single raw column
    * ``is_early_tenure``     - first six months, where churn hazard is highest
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        columns = {col: X[col].to_numpy() for col in X.columns}
        columns.update(
            derive_features(
                X["tenure_months"].to_numpy(dtype=float),
                X["monthly_charges"].to_numpy(dtype=float),
                X["total_charges"].to_numpy(dtype=float),
            )
        )
        return pd.DataFrame(columns, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return np.concatenate([self.feature_names_in_, np.asarray(ENGINEERED_FEATURES, object)])


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Imputation + encoding (+ optional scaling), fit per training fold."""
    numeric_steps: list[tuple[str, object]] = [
        # add_indicator keeps "was missing" as signal: survey non-response is informative.
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    boolean = SimpleImputer(strategy="most_frequent")

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bool", boolean, BOOLEAN_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _margin(estimator, X) -> np.ndarray:
    """Raw log-odds score of a fitted binary classifier."""
    if hasattr(estimator, "get_booster"):
        return estimator.predict(X, output_margin=True)
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    p = np.clip(estimator.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _fit_fold_margin(estimator, X, y, train_idx, val_idx):
    fold_model = clone(estimator).fit(X[train_idx], y[train_idx])
    return val_idx, _margin(fold_model, X[val_idx])


class PlattCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """Platt scaling fit on out-of-fold log-odds.

    ``p = sigmoid(slope * margin(x) + intercept)`` where (slope, intercept) are learned from
    cross-validated margins, so the calibrator never sees in-sample scores. Because the
    map is affine in log-odds, TreeSHAP attributions of the base model stay exact after
    calibration: phi_calibrated = slope * phi and base = slope * base + intercept.
    The map is monotone (slope > 0), so ranking metrics (ROC-AUC, PR-AUC) are unchanged.
    """

    def __init__(
        self,
        estimator=None,
        cv: int = CV_FOLDS,
        random_state: int = RANDOM_STATE,
        n_jobs: int | None = None,
    ):
        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("PlattCalibratedClassifier supports binary targets only.")

        splitter = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        folds = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_fold_margin)(self.estimator, X, y, train_idx, val_idx)
            for train_idx, val_idx in splitter.split(X, y)
        )
        oof_margin = np.empty(len(y), dtype=float)
        for val_idx, margin in folds:
            oof_margin[val_idx] = margin

        calibrator = LogisticRegression(C=1e4, max_iter=1000)
        calibrator.fit(oof_margin.reshape(-1, 1), y)
        self.slope_ = float(calibrator.coef_[0, 0])
        self.intercept_ = float(calibrator.intercept_[0])

        self.estimator_ = clone(self.estimator).fit(X, y)
        self.feature_means_ = X.mean(axis=0)  # background for exact linear SHAP
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.slope_ * _margin(self.estimator_, np.asarray(X, dtype=float)) + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        p = expit(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def make_estimator(model_name: ModelName, params: dict | None = None):
    params = dict(params or {})
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=5000, **params)
    if model_name == "xgboost":
        return XGBClassifier(**{**XGB_BASE_PARAMS, **params})
    raise ValueError(f"Unknown model: {model_name!r}")


def build_model_pipeline(
    model_name: ModelName, params: dict | None = None, calibrate: bool = False
) -> Pipeline:
    """Full raw-input -> probability pipeline. ``params`` are estimator kwargs (no prefix)."""
    estimator = make_estimator(model_name, params)
    if calibrate:
        estimator = PlattCalibratedClassifier(estimator)
    return Pipeline(
        [
            ("schema", SchemaEnforcer()),
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor(scale_numeric=model_name != "xgboost")),
            ("model", estimator),
        ]
    )

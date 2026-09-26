"""Low-latency inference path compiled from a *fitted* sklearn pipeline.

sklearn's ``ColumnTransformer`` and pandas carry a fixed per-call overhead (input
validation, DataFrame construction, joblib dispatch) that dominates single-row latency: it
costs about the same for 1 row as for 1,000. ``CompiledPipeline`` reads the fitted state out
of the serialized pipeline -- imputation medians, missing-indicator columns, scaler moments,
one-hot vocabularies, calibration slope/intercept, the booster -- and applies it with plain
NumPy on records. The same helpers (``normalise_columns``, ``derive_features``) implement the
schema and feature logic in both paths.

Guarantees:
  * No new artifact: everything is derived from ``model_pipeline.joblib`` at load time.
  * Anything outside the supported step set raises ``UnsupportedPipelineError`` so callers
    fall back to ``pipeline.predict_proba``.
  * ``verify_parity`` compares both paths; the service runs it at startup and the test
    suite runs it on thousands of rows, including null / malformed / unseen-category inputs.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipeline import FeatureEngineer, PlattCalibratedClassifier, SchemaEnforcer, derive_features

BlockFn = Callable[[np.ndarray], np.ndarray]


class UnsupportedPipelineError(TypeError):
    """The fitted pipeline contains a step the compiled path does not implement."""


def _is_missing(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "f":
        return np.isnan(values)
    return np.fromiter(
        (v is None or (isinstance(v, float) and v != v) for v in values.ravel()),
        dtype=bool,
        count=values.size,
    ).reshape(values.shape)


def _compile_imputer(imputer: SimpleImputer) -> BlockFn:
    if imputer.keep_empty_features is False and imputer.statistics_.dtype.kind == "f":
        if np.isnan(imputer.statistics_).any():
            raise UnsupportedPipelineError("imputer dropped all-missing features")
    stats = imputer.statistics_
    indicator_cols = imputer.indicator_.features_ if imputer.add_indicator else None
    numeric = stats.dtype.kind == "f"

    def apply(X: np.ndarray) -> np.ndarray:
        X = X.astype(float) if numeric else X.astype(object)
        missing = _is_missing(X)
        filled = np.where(missing, stats, X)
        if indicator_cols is not None:
            filled = np.hstack([filled, missing[:, indicator_cols].astype(float)])
        return filled

    return apply


def _compile_scaler(scaler: StandardScaler) -> BlockFn:
    mean = scaler.mean_ if scaler.with_mean else 0.0
    scale = scaler.scale_ if scaler.with_std else 1.0
    return lambda X: (X.astype(float) - mean) / scale


def _compile_one_hot(encoder: OneHotEncoder) -> BlockFn:
    if encoder.drop is not None or getattr(encoder, "infrequent_categories_", None):
        raise UnsupportedPipelineError("one-hot drop/infrequent categories not supported")
    if encoder.handle_unknown not in ("ignore", "infrequent_if_exist"):
        raise UnsupportedPipelineError("one-hot handle_unknown must ignore unseen levels")
    lookups, offset = [], 0
    for cats in encoder.categories_:
        lookups.append((offset, {c: i for i, c in enumerate(cats.tolist())}))
        offset += len(cats)
    width = offset

    def apply(X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], width))
        for j, (start, index) in enumerate(lookups):
            for row, value in enumerate(X[:, j]):
                position = index.get(value)
                if position is not None:
                    out[row, start + position] = 1.0
        return out

    return apply


def _compile_transformer(transformer) -> list[BlockFn]:
    steps = transformer.steps if isinstance(transformer, Pipeline) else [(None, transformer)]
    compiled = []
    for _, step in steps:
        if isinstance(step, SimpleImputer) and step.strategy in (
            "median",
            "mean",
            "most_frequent",
            "constant",
        ):
            compiled.append(_compile_imputer(step))
        elif isinstance(step, StandardScaler):
            compiled.append(_compile_scaler(step))
        elif isinstance(step, OneHotEncoder):
            compiled.append(_compile_one_hot(step))
        else:
            raise UnsupportedPipelineError(f"unsupported preprocessing step {type(step).__name__}")
    return compiled


class CompiledPipeline:
    def __init__(self, pipeline: Pipeline):
        names = [name for name, _ in pipeline.steps]
        if names != ["schema", "features", "preprocess", "model"]:
            raise UnsupportedPipelineError(f"unexpected pipeline layout {names}")
        schema, features, preprocess, model = (step for _, step in pipeline.steps)
        if not isinstance(schema, SchemaEnforcer) or not isinstance(features, FeatureEngineer):
            raise UnsupportedPipelineError("unexpected schema/feature steps")
        if not isinstance(preprocess, ColumnTransformer):
            raise UnsupportedPipelineError("preprocess step must be a ColumnTransformer")
        if not isinstance(model, PlattCalibratedClassifier):
            raise UnsupportedPipelineError("model must be a PlattCalibratedClassifier")

        self.schema = schema
        self.required_columns = list(schema.required_columns_)
        self.blocks: list[tuple[list[str], list[BlockFn]]] = []
        for name, transformer, columns in preprocess.transformers_:
            if name == "remainder":
                if transformer != "drop":
                    raise UnsupportedPipelineError("remainder must be 'drop'")
                continue
            self.blocks.append((list(columns), _compile_transformer(transformer)))
        self.n_outputs = len(preprocess.get_feature_names_out())

        self.slope, self.intercept = model.slope_, model.intercept_
        self._margin = self._compile_margin(model.estimator_)

    @staticmethod
    def _compile_margin(estimator) -> Callable[[np.ndarray], np.ndarray]:
        if hasattr(estimator, "get_booster"):
            booster = estimator.get_booster()
            try:
                iteration_range = (0, estimator.best_iteration + 1)
            except AttributeError:  # no early stopping: use every tree
                iteration_range = (0, 0)
            return lambda X: booster.inplace_predict(
                X, predict_type="margin", iteration_range=iteration_range
            )
        if isinstance(estimator, LogisticRegression):
            coef, intercept = estimator.coef_[0], estimator.intercept_[0]
            return lambda X: X @ coef + intercept
        raise UnsupportedPipelineError(f"unsupported estimator {type(estimator).__name__}")

    def transform(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        raw = {col: [r.get(col) for r in records] for col in self.required_columns}
        columns = self.schema.normalise_columns(raw)
        columns.update(
            derive_features(
                columns["tenure_months"], columns["monthly_charges"], columns["total_charges"]
            )
        )
        parts = []
        for names, steps in self.blocks:
            block = np.column_stack([columns[name] for name in names])
            for step in steps:
                block = step(block)
            parts.append(block)
        out = np.hstack(parts).astype(float)
        if out.shape[1] != self.n_outputs:
            raise UnsupportedPipelineError("compiled output width differs from the pipeline")
        return out

    def predict_proba(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Calibrated P(churn) for each record (1-D array)."""
        margin = np.asarray(self._margin(self.transform(records)), dtype=float).reshape(-1)
        return expit(self.slope * margin + self.intercept)


def verify_parity(
    pipeline: Pipeline, compiled: CompiledPipeline, records: Sequence[Mapping[str, Any]]
) -> float:
    """Max absolute probability difference between the compiled and sklearn paths."""
    import pandas as pd

    reference = pipeline.predict_proba(pd.DataFrame.from_records(records))[:, 1]
    return float(np.max(np.abs(compiled.predict_proba(records) - reference)))

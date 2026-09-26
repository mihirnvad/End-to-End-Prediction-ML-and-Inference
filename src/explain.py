"""Exact SHAP attributions for the serialized pipeline, without the ``shap`` runtime.

* Tree models use XGBoost's native TreeSHAP (``pred_contribs=True``): exact, fast,
  and it keeps the serving image free of numba/llvmlite.
* Linear models use the closed form phi_j = w_j * (x_j - E[x_j]).

Attributions are computed on the one-hot / imputed model inputs, then summed back to the
feature a caller actually sent (e.g. the three ``contract_type_*`` columns -> ``contract_type``;
``missingindicator_satisfaction_score`` -> ``satisfaction_score``). SHAP values are additive,
so grouping preserves the identity  base_value + sum(phi) = calibrated log-odds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from sklearn.pipeline import Pipeline

from src.pipeline import PlattCalibratedClassifier


@dataclass
class PipelineExplanation:
    values: np.ndarray  # (n_samples, n_features) contributions in calibrated log-odds
    base_value: float  # calibrated log-odds of the average customer
    feature_names: list[str]
    data: pd.DataFrame  # feature values the attributions refer to (post-validation)
    probabilities: np.ndarray


def _source_feature(output_name: str, candidates: list[str]) -> str:
    name = output_name.removeprefix("missingindicator_")
    matches = [c for c in candidates if name == c or name.startswith(f"{c}_")]
    if not matches:
        raise ValueError(f"Cannot map transformed feature {output_name!r} to an input feature")
    return max(matches, key=len)


class PipelineExplainer:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.feature_stage = pipeline[:-2]  # schema enforcement + feature engineering
        self.preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]

        if isinstance(model, PlattCalibratedClassifier):
            self.slope, self.intercept = model.slope_, model.intercept_
            self.estimator = model.estimator_
            self.background_means = model.feature_means_
        else:
            self.slope, self.intercept = 1.0, 0.0
            self.estimator = model
            self.background_means = None

        self.feature_names = list(self.feature_stage.get_feature_names_out())
        self._grouping = self._build_grouping()

    def _build_grouping(self) -> np.ndarray:
        """(n_transformed, n_features) 0/1 matrix mapping model inputs to raw features."""
        index = {name: i for i, name in enumerate(self.feature_names)}
        n_out = len(self.preprocessor.get_feature_names_out())
        grouping = np.zeros((n_out, len(self.feature_names)))
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == "remainder":
                continue
            out_slice = self.preprocessor.output_indices_[name]
            out_names = transformer.get_feature_names_out(columns)
            for offset, out_name in enumerate(out_names):
                grouping[out_slice.start + offset, index[_source_feature(out_name, columns)]] = 1
        return grouping

    def _transformed_contributions(self, Xt: np.ndarray) -> tuple[np.ndarray, float]:
        if hasattr(self.estimator, "get_booster"):
            contribs = self.estimator.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
            return contribs[:, :-1], float(contribs[0, -1])
        if hasattr(self.estimator, "coef_"):
            if self.background_means is None:
                raise ValueError("Linear SHAP requires background feature means.")
            coef = self.estimator.coef_[0]
            contribs = (Xt - self.background_means) * coef
            base = float(self.estimator.intercept_[0] + coef @ self.background_means)
            return contribs, base
        raise TypeError(f"No exact explainer for {type(self.estimator).__name__}")

    def explain(self, X) -> PipelineExplanation:
        frame = self.feature_stage.transform(X)
        Xt = np.asarray(self.preprocessor.transform(frame), dtype=float)
        contribs, base = self._transformed_contributions(Xt)

        values = self.slope * (contribs @ self._grouping)
        base_value = self.slope * base + self.intercept
        probabilities = expit(base_value + values.sum(axis=1))
        return PipelineExplanation(
            values=values,
            base_value=float(base_value),
            feature_names=self.feature_names,
            data=frame[self.feature_names].reset_index(drop=True),
            probabilities=probabilities,
        )

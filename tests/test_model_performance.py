"""Model regression tests: quality gates that fail CI if a code change degrades the model."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import DEFAULT_DATA_PATH, ArtifactPaths
from src.data import load_splits

MIN_HOLDOUT_ROC_AUC = 0.80
PRODUCTION_MIN_ROC_AUC = 0.85


class TestTrainedModelQuality:
    def test_holdout_roc_auc_above_minimum(self, pipeline, splits):
        p = pipeline.predict_proba(splits.X_test)[:, 1]
        auc = roc_auc_score(splits.y_test, p)
        assert auc > MIN_HOLDOUT_ROC_AUC, f"hold-out ROC-AUC {auc:.4f}"

    def test_probabilities_beat_prevalence_forecast(self, pipeline, splits):
        """Brier skill score > 0: better than always predicting the base rate."""
        p = pipeline.predict_proba(splits.X_test)[:, 1]
        prevalence = np.full_like(p, splits.y_train.mean())
        assert brier_score_loss(splits.y_test, p) < 0.8 * brier_score_loss(
            splits.y_test, prevalence
        )

    def test_calibration_map_is_monotone(self, pipeline):
        assert pipeline.named_steps["model"].slope_ > 0

    def test_decision_threshold_is_sane(self, artifacts):
        metadata = json.loads(artifacts.metadata.read_text())
        assert 0.05 <= metadata["decision_threshold"] <= 0.6

    def test_metrics_report_is_complete(self, artifacts):
        metrics = json.loads(artifacts.metrics.read_text())
        champion = metrics["champion"]
        cv = metrics["cross_validation"][champion]["metrics"]
        assert set(cv) >= {"roc_auc", "pr_auc", "f1", "brier", "log_loss"}
        assert len(cv["roc_auc"]["folds"]) == 5
        ci = metrics["holdout_bootstrap_ci"]["roc_auc"][champion]
        assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"]

    def test_tuned_threshold_beats_default_threshold_economics(self, artifacts):
        business = json.loads(artifacts.metrics.read_text())["business_impact_holdout"]
        assert business["payoff_selected_threshold"] >= business["payoff_default_0_5_threshold"]
        assert business["payoff_selected_threshold"] > business["payoff_no_campaign"]


production = ArtifactPaths.from_env()


@pytest.mark.skipif(
    not (production.model.exists() and DEFAULT_DATA_PATH.exists()),
    reason="production artifacts not built (run data/generate_dataset.py and src/train.py)",
)
def test_production_artifact_meets_quality_bar():
    """Gate on the real 25K-row artifact when it exists locally or in CI."""
    model = joblib.load(production.model)
    splits = load_splits(DEFAULT_DATA_PATH)
    p = model.predict_proba(splits.X_test)[:, 1]
    auc = roc_auc_score(splits.y_test, p)
    assert auc > PRODUCTION_MIN_ROC_AUC, f"production hold-out ROC-AUC {auc:.4f}"

    reported = json.loads(production.metrics.read_text())
    champion = reported["champion"]
    assert auc == pytest.approx(reported["holdout"][champion]["roc_auc"], abs=1e-6)

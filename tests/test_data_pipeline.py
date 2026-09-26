"""Unit tests for the data contract, feature pipeline, explainer, and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from src.config import (
    CATEGORY_LEVELS,
    COST_MODEL,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    RAW_FEATURES,
    TARGET,
)
from src.explain import PipelineExplainer
from src.metrics import (
    bootstrap_ci,
    expected_calibration_error,
    optimal_thresholds,
    threshold_curve,
)
from src.pipeline import FeatureEngineer, SchemaEnforcer, SchemaError, build_model_pipeline
from src.serving import CompiledPipeline, UnsupportedPipelineError, verify_parity


def _proba(pipeline, frame: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(frame)[:, 1]


# --------------------------------------------------------------------------------------
# Dataset generator
# --------------------------------------------------------------------------------------
class TestDatasetGenerator:
    def test_is_deterministic_for_a_seed(self, generator):
        a = generator.generate_customers(500, seed=1)
        b = generator.generate_customers(500, seed=1)
        pd.testing.assert_frame_equal(a, b)

    def test_has_contract_columns_realistic_rate_and_missingness(self, generator):
        df = generator.generate_customers(5000, seed=3)
        assert list(df.columns) == ["customer_id", *RAW_FEATURES, TARGET]
        assert df["customer_id"].is_unique
        assert 0.15 < df[TARGET].mean() < 0.35
        assert df["satisfaction_score"].isna().mean() > 0.05
        assert df["avg_monthly_usage_gb"].isna().any()


# --------------------------------------------------------------------------------------
# Schema enforcement (schema drift, nulls, invalid values)
# --------------------------------------------------------------------------------------
class TestSchemaEnforcer:
    @pytest.fixture
    def enforcer(self, raw_frame):
        return SchemaEnforcer().fit(raw_frame)

    def test_normalises_category_formatting(self, enforcer, raw_frame):
        raw_frame.loc[0, "contract_type"] = "  Month-To-Month "
        raw_frame.loc[0, "payment_method"] = "ELECTRONIC_CHECK"
        out = enforcer.transform(raw_frame)
        assert out.loc[0, "contract_type"] == "month_to_month"
        assert out.loc[0, "payment_method"] == "electronic_check"

    def test_missing_required_column_raises(self, enforcer, raw_frame):
        with pytest.raises(SchemaError, match="tenure_months"):
            enforcer.transform(raw_frame.drop(columns=["tenure_months"]))

    def test_extra_columns_dropped_and_column_order_irrelevant(self, enforcer, raw_frame):
        drifted = raw_frame.assign(new_upstream_field=123, another="x")
        drifted = drifted[list(reversed(drifted.columns))]
        out = enforcer.transform(drifted)
        assert list(out.columns) == RAW_FEATURES
        pd.testing.assert_frame_equal(out, enforcer.transform(raw_frame))

    def test_numeric_dtype_drift_is_coerced(self, enforcer, raw_frame):
        stringly = raw_frame.astype(str)
        stringly.loc[0, "monthly_charges"] = "not-a-number"
        out = enforcer.transform(stringly)
        assert out.loc[0, "tenure_months"] == raw_frame.loc[0, "tenure_months"]
        assert np.isnan(out.loc[0, "monthly_charges"])

    @pytest.mark.parametrize(
        "column, bad_value",
        [("tenure_months", -5), ("age", 250), ("satisfaction_score", 9), ("total_charges", np.inf)],
    )
    def test_impossible_values_become_missing(self, enforcer, raw_frame, column, bad_value):
        raw_frame[column] = raw_frame[column].astype(float)
        raw_frame.loc[0, column] = bad_value
        assert np.isnan(enforcer.transform(raw_frame).loc[0, column])

    @pytest.mark.parametrize(
        "value, expected",
        [(True, 1.0), (False, 0.0), ("yes", 1.0), ("No", 0.0), ("1", 1.0), ("maybe", np.nan)],
    )
    def test_boolean_parsing(self, enforcer, raw_frame, value, expected):
        raw_frame["has_partner"] = pd.Series([value], dtype=object)
        result = enforcer.transform(raw_frame).loc[0, "has_partner"]
        assert (np.isnan(result) and np.isnan(expected)) or result == expected

    def test_rejects_unsupported_input_type(self, enforcer):
        with pytest.raises(SchemaError):
            enforcer.transform(np.zeros((1, len(RAW_FEATURES))))


class TestFeatureEngineer:
    def test_derived_features(self):
        frame = pd.DataFrame(
            {
                "tenure_months": [0.0, 10.0, 10.0, np.nan],
                "monthly_charges": [80.0, 60.0, 60.0, 50.0],
                "total_charges": [0.0, 500.0, np.nan, 100.0],
            }
        )
        out = FeatureEngineer().fit(frame).transform(frame)
        assert out["avg_monthly_spend"].tolist()[:2] == [80.0, 50.0]
        assert out["charge_increase_pct"].iloc[0] == pytest.approx(0.0)
        assert out["charge_increase_pct"].iloc[1] == pytest.approx(0.2)
        assert out[["avg_monthly_spend", "charge_increase_pct"]].iloc[2].isna().all()
        assert out["is_early_tenure"].tolist()[:2] == [1.0, 0.0]
        assert np.isnan(out["is_early_tenure"].iloc[3])


# --------------------------------------------------------------------------------------
# Trained pipeline robustness
# --------------------------------------------------------------------------------------
class TestPipelineRobustness:
    def test_nulls_in_every_feature_still_score(self, pipeline, raw_frame):
        all_null = pd.DataFrame([{c: None for c in RAW_FEATURES}])
        p = _proba(pipeline, pd.concat([raw_frame, all_null], ignore_index=True))
        assert np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))

    def test_unknown_category_is_tolerated(self, pipeline, raw_frame):
        raw_frame.loc[0, "payment_method"] = "crypto_wallet"
        p = _proba(pipeline, raw_frame)
        assert np.isfinite(p).all()

    def test_formatting_variants_score_identically(self, pipeline, raw_frame):
        variant = raw_frame.copy()
        variant.loc[0, "contract_type"] = " MONTH-TO-MONTH"
        np.testing.assert_allclose(_proba(pipeline, variant), _proba(pipeline, raw_frame))

    def test_schema_drift_and_dtype_drift_do_not_change_scores(self, pipeline, splits):
        X = splits.X_test.head(200)
        drifted = (
            X.astype(object)
            .astype(str)
            .assign(unused_col=1)[["unused_col", *reversed(RAW_FEATURES)]]
        )
        drifted = drifted.replace({"nan": None})
        np.testing.assert_allclose(_proba(pipeline, drifted), _proba(pipeline, X), atol=1e-9)

    def test_missing_column_fails_loudly(self, pipeline, raw_frame):
        with pytest.raises(SchemaError):
            pipeline.predict_proba(raw_frame.drop(columns=["contract_type"]))


class TestLeakageFreeFitting:
    def test_preprocessing_statistics_come_from_training_rows_only(self, splits):
        """Imputer medians must reflect the rows the pipeline was fit on, nothing else."""
        X, y = splits.X_train, splits.y_train
        subset = X.index < len(X) // 3
        model = build_model_pipeline("logistic_regression", {"C": 1.0})
        model.fit(X[subset], y[subset])

        imputer = model.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
        medians = dict(zip(NUMERIC_FEATURES + ENGINEERED_FEATURES, imputer.statistics_))
        assert medians["monthly_charges"] == pytest.approx(
            X.loc[subset, "monthly_charges"].median()
        )
        assert medians["monthly_charges"] != pytest.approx(X["monthly_charges"].median())

    def test_one_hot_vocabulary_is_learned_not_hardcoded(self, splits):
        X, y = splits.X_train, splits.y_train
        no_dsl = X["internet_service"] != "dsl"
        model = build_model_pipeline("logistic_regression").fit(X[no_dsl], y[no_dsl])
        encoder = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
        internet_levels = encoder.categories_[1 + 1]  # contract, payment, internet
        assert "dsl" not in internet_levels
        assert set(internet_levels) <= set(CATEGORY_LEVELS["internet_service"])


# --------------------------------------------------------------------------------------
# Compiled serving path
# --------------------------------------------------------------------------------------
class TestCompiledPipeline:
    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict]:
        return frame.astype(object).where(frame.notna(), None).to_dict("records")

    def test_matches_sklearn_on_holdout(self, pipeline, splits):
        records = self._records(splits.X_test)
        assert verify_parity(pipeline, CompiledPipeline(pipeline), records) < 1e-6

    def test_matches_sklearn_on_messy_inputs(self, pipeline, raw_frame):
        base = self._records(raw_frame)[0]
        messy = [
            {c: None for c in RAW_FEATURES},
            {**base, "payment_method": "crypto_wallet", "region": None},
            {**base, "contract_type": " TWO-YEAR ", "has_partner": "no", "age": "41"},
            {**base, "tenure_months": 0, "total_charges": 0.0},
            {**base, "tenure_months": -1, "monthly_charges": "abc", "satisfaction_score": 11},
        ]
        assert verify_parity(pipeline, CompiledPipeline(pipeline), messy) < 1e-6

    def test_linear_champion_is_supported(self, splits):
        X, y = splits.X_train.head(2000), splits.y_train.head(2000)
        model = build_model_pipeline("logistic_regression", {"C": 1.0}, calibrate=True).fit(X, y)
        records = self._records(splits.X_test.head(300))
        assert verify_parity(model, CompiledPipeline(model), records) < 1e-9

    def test_unknown_pipeline_layout_is_rejected(self, splits):
        uncalibrated = build_model_pipeline("logistic_regression").fit(
            splits.X_train.head(500), splits.y_train.head(500)
        )
        with pytest.raises(UnsupportedPipelineError):
            CompiledPipeline(uncalibrated)


# --------------------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------------------
class TestExplainer:
    def test_shap_values_are_exactly_additive(self, pipeline, splits):
        X = splits.X_test.head(300)
        explanation = PipelineExplainer(pipeline).explain(X)
        expected = _proba(pipeline, X)
        np.testing.assert_allclose(explanation.probabilities, expected, rtol=1e-4, atol=1e-6)
        reconstructed = explanation.base_value + explanation.values.sum(axis=1)
        np.testing.assert_allclose(reconstructed, logit(expected), atol=1e-4)

    def test_attributions_are_grouped_by_input_feature(self, pipeline, raw_frame):
        explanation = PipelineExplainer(pipeline).explain(raw_frame)
        assert explanation.feature_names == RAW_FEATURES + ENGINEERED_FEATURES
        assert explanation.values.shape == (1, len(RAW_FEATURES) + len(ENGINEERED_FEATURES))

    def test_linear_models_are_explainable_too(self, splits):
        X, y = splits.X_train.head(2000), splits.y_train.head(2000)
        model = build_model_pipeline("logistic_regression", {"C": 1.0}, calibrate=True).fit(X, y)
        explanation = PipelineExplainer(model).explain(X.head(50))
        np.testing.assert_allclose(explanation.probabilities, _proba(model, X.head(50)), atol=1e-8)


# --------------------------------------------------------------------------------------
# Metrics & threshold economics
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def calibrated_sample():
    rng = np.random.default_rng(0)
    p = rng.beta(1.2, 3.0, 200_000)
    y = (rng.random(p.size) < p).astype(int)
    return y, p


class TestMetrics:
    def test_ece_near_zero_for_calibrated_scores(self, calibrated_sample):
        y, p = calibrated_sample
        assert expected_calibration_error(y, p) < 0.01
        assert expected_calibration_error(y, np.clip(p * 1.6, 0, 1)) > 0.05

    def test_business_threshold_recovers_bayes_optimum(self, calibrated_sample):
        y, p = calibrated_sample
        best = optimal_thresholds(threshold_curve(y, p, COST_MODEL))["business"]
        assert best == pytest.approx(COST_MODEL.bayes_optimal_threshold, abs=0.03)

    def test_threshold_curve_counts_are_consistent(self, calibrated_sample):
        y, p = calibrated_sample
        curve = threshold_curve(y[:1000], p[:1000])
        assert (curve[["tp", "fp", "fn", "tn"]].sum(axis=1) == 1000).all()
        assert curve["recall"].is_monotonic_decreasing

    def test_bootstrap_interval_brackets_estimate(self, calibrated_sample):
        y, p = calibrated_sample
        y, p = y[:3000], p[:3000]
        noisy = np.clip(p + np.random.default_rng(1).normal(0, 0.2, p.size), 0, 1)
        ci = bootstrap_ci(y, {"good": p, "noisy": noisy}, n_boot=100)
        assert ci["good"]["ci_low"] <= ci["good"]["estimate"] <= ci["good"]["ci_high"]
        assert ci["good_minus_noisy"]["ci_low"] > 0

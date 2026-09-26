"""Integration tests for the FastAPI inference service (in-process via TestClient)."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from fastapi.testclient import TestClient

from app.schemas import MAX_BATCH_SIZE
from app.service import ModelService, load_model_service


def _batch(customer: dict, n: int) -> dict:
    rng = np.random.default_rng(0)
    customers = []
    for i in range(n):
        c = dict(customer)
        c["customer_id"] = f"CUST-{i:06d}"
        c["tenure_months"] = int(rng.integers(0, 72))
        c["monthly_charges"] = float(np.round(rng.uniform(20, 140), 2))
        c["total_charges"] = float(np.round(c["tenure_months"] * c["monthly_charges"], 2))
        c["num_support_tickets"] = int(rng.integers(0, 6))
        c["satisfaction_score"] = int(rng.integers(1, 6))
        customers.append(c)
    return {"customers": customers}


# --------------------------------------------------------------------------------------
# Operations endpoints
# --------------------------------------------------------------------------------------
class TestOperations:
    def test_health_reports_loaded_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"]
        assert 0 < body["decision_threshold"] < 1
        assert body["uptime_seconds"] >= 0

    def test_model_info_exposes_threshold_calibration_and_metrics(self, client):
        body = client.get("/model/info").json()
        assert body["calibration"]["method"] == "platt_on_oof_log_odds"
        assert body["calibration"]["slope"] > 0
        assert body["holdout_metrics"]["roc_auc"] > 0.5
        assert set(body["thresholds"]) >= {"business_optimal", "f1_optimal"}

    def test_compiled_inference_path_is_active_and_matches_sklearn(
        self, client, artifact_dir, customer
    ):
        assert client.get("/model/info").json()["inference_path"] == "compiled"
        record = {k: v for k, v in customer.items() if k != "customer_id"}
        sklearn_only = ModelService(artifact_dir, use_fast_path=False)
        assert sklearn_only.compiled is None
        expected = sklearn_only.predict([record])[0]["churn_probability"]
        served = client.post("/predict", json=customer).json()["churn_probability"]
        assert served == pytest.approx(expected, abs=1e-6)

    def test_openapi_documents_all_routes(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/health", "/predict", "/predict/batch", "/explain", "/metrics"} <= set(paths)

    def test_latency_headers_on_every_response(self, client, customer):
        response = client.post("/predict", json=customer)
        for header in ("x-process-time-ms", "x-latency-p50-ms", "x-latency-p95-ms"):
            assert float(response.headers[header]) >= 0

    def test_metrics_endpoint_tracks_route_percentiles(self, client, customer):
        for _ in range(5):
            client.post("/predict", json=customer)
        routes = client.get("/metrics").json()["routes"]
        stats = routes["POST /predict"]
        assert stats["count"] >= 5
        assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]


# --------------------------------------------------------------------------------------
# Single prediction
# --------------------------------------------------------------------------------------
class TestPredict:
    def test_returns_calibrated_probability_decision_and_tier(self, client, customer):
        response = client.post("/predict", json=customer)
        assert response.status_code == 200
        body = response.json()
        assert body["customer_id"] == customer["customer_id"]
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["churn_prediction"] == (body["churn_probability"] >= body["decision_threshold"])
        assert body["risk_tier"] in {"low", "medium", "high"}
        assert (body["risk_tier"] == "low") == (not body["churn_prediction"])

    def test_nullable_fields_accept_null(self, client, customer):
        customer.update(satisfaction_score=None, total_charges=None, avg_monthly_usage_gb=None)
        assert client.post("/predict", json=customer).status_code == 200

    def test_customer_id_is_optional(self, client, customer):
        customer.pop("customer_id")
        response = client.post("/predict", json=customer)
        assert response.status_code == 200
        assert response.json()["customer_id"] is None

    def test_loyal_customer_scores_below_at_risk_customer(self, client, customer):
        loyal = dict(
            customer,
            contract_type="two_year",
            tenure_months=60,
            total_charges=60 * customer["monthly_charges"],
            satisfaction_score=5,
            num_support_tickets=0,
            days_since_last_login=1,
            payment_method="credit_card",
        )
        at_risk = client.post("/predict", json=customer).json()["churn_probability"]
        assert client.post("/predict", json=loyal).json()["churn_probability"] < at_risk

    @pytest.mark.parametrize(
        "mutation, field",
        [
            (lambda c: c.pop("tenure_months"), "tenure_months"),  # missing required
            (lambda c: c.update(contract_type="weekly"), "contract_type"),  # invalid category
            (lambda c: c.update(age=12), "age"),  # below range
            (lambda c: c.update(monthly_charges=-10.0), "monthly_charges"),  # negative
            (lambda c: c.update(satisfaction_score=7), "satisfaction_score"),  # above range
            (lambda c: c.update(tenure_months="12"), "tenure_months"),  # strict: no str->int
            (lambda c: c.update(paperless_billing="yes"), "paperless_billing"),  # strict bool
            (lambda c: c.update(favourite_colour="blue"), "favourite_colour"),  # extra field
        ],
        ids=[
            "missing",
            "bad-category",
            "age-range",
            "negative",
            "score-range",
            "str-int",
            "str-bool",
            "extra",
        ],
    )
    def test_invalid_payloads_return_422(self, client, customer, mutation, field):
        mutation(customer)
        response = client.post("/predict", json=customer)
        assert response.status_code == 422
        locations = [err["loc"] for err in response.json()["detail"]]
        assert any(field in loc for loc in locations)

    def test_malformed_json_returns_422(self, client):
        response = client.post(
            "/predict", content=b"{not json", headers={"content-type": "application/json"}
        )
        assert response.status_code == 422


# --------------------------------------------------------------------------------------
# Batch prediction
# --------------------------------------------------------------------------------------
class TestBatchPredict:
    def test_batch_matches_single_predictions_in_order(self, client, customer):
        payload = _batch(customer, 25)
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 25
        assert [p["customer_id"] for p in body["predictions"]] == [
            c["customer_id"] for c in payload["customers"]
        ]
        assert body["n_predicted_churners"] == sum(
            p["churn_prediction"] for p in body["predictions"]
        )
        for request_item, batch_item in zip(payload["customers"][:5], body["predictions"][:5]):
            single = client.post("/predict", json=request_item).json()
            assert single["churn_probability"] == pytest.approx(batch_item["churn_probability"])

    def test_empty_batch_rejected(self, client):
        assert client.post("/predict/batch", json={"customers": []}).status_code == 422

    def test_oversized_batch_rejected(self, client, customer):
        payload = _batch(customer, MAX_BATCH_SIZE + 1)
        assert client.post("/predict/batch", json=payload).status_code == 422

    def test_one_invalid_record_rejects_batch_with_its_index(self, client, customer):
        payload = _batch(customer, 3)
        payload["customers"][2]["region"] = "atlantis"
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"][:3] == ["body", "customers", 2]

    def test_batch_throughput(self, client, customer):
        payload = _batch(customer, MAX_BATCH_SIZE)
        client.post("/predict/batch", json=payload)  # warm-up
        started = time.perf_counter()
        response = client.post("/predict/batch", json=payload)
        elapsed = time.perf_counter() - started
        assert response.status_code == 200
        assert response.json()["count"] == MAX_BATCH_SIZE
        assert elapsed < 2.0, f"{MAX_BATCH_SIZE} rows took {elapsed:.2f}s"


# --------------------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------------------
class TestExplain:
    def test_shap_contributions_reconstruct_probability(self, client, customer):
        explanation = client.post("/explain", json=customer).json()
        prediction = client.post("/predict", json=customer).json()

        log_odds = explanation["base_value"] + sum(
            c["shap_value"] for c in explanation["contributions"]
        )
        assert 1 / (1 + math.exp(-log_odds)) == pytest.approx(
            prediction["churn_probability"], abs=1e-5
        )
        magnitudes = [abs(c["shap_value"]) for c in explanation["contributions"]]
        assert magnitudes == sorted(magnitudes, reverse=True)
        features = {c["feature"] for c in explanation["contributions"]}
        assert {"contract_type", "tenure_months", "charge_increase_pct"} <= features


# --------------------------------------------------------------------------------------
# Failure modes & latency
# --------------------------------------------------------------------------------------
class TestSklearnFallbackPath:
    @pytest.fixture
    def fallback_client(self, artifact_dir):
        from app.main import create_app

        load_model_service.cache_clear()  # force a fresh load that honours CHURN_FAST_PATH
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("CHURN_ARTIFACT_DIR", str(artifact_dir))
            mp.setenv("CHURN_FAST_PATH", "0")
            with TestClient(create_app()) as test_client:
                yield test_client
        load_model_service.cache_clear()

    def test_fallback_serves_identical_predictions(self, fallback_client, client, customer):
        assert fallback_client.get("/model/info").json()["inference_path"] == "sklearn"
        fallback = fallback_client.post("/predict", json=customer)
        assert fallback.status_code == 200
        compiled = client.post("/predict", json=customer).json()
        assert fallback.json()["churn_probability"] == pytest.approx(
            compiled["churn_probability"], abs=1e-6
        )


class TestServiceUnavailable:
    def test_health_and_predict_return_503_without_model(self, unavailable_client, customer):
        health = unavailable_client.get("/health")
        assert health.status_code == 503
        assert health.json()["model_loaded"] is False
        assert unavailable_client.post("/predict", json=customer).status_code == 503


def test_single_prediction_server_latency_p95(client, customer):
    """Server-side p95 guardrail (in-process). See scripts/benchmark_latency.py for HTTP."""
    timings = []
    for _ in range(200):
        response = client.post("/predict", json=customer)
        timings.append(float(response.headers["x-process-time-ms"]))
    p95 = float(np.percentile(timings, 95))
    assert p95 < 25.0, f"p95 {p95:.1f}ms"

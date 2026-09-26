"""Model loading, integrity checking, and cached inference.

The pipeline is deserialised once per process (``load_model_service`` is memoised), its
SHA-256 is checked against the training metadata, and a warm-up prediction runs before the
service reports healthy, so the first real request does not pay one-off initialisation costs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from app.schemas import RiskTier
from src.config import HIGH_RISK_PROBABILITY, RAW_FEATURES, ArtifactPaths
from src.explain import PipelineExplainer
from src.serving import CompiledPipeline, UnsupportedPipelineError, verify_parity

logger = logging.getLogger("churn.service")

# XGBoost scores in float32, so paths agree to ~1e-7; anything above 1e-6 is a real defect.
PARITY_TOLERANCE = 1e-6


class ModelIntegrityError(RuntimeError):
    """The model artifact does not match the checksum recorded at training time."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelService:
    def __init__(
        self,
        artifact_dir: Path | str,
        verify_checksum: bool = True,
        use_fast_path: bool | None = None,
    ):
        self.paths = ArtifactPaths(Path(artifact_dir))
        if not self.paths.model.exists() or not self.paths.metadata.exists():
            raise FileNotFoundError(
                f"No trained model in {self.paths.root}. Run `python src/train.py` first."
            )
        self.metadata: dict[str, Any] = json.loads(self.paths.metadata.read_text())

        if verify_checksum:
            actual = _sha256(self.paths.model)
            if actual != self.metadata["artifact_sha256"]:
                raise ModelIntegrityError(
                    f"Checksum mismatch for {self.paths.model}: expected "
                    f"{self.metadata['artifact_sha256'][:12]}..., got {actual[:12]}..."
                )

        self.pipeline = joblib.load(self.paths.model)
        self.threshold = float(self.metadata["decision_threshold"])
        self.version: str = self.metadata["model_version"]
        self.model_type: str = self.metadata["model_type"]
        self.loaded_at = time.time()
        if use_fast_path is None:
            use_fast_path = os.getenv("CHURN_FAST_PATH", "1") != "0"
        self.compiled = self._compile() if use_fast_path else None

    def _compile(self) -> CompiledPipeline | None:
        """Build the NumPy serving path; keep it only if it matches sklearn within tolerance.

        The fast path is an optimisation: any failure here degrades to the sklearn path
        instead of preventing the model from serving.
        """
        try:
            compiled = CompiledPipeline(self.pipeline)
            max_diff = verify_parity(self.pipeline, compiled, self._parity_probes())
        except UnsupportedPipelineError as exc:
            logger.warning("Compiled inference path unavailable (%s); using sklearn path", exc)
            return None
        except Exception:
            logger.exception("Compiled inference path failed its self-check; using sklearn path")
            return None
        if max_diff > PARITY_TOLERANCE:
            logger.error("Compiled path disagrees with sklearn (%.2e); disabled", max_diff)
            return None
        logger.info("Compiled inference path enabled (parity max |dp| = %.1e)", max_diff)
        return compiled

    def _parity_probes(self) -> list[dict[str, Any]]:
        """Typical, sparse, malformed, and boundary records for the startup parity check."""
        base = dict(self.metadata["features"]["defaults"])
        levels = self.metadata["features"]["category_levels"]
        probes = [base, {col: None for col in RAW_FEATURES}]
        probes += [{**base, "contract_type": level} for level in levels["contract_type"]]
        probes += [
            {**base, "payment_method": "crypto_wallet", "contract_type": " Month-To-Month "},
            {**base, "tenure_months": 0, "total_charges": 0.0, "satisfaction_score": None},
            {**base, "tenure_months": -3, "age": 250, "avg_monthly_usage_gb": None},
            {**base, "paperless_billing": "yes", "has_partner": 0, "monthly_charges": "79.5"},
        ]
        return probes

    # ------------------------------------------------------------------ inference
    def _frame(self, records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame.from_records(records, columns=RAW_FEATURES)

    def predict_proba(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if self.compiled is not None:
            return self.compiled.predict_proba(records)
        return self.pipeline.predict_proba(self._frame(records))[:, 1]

    def risk_tiers(self, probabilities: np.ndarray) -> list[RiskTier]:
        high_cut = max(HIGH_RISK_PROBABILITY, self.threshold)
        codes = np.select(
            [probabilities < self.threshold, probabilities < high_cut], [0, 1], default=2
        )
        tiers = (RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH)
        return [tiers[c] for c in codes]

    def predict(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Vectorised: one pipeline call regardless of batch size."""
        proba = self.predict_proba(records)
        tiers = self.risk_tiers(proba)
        return [
            {
                "churn_probability": float(p),
                "churn_prediction": bool(p >= self.threshold),
                "risk_tier": tier,
            }
            for p, tier in zip(proba, tiers)
        ]

    @cached_property
    def explainer(self) -> PipelineExplainer:
        return PipelineExplainer(self.pipeline)

    def explain(self, record: Mapping[str, Any]) -> dict[str, Any]:
        explanation = self.explainer.explain(self._frame([record]))
        values = explanation.values[0]
        row = explanation.data.iloc[0]
        order = np.argsort(-np.abs(values))
        contributions = [
            {
                "feature": explanation.feature_names[i],
                "value": _json_value(row.iloc[i]),
                "shap_value": float(values[i]),
            }
            for i in order
        ]
        base = explanation.base_value
        return {
            "churn_probability": float(explanation.probabilities[0]),
            "base_value": base,
            "base_probability": float(1.0 / (1.0 + np.exp(-base))),
            "contributions": contributions,
        }

    def warmup(self, n_calls: int = 3) -> None:
        record = self.metadata["features"]["defaults"]
        for _ in range(n_calls):
            self.predict([record])
        self.explain(record)

    # ------------------------------------------------------------------ metadata
    @cached_property
    def holdout_metrics(self) -> dict[str, Any] | None:
        if not self.paths.metrics.exists():
            return None
        metrics = json.loads(self.paths.metrics.read_text())
        return metrics.get("holdout", {}).get(self.model_type)


def _json_value(value: Any) -> float | str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.floating, np.integer, int, float)):
        return float(value)
    return str(value)


@lru_cache(maxsize=4)
def load_model_service(artifact_dir: str) -> ModelService:
    """Process-wide cache: each artifact directory is deserialised exactly once."""
    started = time.perf_counter()
    service = ModelService(artifact_dir)
    logger.info(
        "Loaded model %s from %s in %.0f ms",
        service.version,
        artifact_dir,
        (time.perf_counter() - started) * 1000,
    )
    return service

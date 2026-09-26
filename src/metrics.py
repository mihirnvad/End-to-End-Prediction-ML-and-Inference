"""Evaluation metrics: discrimination, calibration, threshold economics, bootstrap CIs."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import COST_MODEL, CostModel

DEFAULT_THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Weighted mean |observed rate - mean predicted probability| over equal-width bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15))),
        "ece": expected_calibration_error(y_true, y_prob),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }


def confusion_counts(y_true, y_prob, threshold: float) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_prob) >= threshold
    return {
        "tn": int((~y_true & ~y_pred).sum()),
        "fp": int((~y_true & y_pred).sum()),
        "fn": int((y_true & ~y_pred).sum()),
        "tp": int((y_true & y_pred).sum()),
    }


def campaign_payoff(tp, fp, cost_model: CostModel = COST_MODEL):
    """Net value of contacting every flagged customer, relative to running no campaign."""
    return tp * cost_model.true_positive_value - fp * cost_model.false_positive_cost


def threshold_curve(
    y_true,
    y_prob,
    cost_model: CostModel = COST_MODEL,
    thresholds: np.ndarray = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Vectorised precision / recall / F1 / business payoff at each candidate threshold."""
    y_true = np.asarray(y_true).astype(bool)
    y_prob = np.asarray(y_prob)
    thresholds = np.asarray(thresholds, dtype=float)

    pred = y_prob[None, :] >= thresholds[:, None]
    tp = (pred & y_true).sum(axis=1)
    fp = (pred & ~y_true).sum(axis=1)
    fn = (~pred & y_true).sum(axis=1)
    tn = (~pred & ~y_true).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)

    payoff = campaign_payoff(tp, fp, cost_model)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "flagged_rate": (tp + fp) / len(y_true),
            "payoff": payoff,
            "payoff_per_1k_customers": payoff / len(y_true) * 1000,
        }
    )


def optimal_thresholds(curve: pd.DataFrame) -> dict[str, float]:
    return {
        "business": float(curve.loc[curve["payoff"].idxmax(), "threshold"]),
        "f1": float(curve.loc[curve["f1"].idxmax(), "threshold"]),
    }


def bootstrap_ci(
    y_true,
    predictions: dict[str, np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CIs per model, plus a paired CI for (first - second) model.

    Resampling is stratified so every replicate keeps the original class balance.
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    names = list(predictions)
    scores = {name: np.empty(n_boot) for name in names}

    for i in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        )
        for name in names:
            scores[name][i] = metric(y_true[idx], np.asarray(predictions[name])[idx])

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {
        name: {
            "estimate": float(metric(y_true, np.asarray(predictions[name]))),
            "ci_low": float(np.percentile(s, lo_q)),
            "ci_high": float(np.percentile(s, hi_q)),
        }
        for name, s in scores.items()
    }
    if len(names) >= 2:
        diff = scores[names[0]] - scores[names[1]]
        out[f"{names[0]}_minus_{names[1]}"] = {
            "estimate": out[names[0]]["estimate"] - out[names[1]]["estimate"],
            "ci_low": float(np.percentile(diff, lo_q)),
            "ci_high": float(np.percentile(diff, hi_q)),
        }
    return out

"""Train, tune, calibrate, and select the churn model, then persist all artifacts.

Protocol
  1. Stratified 80/20 split. The hold-out set is untouched until step 6.
  2. Hyperparameter search per candidate with StratifiedKFold(5), scored on ROC-AUC.
  3. Re-run 5-fold CV for each tuned *calibrated* candidate to collect ROC-AUC, PR-AUC,
     F1, Brier, log-loss, ECE per fold, plus out-of-fold (OOF) probabilities.
  4. Champion = highest mean CV ROC-AUC.
  5. Decision threshold = argmax of expected campaign payoff on the champion's OOF
     probabilities (never on the hold-out set).
  6. Refit on the full training split; evaluate once on the hold-out set with
     stratified-bootstrap confidence intervals.

Usage:
    python src/train.py            # full search (30 XGBoost candidates)
    python src/train.py --fast     # CI / smoke-test mode
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

from src.config import (  # noqa: E402
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORY_LEVELS,
    COST_MODEL,
    CV_FOLDS,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_PATH,
    ENGINEERED_FEATURES,
    ID_COLUMN,
    LOGREG_PARAM_GRID,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    XGB_PARAM_DISTRIBUTIONS,
    XGB_SEARCH_ITERATIONS,
    ArtifactPaths,
)
from src.data import load_splits  # noqa: E402
from src.metrics import (  # noqa: E402
    bootstrap_ci,
    campaign_payoff,
    classification_metrics,
    confusion_counts,
    optimal_thresholds,
    threshold_curve,
)
from src.pipeline import SchemaEnforcer, build_model_pipeline  # noqa: E402

logger = logging.getLogger("churn.train")

CANDIDATES = ("logistic_regression", "xgboost")
CV_METRICS = ("roc_auc", "pr_auc", "f1", "brier", "log_loss", "ece")


def _builtin(value):
    """Make numpy scalars JSON-serialisable."""
    return value.item() if isinstance(value, np.generic) else value


def tune_hyperparameters(name, X, y, cv, n_iter: int, n_jobs: int, seed: int):
    pipeline = build_model_pipeline(name)
    if name == "logistic_regression":
        search = GridSearchCV(
            pipeline, LOGREG_PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=n_jobs, refit=False
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            XGB_PARAM_DISTRIBUTIONS,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            n_jobs=n_jobs,
            refit=False,
            random_state=seed,
        )
    search.fit(X, y)
    params = {k.removeprefix("model__"): _builtin(v) for k, v in search.best_params_.items()}
    return params, float(search.best_score_), len(search.cv_results_["params"])


def _fit_predict_fold(pipeline, X, y, train_idx, val_idx):
    model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
    return val_idx, model.predict_proba(X.iloc[val_idx])[:, 1]


def cross_validate_pipeline(pipeline, X, y, cv, n_jobs: int):
    """Per-fold metrics and out-of-fold probabilities from one pass over the folds."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_fold)(pipeline, X, y, tr, va) for tr, va in cv.split(X, y)
    )
    oof = np.empty(len(y), dtype=float)
    fold_metrics = []
    for val_idx, proba in results:
        oof[val_idx] = proba
        fold_metrics.append(classification_metrics(y.iloc[val_idx], proba))
    return fold_metrics, oof


def summarise_folds(fold_metrics: list[dict]) -> dict:
    summary = {}
    for metric in CV_METRICS:
        values = np.array([fold[metric] for fold in fold_metrics])
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "folds": [round(float(v), 5) for v in values],
        }
    return summary


def feature_profile(X_train: pd.DataFrame) -> tuple[dict, dict]:
    """Training-set defaults and ranges, used by the dashboard to seed its inputs."""
    clean = SchemaEnforcer().fit(X_train).transform(X_train)
    defaults, ranges = {}, {}
    for col in NUMERIC_FEATURES:
        series = clean[col].dropna()
        defaults[col] = float(series.median())
        ranges[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "p01": float(series.quantile(0.01)),
            "p99": float(series.quantile(0.99)),
        }
    for col in CATEGORICAL_FEATURES:
        defaults[col] = str(clean[col].mode().iloc[0])
    for col in BOOLEAN_FEATURES:
        defaults[col] = bool(clean[col].mode().iloc[0])
    return defaults, ranges


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(
    data_path: Path = DEFAULT_DATA_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    n_iter: int = XGB_SEARCH_ITERATIONS,
    n_boot: int = 1000,
    n_jobs: int = -1,
    seed: int = RANDOM_STATE,
) -> dict:
    started = time.perf_counter()
    paths = ArtifactPaths(Path(artifact_dir))
    paths.root.mkdir(parents=True, exist_ok=True)

    splits = load_splits(data_path, seed=seed)
    X_train, X_test, y_train, y_test = splits.X_train, splits.X_test, splits.y_train, splits.y_test
    logger.info(
        "Loaded %d rows | train=%d test=%d | churn rate %.1f%%",
        len(X_train) + len(X_test),
        len(X_train),
        len(X_test),
        100 * y_train.mean(),
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)

    # 2-3. Tune each candidate, then cross-validate its calibrated form -------------
    results: dict[str, dict] = {}
    for name in CANDIDATES:
        t0 = time.perf_counter()
        params, search_auc, n_candidates = tune_hyperparameters(
            name, X_train, y_train, cv, n_iter, n_jobs, seed
        )
        logger.info(
            "[%s] search: %d candidates x %d folds, best ROC-AUC %.4f (%.0fs) params=%s",
            name,
            n_candidates,
            CV_FOLDS,
            search_auc,
            time.perf_counter() - t0,
            params,
        )
        pipeline = build_model_pipeline(name, params, calibrate=True)
        fold_metrics, oof = cross_validate_pipeline(pipeline, X_train, y_train, cv, n_jobs)
        cv_summary = summarise_folds(fold_metrics)
        logger.info(
            "[%s] calibrated %d-fold CV: ROC-AUC %.4f +/- %.4f | PR-AUC %.4f | Brier %.4f",
            name,
            CV_FOLDS,
            cv_summary["roc_auc"]["mean"],
            cv_summary["roc_auc"]["std"],
            cv_summary["pr_auc"]["mean"],
            cv_summary["brier"]["mean"],
        )
        results[name] = {
            "params": params,
            "search": {"n_candidates": n_candidates, "best_roc_auc": search_auc},
            "cv": cv_summary,
            "oof": oof,
        }

    # 4. Champion selection ------------------------------------------------------------
    champion = max(CANDIDATES, key=lambda n: results[n]["cv"]["roc_auc"]["mean"])
    baseline = "logistic_regression"
    logger.info("Champion: %s", champion)

    # 5. Threshold tuning on OOF probabilities ----------------------------------------
    thresholds = {}
    for name in CANDIDATES:
        curve = threshold_curve(y_train, results[name]["oof"], COST_MODEL)
        thresholds[name] = optimal_thresholds(curve)
    selected_threshold = thresholds[champion]["business"]
    logger.info(
        "Threshold: business-optimal %.2f (Bayes-optimal for calibrated probs: %.2f), "
        "F1-optimal %.2f",
        selected_threshold,
        COST_MODEL.bayes_optimal_threshold,
        thresholds[champion]["f1"],
    )

    # 6. Refit on the full training split and score the hold-out set once --------------
    final_models, test_proba = {}, {}
    for name in CANDIDATES:
        model = build_model_pipeline(name, results[name]["params"], calibrate=True)
        model.set_params(model__n_jobs=n_jobs).fit(X_train, y_train)  # parallel calib folds
        final_models[name] = model.set_params(model__n_jobs=None)
        test_proba[name] = model.predict_proba(X_test)[:, 1]

    holdout = {
        name: classification_metrics(y_test, test_proba[name], thresholds[name]["business"])
        for name in CANDIDATES
    }
    holdout[champion]["confusion_matrix"] = confusion_counts(
        y_test, test_proba[champion], selected_threshold
    )
    ordered = {champion: test_proba[champion], baseline: test_proba[baseline]}
    if champion == baseline:
        ordered = {champion: test_proba[champion], "xgboost": test_proba["xgboost"]}
    ci = {
        "roc_auc": bootstrap_ci(y_test, ordered, roc_auc_score, n_boot=n_boot, seed=seed),
        "pr_auc": bootstrap_ci(y_test, ordered, average_precision_score, n_boot=n_boot, seed=seed),
    }

    test_curve = threshold_curve(y_test, test_proba[champion], COST_MODEL)
    n_pos, n_neg = int(y_test.sum()), int((1 - y_test).sum())

    def payoff_at(t: float) -> float:
        row = test_curve.iloc[(test_curve["threshold"] - t).abs().argmin()]
        return float(row["payoff"])

    business = {
        "cost_model": {
            "offer_cost": COST_MODEL.offer_cost,
            "customer_lifetime_value": COST_MODEL.customer_lifetime_value,
            "retention_success_rate": COST_MODEL.retention_success_rate,
        },
        "n_test_customers": len(y_test),
        "payoff_selected_threshold": payoff_at(selected_threshold),
        "payoff_default_0_5_threshold": payoff_at(0.5),
        "payoff_f1_threshold": payoff_at(thresholds[champion]["f1"]),
        "payoff_contact_everyone": float(campaign_payoff(n_pos, n_neg, COST_MODEL)),
        "payoff_no_campaign": 0.0,
    }

    # 7. Persist ------------------------------------------------------------------------
    joblib.dump(final_models[champion], paths.model)
    joblib.dump(final_models[baseline], paths.baseline_model)
    artifact_hash = _sha256(paths.model)
    trained_at = datetime.now(timezone.utc)
    model_version = f"{champion}-{trained_at:%Y%m%dT%H%M%SZ}-{artifact_hash[:8]}"
    calibrator = final_models[champion].named_steps["model"]
    defaults, ranges = feature_profile(X_train)

    metadata = {
        "model_version": model_version,
        "model_type": champion,
        "trained_at": trained_at.isoformat(),
        "artifact_sha256": artifact_hash,
        "decision_threshold": selected_threshold,
        "thresholds": {
            "business_optimal": selected_threshold,
            "f1_optimal": thresholds[champion]["f1"],
            "bayes_optimal_calibrated": round(COST_MODEL.bayes_optimal_threshold, 4),
        },
        "calibration": {
            "method": "platt_on_oof_log_odds",
            "slope": calibrator.slope_,
            "intercept": calibrator.intercept_,
        },
        "hyperparameters": results[champion]["params"],
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "boolean": BOOLEAN_FEATURES,
            "engineered": ENGINEERED_FEATURES,
            "category_levels": CATEGORY_LEVELS,
            "defaults": defaults,
            "ranges": ranges,
        },
        "training_data": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_churn_rate": float(y_train.mean()),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    metrics = {
        "champion": champion,
        "cross_validation": {
            name: {
                "cv_folds": CV_FOLDS,
                "hyperparameter_search": results[name]["search"],
                "hyperparameters": results[name]["params"],
                "metrics": results[name]["cv"],
            }
            for name in CANDIDATES
        },
        "thresholds": thresholds,
        "holdout": holdout,
        "holdout_bootstrap_ci": ci,
        "business_impact_holdout": business,
    }
    paths.metadata.write_text(json.dumps(metadata, indent=2))
    paths.metrics.write_text(json.dumps(metrics, indent=2))

    pd.DataFrame(
        {
            ID_COLUMN: splits.ids_test,
            TARGET: y_test,
            "champion_proba": test_proba[champion],
            "baseline_proba": test_proba[baseline],
        }
    ).to_csv(paths.test_predictions, index=False)
    pd.DataFrame(
        {
            ID_COLUMN: splits.ids_train,
            TARGET: y_train,
            "champion_proba": results[champion]["oof"],
            "baseline_proba": results[baseline]["oof"],
        }
    ).to_csv(paths.oof_predictions, index=False)

    _log_summary(metrics, champion, baseline, time.perf_counter() - started)
    logger.info("Artifacts written to %s (model %s)", paths.root, model_version)
    return metrics


def _log_summary(metrics: dict, champion: str, baseline: str, elapsed: float) -> None:
    rows = []
    for name in (champion, baseline) if champion != baseline else (champion,):
        cv_m = metrics["cross_validation"][name]["metrics"]
        ho = metrics["holdout"][name]
        rows.append(
            f"  {name:<20} CV ROC-AUC {cv_m['roc_auc']['mean']:.4f}+/-{cv_m['roc_auc']['std']:.4f}"
            f" | hold-out ROC-AUC {ho['roc_auc']:.4f} PR-AUC {ho['pr_auc']:.4f}"
            f" Brier {ho['brier']:.4f} F1 {ho['f1']:.4f} @ t={ho['threshold']:.2f}"
        )
    ci = metrics["holdout_bootstrap_ci"]["roc_auc"][champion]
    biz = metrics["business_impact_holdout"]
    logger.info(
        "Summary (%.0fs)\n%s\n  hold-out ROC-AUC 95%% CI [%.4f, %.4f]\n"
        "  campaign payoff on hold-out: tuned threshold $%s | t=0.5 $%s | contact-all $%s",
        elapsed,
        "\n".join(rows),
        ci["ci_low"],
        ci["ci_high"],
        f"{biz['payoff_selected_threshold']:,.0f}",
        f"{biz['payoff_default_0_5_threshold']:,.0f}",
        f"{biz['payoff_contact_everyone']:,.0f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn prediction pipeline.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--n-iter", type=int, default=XGB_SEARCH_ITERATIONS)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--fast", action="store_true", help="4 search candidates, 200 bootstraps")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    train(
        data_path=args.data_path,
        artifact_dir=args.artifact_dir,
        n_iter=4 if args.fast else args.n_iter,
        n_boot=200 if args.fast else args.n_boot,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

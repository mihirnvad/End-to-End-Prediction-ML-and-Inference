"""Generate evaluation figures for the trained champion (and baseline) pipelines.

Reloads the serialized artifacts (verifying the round-trip reproduces training-time
hold-out predictions), then writes high-resolution figures to ``artifacts/figures``:

    roc_curve.png            ROC curves with AUC, champion vs. baseline
    pr_curve.png             Precision-recall curves with average precision
    confusion_matrix.png     Hold-out confusion matrix at the tuned threshold
    calibration_curve.png    Reliability diagram + predicted-probability histogram
    threshold_analysis.png   Campaign payoff and precision/recall/F1 vs. threshold
    shap_summary.png         Global SHAP beeswarm (TreeSHAP, grouped by input feature)
    shap_importance.png      Mean |SHAP| per input feature

Usage:
    python src/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    COST_MODEL,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_PATH,
    RANDOM_STATE,
    ArtifactPaths,
)
from src.data import load_splits  # noqa: E402
from src.explain import PipelineExplainer  # noqa: E402
from src.metrics import confusion_counts, threshold_curve  # noqa: E402

logger = logging.getLogger("churn.evaluate")

# Validated categorical slots (colour-blind-safe order); colour follows the model entity.
CHAMPION_COLOR = "#2a78d6"
BASELINE_COLOR = "#eb6834"
SERIES_3 = "#1baf7a"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DPI = 200

MODEL_LABELS = {"xgboost": "XGBoost", "logistic_regression": "Logistic Regression"}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "text.parse_math": False,  # "$50" is currency, not mathtext
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "font.size": 10,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_roc(y, probas: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, label="Chance (AUC = 0.500)")
    for name, color in (("champion", CHAMPION_COLOR), ("baseline", BASELINE_COLOR)):
        label, p = probas[name]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, label=f"{label} (AUC = {roc_auc_score(y, p):.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("ROC curve - hold-out set")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_pr(y, probas: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.2))
    prevalence = float(np.mean(y))
    ax.axhline(prevalence, color=AXIS, linewidth=1, label=f"No-skill (AP = {prevalence:.3f})")
    for name, color in (("champion", CHAMPION_COLOR), ("baseline", BASELINE_COLOR)):
        label, p = probas[name]
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, color=color, label=f"{label} (AP = {ap:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision")
    ax.set_title("Precision-recall curve - hold-out set")
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_confusion(y, p, threshold: float, path: Path) -> None:
    c = confusion_counts(y, p, threshold)
    counts = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    rates = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.imshow(rates, cmap=BLUE_RAMP, vmin=0, vmax=1)
    ax.grid(False)
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            text_color = "#ffffff" if rates[i, j] > 0.55 else INK
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{counts[i, j]:,}\n({rates[i, j]:.1%} of row)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )
    ax.set_xticks([0, 1], ["Predicted: stays", "Predicted: churns"])
    ax.set_yticks([0, 1], ["Actual: stays", "Actual: churns"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Confusion matrix @ tuned threshold {threshold:.2f}")
    _save(fig, path)


def plot_calibration(y, probas: dict[str, np.ndarray], path: Path) -> None:
    fig, (ax, hist_ax) = plt.subplots(
        2, 1, figsize=(6, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, label="Perfect calibration")
    bins = np.linspace(0, 1, 21)
    for name, color in (("champion", CHAMPION_COLOR), ("baseline", BASELINE_COLOR)):
        label, p = probas[name]
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(
            mean_pred,
            frac_pos,
            color=color,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=f"{label} (Brier = {brier_score_loss(y, p):.4f})",
        )
        hist_ax.hist(p, bins=bins, color=color, histtype="step", linewidth=1.6, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed churn rate")
    ax.set_title("Calibration (reliability) - hold-out set")
    ax.legend(loc="upper left")
    hist_ax.set(xlabel="Predicted churn probability", ylabel="Customers")
    _save(fig, path)


def plot_threshold_analysis(
    y_test, p_test, y_oof, p_oof, selected: float, bayes: float, path: Path
) -> None:
    test_curve = threshold_curve(y_test, p_test, COST_MODEL)
    oof_curve = threshold_curve(y_oof, p_oof, COST_MODEL)

    fig, (ax_pay, ax_pr) = plt.subplots(2, 1, figsize=(7, 7.4), sharex=True)
    ax_pay.plot(
        oof_curve["threshold"],
        oof_curve["payoff_per_1k_customers"],
        color=SERIES_3,
        label="Training out-of-fold (used to select)",
    )
    ax_pay.plot(
        test_curve["threshold"],
        test_curve["payoff_per_1k_customers"],
        color=CHAMPION_COLOR,
        label="Hold-out (validation)",
    )
    ax_pay.axhline(0, color=AXIS, linewidth=1)
    # Label the larger threshold to the right of its line and the smaller to the left,
    # so the two annotations never collide when the thresholds are close.
    markers = sorted(
        [(selected, f"selected {selected:.2f}"), (bayes, f"Bayes-optimal {bayes:.2f}")]
    )
    for (x, text), (dx, ha) in zip(markers, ((-4, "right"), (4, "left"))):
        ax_pay.axvline(x, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
        ax_pay.annotate(
            text,
            xy=(x, 0.06),
            xycoords=("data", "axes fraction"),
            xytext=(dx, 0),
            textcoords="offset points",
            ha=ha,
            color=INK_SECONDARY,
            fontsize=8.5,
        )
    ax_pay.set(ylabel="Net campaign value per 1k customers")
    ax_pay.set_title(
        f"Threshold economics (offer ${COST_MODEL.offer_cost:.0f}, CLV "
        f"${COST_MODEL.customer_lifetime_value:.0f}, save rate "
        f"{COST_MODEL.retention_success_rate:.0%})"
    )
    ax_pay.legend(loc="lower right")

    ax_pay.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))

    for metric, label, color in (
        ("precision", "Precision", CHAMPION_COLOR),
        ("recall", "Recall", BASELINE_COLOR),
        ("f1", "F1", SERIES_3),
    ):
        ax_pr.plot(test_curve["threshold"], test_curve[metric], color=color, label=label)
    ax_pr.axvline(selected, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    ax_pr.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Decision threshold", ylabel="Score (hold-out)")
    ax_pr.legend(loc="lower left")
    _save(fig, path)


def plot_shap(explainer: PipelineExplainer, X: pd.DataFrame, figures: Path) -> pd.Series:
    explanation = explainer.explain(X)
    display = explanation.data.copy()
    for col in CATEGORICAL_FEATURES:
        display[col] = np.nan  # categorical levels have no low->high colour; render neutral
    shap_exp = shap.Explanation(
        values=explanation.values,
        base_values=np.full(len(X), explanation.base_value),
        data=display.to_numpy(dtype=float),
        feature_names=explanation.feature_names,
    )

    plt.figure()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)
        shap.plots.beeswarm(
            shap_exp, max_display=15, color=BLUE_RAMP, show=False, plot_size=(8, 6.5)
        )
    fig = plt.gcf()
    fig.axes[0].set_title("Global SHAP summary - hold-out sample", loc="left")
    fig.axes[0].set_xlabel("SHAP value (impact on calibrated churn log-odds)")
    _save(fig, figures / "shap_summary.png")

    importance = pd.Series(
        np.abs(explanation.values).mean(axis=0), index=explanation.feature_names
    ).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    top = importance.tail(15)
    ax.barh(top.index, top.values, color=CHAMPION_COLOR, height=0.62)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("Global feature importance (mean |SHAP|)")
    _save(fig, figures / "shap_importance.png")
    return importance.sort_values(ascending=False)


def evaluate(
    data_path: Path = DEFAULT_DATA_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    shap_samples: int = 2000,
) -> dict:
    _apply_style()
    paths = ArtifactPaths(Path(artifact_dir))
    paths.figures.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(paths.metadata.read_text())
    champion = joblib.load(paths.model)
    baseline = joblib.load(paths.baseline_model)
    splits = load_splits(data_path)
    y_test = splits.y_test.to_numpy()

    p_champion = champion.predict_proba(splits.X_test)[:, 1]
    p_baseline = baseline.predict_proba(splits.X_test)[:, 1]

    # Round-trip check: the reloaded artifact must reproduce training-time predictions.
    if paths.test_predictions.exists():
        saved = pd.read_csv(paths.test_predictions)
        max_diff = float(np.max(np.abs(saved["champion_proba"].to_numpy() - p_champion)))
        if max_diff > 1e-6:
            raise RuntimeError(f"Reloaded model disagrees with training output ({max_diff:.2e})")
        logger.info("Artifact round-trip verified (max |dp| = %.1e)", max_diff)

    champion_label = MODEL_LABELS.get(metadata["model_type"], metadata["model_type"])
    probas = {
        "champion": (f"{champion_label} (champion)", p_champion),
        "baseline": ("Logistic Regression (baseline)", p_baseline),
    }
    threshold = float(metadata["decision_threshold"])
    figures = paths.figures

    plot_roc(y_test, probas, figures / "roc_curve.png")
    plot_pr(y_test, probas, figures / "pr_curve.png")
    plot_confusion(y_test, p_champion, threshold, figures / "confusion_matrix.png")
    plot_calibration(y_test, probas, figures / "calibration_curve.png")

    oof = pd.read_csv(paths.oof_predictions)
    plot_threshold_analysis(
        y_test,
        p_champion,
        oof["churned"].to_numpy(),
        oof["champion_proba"].to_numpy(),
        threshold,
        metadata["thresholds"]["bayes_optimal_calibrated"],
        figures / "threshold_analysis.png",
    )

    sample = splits.X_test.sample(
        n=min(shap_samples, len(splits.X_test)), random_state=RANDOM_STATE
    )
    importance = plot_shap(PipelineExplainer(champion), sample, figures)

    summary = {
        "roc_auc": float(roc_auc_score(y_test, p_champion)),
        "pr_auc": float(average_precision_score(y_test, p_champion)),
        "brier": float(brier_score_loss(y_test, p_champion)),
        "top_features": importance.head(10).round(4).to_dict(),
    }
    logger.info(
        "Hold-out ROC-AUC %.4f | PR-AUC %.4f | Brier %.4f",
        summary["roc_auc"],
        summary["pr_auc"],
        summary["brier"],
    )
    logger.info("Top SHAP features: %s", ", ".join(importance.head(8).index))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation figures.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--shap-samples", type=int, default=2000)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    evaluate(args.data_path, args.artifact_dir, args.shap_samples)


if __name__ == "__main__":
    main()

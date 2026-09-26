"""Interactive churn-model dashboard.

Run:
    streamlit run frontend/dashboard.py

When the FastAPI service at ``API_URL`` is reachable the dashboard is a pure API client
(``/predict`` + ``/explain``); otherwise it loads the same serialized pipeline in-process.
The threshold explorer re-scores nothing: it recomputes the confusion matrix and campaign
economics from the saved hold-out and out-of-fold probabilities on every slider move.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import COST_MODEL, TARGET, ArtifactPaths, CostModel  # noqa: E402
from src.metrics import confusion_counts, optimal_thresholds, threshold_curve  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PATHS = ArtifactPaths.from_env()

# Validated categorical slots; colour follows the entity (champion is always blue).
CHAMPION = "#2a78d6"
BASELINE = "#eb6834"
SERIES_3 = "#1baf7a"
RAISES_RISK = "#e34948"
LOWERS_RISK = "#2a78d6"
MUTED = "#898781"
CM_RAMP = ["#cde2fb", "#184f95"]

FEATURE_LABELS = {
    "age": "Age",
    "tenure_months": "Tenure (months)",
    "monthly_charges": "Monthly charges",
    "total_charges": "Lifetime billing",
    "num_support_tickets": "Support tickets (90d)",
    "avg_monthly_usage_gb": "Monthly usage (GB)",
    "days_since_last_login": "Days since last login",
    "num_addon_services": "Add-on services",
    "satisfaction_score": "Satisfaction (CSAT)",
    "late_payments_12m": "Late payments (12m)",
    "contract_type": "Contract",
    "payment_method": "Payment method",
    "internet_service": "Internet service",
    "plan_tier": "Plan tier",
    "region": "Region",
    "paperless_billing": "Paperless billing",
    "has_partner": "Has partner",
    "avg_monthly_spend": "Avg. historical bill",
    "charge_increase_pct": "Price increase vs. history",
    "is_early_tenure": "First 6 months",
}
BOOLEAN_DISPLAY = {"paperless_billing", "has_partner", "is_early_tenure"}
MONEY_DISPLAY = {"monthly_charges", "total_charges", "avg_monthly_spend"}

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")


# --------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_artifacts() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    metadata = json.loads(PATHS.metadata.read_text())
    metrics = json.loads(PATHS.metrics.read_text())
    return (
        metadata,
        metrics,
        pd.read_csv(PATHS.test_predictions),
        pd.read_csv(PATHS.oof_predictions),
    )


@st.cache_resource(show_spinner="Loading model...")
def local_service():
    from app.service import load_model_service

    return load_model_service(str(PATHS.root))


@st.cache_data(ttl=15, show_spinner=False)
def api_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=1.5)
        return response.json() if response.ok else None
    except requests.RequestException:
        return None


def score_customer(record: dict, use_api: bool) -> tuple[dict, dict, str]:
    """(prediction, explanation, source). Same response shapes from API and local model."""
    if use_api:
        try:
            prediction = requests.post(f"{API_URL}/predict", json=record, timeout=3)
            explanation = requests.post(f"{API_URL}/explain", json=record, timeout=3)
            if prediction.ok and explanation.ok:
                return prediction.json(), explanation.json(), "api"
        except requests.RequestException:
            pass
    service = local_service()
    features = {k: v for k, v in record.items() if k != "customer_id"}
    prediction = service.predict([features])[0]
    prediction["risk_tier"] = prediction["risk_tier"].value
    return prediction, service.explain(features), "local"


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------
def pretty(level: str) -> str:
    return level.replace("_", " ").replace("month to month", "month-to-month").capitalize()


def format_value(feature: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if feature in BOOLEAN_DISPLAY:
        return "yes" if float(value) >= 0.5 else "no"
    if feature == "charge_increase_pct":
        return f"{float(value):+.0%}"
    if feature in MONEY_DISPLAY:
        return f"${float(value):,.0f}"
    if isinstance(value, str):
        return pretty(value)
    return f"{float(value):,.0f}" if float(value).is_integer() else f"{float(value):,.1f}"


def ink() -> str:
    """Primary text colour for chart annotations in the active light/dark theme."""
    try:
        return "#ececea" if st.context.theme.type == "dark" else "#0b0b0b"
    except Exception:  # older Streamlit without theme introspection
        return MUTED


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def shap_waterfall(explanation: dict, top_k: int = 10) -> alt.Chart:
    contributions = explanation["contributions"]
    rows = [
        {
            "label": f"{FEATURE_LABELS.get(c['feature'], c['feature'])} = "
            f"{format_value(c['feature'], c['value'])}",
            "shap": c["shap_value"],
        }
        for c in contributions[:top_k]
    ]
    rest = contributions[top_k:]
    if rest:
        rows.append(
            {"label": f"{len(rest)} other features", "shap": sum(c["shap_value"] for c in rest)}
        )

    cumulative = explanation["base_value"]
    for order, row in enumerate(rows):
        row.update(order=order, start=cumulative, end=cumulative + row["shap"])
        row["direction"] = "Raises churn risk" if row["shap"] > 0 else "Lowers churn risk"
        row["shap_label"] = f"{row['shap']:+.2f}"
        row["p_after"] = f"{sigmoid(row['end']):.1%}"
        cumulative = row["end"]
    df = pd.DataFrame(rows)

    y = alt.Y("label:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelLimit=260))
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadius=3, height={"band": 0.7})
        .encode(
            y=y,
            x=alt.X("start:Q", title="Churn log-odds (SHAP, calibrated)"),
            x2="end:Q",
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Raises churn risk", "Lowers churn risk"],
                    range=[RAISES_RISK, LOWERS_RISK],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Feature"),
                alt.Tooltip("shap_label:N", title="SHAP (log-odds)"),
                alt.Tooltip("p_after:N", title="Probability after this step"),
            ],
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=4, fontSize=11, color=ink())
        .encode(y=y, x=alt.X("max_end:Q"), text="shap_label:N")
        .transform_calculate(max_end="max(datum.start, datum.end)")
    )
    base_rule = (
        alt.Chart(pd.DataFrame({"x": [explanation["base_value"]]}))
        .mark_rule(color=MUTED, strokeWidth=1)
        .encode(x="x:Q")
    )
    final_rule = (
        alt.Chart(pd.DataFrame({"x": [cumulative]}))
        .mark_rule(strokeWidth=1.5, strokeDash=[4, 3], color=ink())
        .encode(x="x:Q")
    )
    return (bars + labels + base_rule + final_rule).properties(height=36 * len(df) + 40)


def confusion_heatmap(counts: dict[str, int]) -> alt.Chart:
    cells = [
        ("Stays", "Stays", counts["tn"], "True negative"),
        ("Stays", "Churns", counts["fp"], "False positive"),
        ("Churns", "Stays", counts["fn"], "False negative"),
        ("Churns", "Churns", counts["tp"], "True positive"),
    ]
    df = pd.DataFrame(cells, columns=["actual", "predicted", "count", "cell"])
    df["row_rate"] = df["count"] / df.groupby("actual")["count"].transform("sum").clip(lower=1)
    df["text"] = df.apply(lambda r: f"{r['count']:,}\n{r['row_rate']:.0%}", axis=1)

    order = ["Stays", "Churns"]
    base = alt.Chart(df).encode(
        x=alt.X("predicted:N", sort=order, title="Predicted", scale=alt.Scale(paddingInner=0.04)),
        y=alt.Y("actual:N", sort=order, title="Actual", scale=alt.Scale(paddingInner=0.04)),
    )
    heat = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("row_rate:Q", scale=alt.Scale(domain=[0, 1], range=CM_RAMP), legend=None),
        tooltip=[
            alt.Tooltip("cell:N", title="Cell"),
            alt.Tooltip("count:Q", title="Customers", format=","),
            alt.Tooltip("row_rate:Q", title="Share of actual class", format=".1%"),
        ],
    )
    text = base.mark_text(fontSize=14, lineBreak="\n").encode(
        text="text:N",
        color=alt.condition("datum.row_rate > 0.55", alt.value("#ffffff"), alt.value("#0b0b0b")),
    )
    return (heat + text).properties(height=300)


def crosshair_lines(
    df: pd.DataFrame,
    y_title: str,
    color_domain: list[str],
    color_range: list[str],
    markers: dict[str, float],
    y_format: str = ".2f",
) -> alt.Chart:
    """Long-format (threshold, series, value) line chart with a nearest-threshold crosshair."""
    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["threshold"], empty=False, clear="pointerout"
    )
    x = alt.X("threshold:Q", title="Decision threshold", scale=alt.Scale(domain=[0, 1]))
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=color_domain, range=color_range),
        legend=alt.Legend(orient="top", title=None) if len(color_domain) > 1 else None,
    )
    lines = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(x=x, y=alt.Y("value:Q", title=y_title), color=color)
    )
    hover_targets = (
        alt.Chart(df).mark_rule(strokeWidth=12, opacity=0).encode(x=x).add_params(nearest)
    )
    wide = df.pivot(index="threshold", columns="series", values="value").reset_index()
    rule = (
        alt.Chart(wide)
        .mark_rule(color=MUTED)
        .encode(
            x=x,
            tooltip=[alt.Tooltip("threshold:Q", title="Threshold", format=".2f")]
            + [alt.Tooltip(f"{s}:Q", title=s, format=y_format) for s in color_domain],
        )
        .transform_filter(nearest)
    )
    points = lines.mark_point(filled=True, size=60).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    marker_df = pd.DataFrame({"threshold": list(markers.values()), "label": list(markers)})
    marker_rules = (
        alt.Chart(marker_df)
        .mark_rule(strokeDash=[4, 3], strokeWidth=1, color=ink())
        .encode(x="threshold:Q", tooltip=["label:N", alt.Tooltip("threshold:Q", format=".2f")])
    )
    marker_text = (
        alt.Chart(marker_df)
        .mark_text(align="left", dx=4, dy=-6, fontSize=11, baseline="bottom", color=ink())
        .encode(x="threshold:Q", y=alt.value(12), text="label:N")
    )
    return alt.layer(lines, hover_targets, rule, points, marker_rules, marker_text).properties(
        height=300
    )


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------
if not PATHS.metadata.exists():
    st.error(
        f"No trained model found in `{PATHS.root}`. Run `python data/generate_dataset.py` "
        "and `python src/train.py` first."
    )
    st.stop()

metadata, metrics, test_df, oof_df = load_artifacts()
levels = metadata["features"]["category_levels"]
defaults = metadata["features"]["defaults"]
model_threshold = float(metadata["decision_threshold"])

if "threshold" not in st.session_state:
    st.session_state.threshold = model_threshold

# ---- Sidebar -------------------------------------------------------------------------
with st.sidebar:
    st.header("Churn Risk Console")
    health = api_health()
    use_api = bool(health and health.get("model_loaded"))
    if use_api:
        st.success(f"Connected to inference API\n\n`{health['model_version']}`")
    else:
        st.info(f"API unreachable at `{API_URL}` - scoring with the local model artifact.")

    st.subheader("Campaign economics")
    offer_cost = st.number_input(
        "Retention offer cost ($)", 1.0, 1000.0, COST_MODEL.offer_cost, 5.0
    )
    clv = st.number_input(
        "Customer lifetime value ($)", 10.0, 20000.0, COST_MODEL.customer_lifetime_value, 50.0
    )
    save_rate = st.slider(
        "Offer success rate", 0.05, 1.0, COST_MODEL.retention_success_rate, 0.05, format="%.2f"
    )
    costs = CostModel(offer_cost, clv, save_rate)

    oof_curve = threshold_curve(oof_df[TARGET], oof_df["champion_proba"], costs)
    recommended = optimal_thresholds(oof_curve)["business"]
    st.caption(
        f"Break-even probability: **{costs.bayes_optimal_threshold:.2f}** · "
        f"payoff-optimal threshold (tuned on training out-of-fold predictions): "
        f"**{recommended:.2f}**"
    )

    def _apply_recommended() -> None:
        st.session_state.threshold = recommended

    st.button("Use recommended threshold", on_click=_apply_recommended, width="stretch")
    st.slider("Decision threshold", 0.01, 0.99, key="threshold", step=0.01)
    threshold = float(st.session_state.threshold)
    st.caption(f"Deployed model threshold: {model_threshold:.2f}")

sandbox_tab, threshold_tab, report_tab = st.tabs(
    ["Prediction sandbox", "Threshold explorer", "Model report"]
)

# ---- Prediction sandbox --------------------------------------------------------------
with sandbox_tab:
    st.subheader("Score a customer")
    st.caption("Adjust any input - the prediction and its SHAP explanation update live.")
    account, engagement, risk = st.columns(3)

    def pick(label: str, feature: str, column) -> str:
        options = levels[feature]
        return column.selectbox(
            label, options, index=options.index(defaults[feature]), format_func=pretty
        )

    with account:
        st.markdown("**Account**")
        contract = pick("Contract", "contract_type", account)
        tenure = account.slider("Tenure (months)", 0, 72, int(defaults["tenure_months"]))
        plan = pick("Plan tier", "plan_tier", account)
        internet = pick("Internet service", "internet_service", account)
        payment = pick("Payment method", "payment_method", account)
        monthly = account.slider(
            "Monthly charges ($)", 15.0, 200.0, float(round(defaults["monthly_charges"])), 0.5
        )
        hike = account.slider("Recent price increase vs. historical average (%)", 0, 40, 0)
        total = round(tenure * monthly / (1 + hike / 100), 2)
        account.caption(f"Implied lifetime billing: ${total:,.2f}")

    with engagement:
        st.markdown("**Engagement**")
        days = engagement.slider(
            "Days since last login", 0, 120, int(defaults["days_since_last_login"])
        )
        usage = engagement.slider(
            "Avg. monthly usage (GB)",
            0.0,
            600.0,
            float(round(defaults["avg_monthly_usage_gb"])),
            5.0,
        )
        addons = engagement.slider("Add-on services", 0, 6, int(defaults["num_addon_services"]))
        csat = engagement.select_slider(
            "Satisfaction (CSAT)",
            options=["Not answered", 1, 2, 3, 4, 5],
            value=int(defaults["satisfaction_score"]),
        )
        paperless = engagement.toggle(
            "Paperless billing", value=bool(defaults["paperless_billing"])
        )

    with risk:
        st.markdown("**Risk signals & profile**")
        tickets = risk.slider(
            "Support tickets (last 90 days)", 0, 10, int(defaults["num_support_tickets"])
        )
        late = risk.slider("Late payments (12 months)", 0, 12, int(defaults["late_payments_12m"]))
        age = risk.slider("Age", 18, 85, int(defaults["age"]))
        region = pick("Region", "region", risk)
        partner = risk.toggle("Has partner", value=bool(defaults["has_partner"]))

    record = {
        "customer_id": "SANDBOX",
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": float(monthly),
        "total_charges": float(total),
        "num_support_tickets": tickets,
        "avg_monthly_usage_gb": float(usage),
        "days_since_last_login": days,
        "num_addon_services": addons,
        "satisfaction_score": None if csat == "Not answered" else int(csat),
        "late_payments_12m": late,
        "contract_type": contract,
        "payment_method": payment,
        "internet_service": internet,
        "plan_tier": plan,
        "region": region,
        "paperless_billing": paperless,
        "has_partner": partner,
    }
    prediction, explanation, source = score_customer(record, use_api)
    p = prediction["churn_probability"]
    base_p = explanation["base_probability"]
    act = p >= threshold
    expected_value = p * costs.true_positive_value - (1 - p) * costs.false_positive_cost

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Churn risk",
        f"{p:.1%}",
        delta=f"{(p - base_p) * 100:+.1f} pts vs avg",
        delta_color="inverse",
        help=f"Average customer: {base_p:.1%}",
    )
    m2.metric(
        "Decision",
        "Offer" if act else "No action",
        help=f"Send a retention offer when churn risk >= {threshold:.2f}",
    )
    sign = "-" if expected_value < 0 else ""
    m3.metric(
        "Offer EV",
        f"{sign}${abs(expected_value):,.0f}",
        help="Expected net value of sending this customer an offer under the sidebar economics",
    )
    m4.metric(
        "Risk tier",
        prediction["risk_tier"].title(),
        help=f"Tier at the deployed operating threshold ({model_threshold:.2f})",
    )

    st.markdown("**Why this score? Local SHAP waterfall**")
    st.altair_chart(shap_waterfall(explanation), width="stretch")
    st.caption(
        f"Starts at the average customer ({base_p:.1%}, grey line) and adds each feature's "
        f"exact TreeSHAP contribution to reach this customer ({p:.1%}, dashed line). "
        f"Scored via {'the FastAPI service' if source == 'api' else 'the local model'}."
    )
    st.markdown("**Top drivers**")
    for c in explanation["contributions"][:5]:
        arrow = "raises" if c["shap_value"] > 0 else "lowers"
        name = FEATURE_LABELS.get(c["feature"], c["feature"])
        st.markdown(
            f"- **{name}** = {format_value(c['feature'], c['value'])} {arrow} risk "
            f"by {abs(c['shap_value']):.2f} log-odds"
        )

# ---- Threshold explorer --------------------------------------------------------------
with threshold_tab:
    st.subheader("How the decision threshold trades off errors and money")
    st.caption(
        f"Hold-out set: {len(test_df):,} customers never used for training, tuning, or "
        "threshold selection. Every number below recomputes as you move the sidebar controls."
    )
    y_test, p_test = test_df[TARGET].to_numpy(), test_df["champion_proba"].to_numpy()
    counts = confusion_counts(y_test, p_test, threshold)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    payoff = tp * costs.true_positive_value - fp * costs.false_positive_cost
    test_curve = threshold_curve(y_test, p_test, costs)
    best_payoff = float(test_curve["payoff"].max())

    k1, k2, k3 = st.columns(3)
    k1.metric("Precision", f"{precision:.1%}")
    k2.metric("Recall", f"{recall:.1%}")
    k3.metric("F1", f"{f1:.3f}")
    k4, k5 = st.columns(2)
    k4.metric("Customers targeted", f"{(tp + fp) / len(y_test):.1%}")
    k5.metric(
        "Net campaign value",
        f"${payoff:,.0f}",
        delta=f"-${best_payoff - payoff:,.0f} vs. best" if payoff < best_payoff else "optimal",
        delta_color="off" if payoff >= best_payoff else "normal",
    )

    cm_col, pay_col = st.columns(2)
    with cm_col:
        st.markdown(f"**Confusion matrix at threshold {threshold:.2f}**")
        st.caption("Cell shading and % = share of the actual class (row).")
        st.altair_chart(confusion_heatmap(counts), width="stretch")
    with pay_col:
        st.markdown("**Net campaign value per 1,000 customers**")
        pay_df = pd.DataFrame(
            {
                "threshold": test_curve["threshold"],
                "series": "Hold-out",
                "value": test_curve["payoff_per_1k_customers"],
            }
        )
        st.altair_chart(
            crosshair_lines(
                pay_df,
                "Net value per 1k customers ($)",
                ["Hold-out"],
                [CHAMPION],
                (
                    {"current = recommended": threshold}
                    if abs(threshold - recommended) < 0.03
                    else {"current": threshold, "recommended": recommended}
                ),
                y_format=",.0f",
            ),
            width="stretch",
        )

    st.markdown("**Precision, recall and F1 across thresholds (hold-out)**")
    prf = test_curve.melt(
        id_vars="threshold", value_vars=["precision", "recall", "f1"], var_name="series"
    )
    prf["series"] = prf["series"].map({"precision": "Precision", "recall": "Recall", "f1": "F1"})
    st.altair_chart(
        crosshair_lines(
            prf,
            "Score",
            ["Precision", "Recall", "F1"],
            [CHAMPION, BASELINE, SERIES_3],
            {"current": threshold},
        ),
        width="stretch",
    )

# ---- Model report --------------------------------------------------------------------
with report_tab:
    champion = metrics["champion"]
    names = {"xgboost": "XGBoost", "logistic_regression": "Logistic Regression"}
    cv = metrics["cross_validation"]
    holdout = metrics["holdout"]
    ci = metrics["holdout_bootstrap_ci"]

    st.subheader(f"Champion: {names[champion]} (calibrated)")
    h1, h2, h3, h4 = st.columns(4)
    cv_auc = cv[champion]["metrics"]["roc_auc"]
    h1.metric("5-fold CV ROC-AUC", f"{cv_auc['mean']:.3f} ± {cv_auc['std']:.3f}")
    auc_ci = ci["roc_auc"][champion]
    h2.metric(
        "Hold-out ROC-AUC",
        f"{auc_ci['estimate']:.3f}",
        help=f"95% bootstrap CI [{auc_ci['ci_low']:.3f}, {auc_ci['ci_high']:.3f}]",
    )
    h3.metric("Hold-out PR-AUC", f"{holdout[champion]['pr_auc']:.3f}")
    h4.metric("Hold-out Brier score", f"{holdout[champion]['brier']:.4f}")

    diff_key = next((k for k in ci["roc_auc"] if "_minus_" in k), None)
    if diff_key:
        d = ci["roc_auc"][diff_key]
        st.caption(
            f"Paired bootstrap, ROC-AUC {names[champion]} minus baseline: {d['estimate']:+.4f} "
            f"(95% CI {d['ci_low']:+.4f} to {d['ci_high']:+.4f})."
        )

    table_cv, table_ho = st.columns(2)
    with table_cv:
        st.markdown("**Stratified 5-fold cross-validation (training split)**")
        rows = []
        for metric in ["roc_auc", "pr_auc", "f1", "brier", "log_loss", "ece"]:
            row = {"Metric": metric.replace("_", " ").upper().replace("LOG LOSS", "Log-loss")}
            for model, label in names.items():
                m = cv[model]["metrics"][metric]
                row[label] = f"{m['mean']:.4f} ± {m['std']:.4f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with table_ho:
        st.markdown("**Hold-out set (each model at its own tuned threshold)**")
        rows = []
        for metric in ["roc_auc", "pr_auc", "f1", "precision", "recall", "brier", "threshold"]:
            row = {"Metric": metric.replace("_", " ").upper()}
            for model, label in names.items():
                row[label] = f"{holdout[model][metric]:.4f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    biz = metrics["business_impact_holdout"]
    st.markdown(f"**Campaign value on the {biz['n_test_customers']:,}-customer hold-out**")
    st.dataframe(
        pd.DataFrame(
            [
                ("Tuned threshold (deployed)", biz["payoff_selected_threshold"]),
                ("F1-optimal threshold", biz["payoff_f1_threshold"]),
                ("Default 0.5 threshold", biz["payoff_default_0_5_threshold"]),
                ("Contact every customer", biz["payoff_contact_everyone"]),
                ("No campaign", biz["payoff_no_campaign"]),
            ],
            columns=["Policy", "Net value ($)"],
        ).style.format({"Net value ($)": "${:,.0f}"}),
        hide_index=True,
        width="stretch",
    )

    figures = [
        ("roc_curve.png", "ROC curve"),
        ("pr_curve.png", "Precision-recall curve"),
        ("calibration_curve.png", "Calibration"),
        ("confusion_matrix.png", "Confusion matrix"),
        ("shap_summary.png", "Global SHAP summary"),
        ("threshold_analysis.png", "Threshold economics"),
    ]
    available = [(PATHS.figures / f, title) for f, title in figures if (PATHS.figures / f).exists()]
    if available:
        st.markdown("**Evaluation figures** (generated by `src/evaluate.py`)")
        for i in range(0, len(available), 2):
            cols = st.columns(2)
            for col, (path, title) in zip(cols, available[i : i + 2]):
                col.image(str(path), caption=title, width="stretch")
    else:
        st.info("Run `python src/evaluate.py` to generate evaluation figures.")

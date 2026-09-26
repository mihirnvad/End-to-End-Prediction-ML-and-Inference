"""Generate a realistic synthetic customer-subscription churn dataset.

The generator simulates a telecom/SaaS-style subscriber base with:

* heterogeneous tenure distributions per contract type,
* latent drivers (engagement, service quality, financial stress, competitor pressure)
  that are only partially observable through the recorded features,
* non-linear effects and interactions (early-tenure hazard, ticket thresholds,
  month-to-month x new-customer interaction, hidden price increases), so a gradient-boosted
  model has real structure to find beyond a linear baseline,
* realistic data-quality issues: missing-not-at-random survey scores, telemetry gaps,
  billing-sync gaps, and inconsistently-cased category strings.

Usage:
    python data/generate_dataset.py --n-samples 25000 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    CATEGORY_LEVELS,
    DEFAULT_DATA_PATH,
    ID_COLUMN,
    RANDOM_STATE,
    RAW_FEATURES,
    TARGET,
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _mangle_case(values: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    """Simulate upstream systems that emit inconsistent casing / whitespace."""
    out = values.astype(object).copy()
    idx = np.flatnonzero(rng.random(len(out)) < rate)
    styles = [str.upper, str.title, lambda s: f" {s}", lambda s: f"{s} "]
    for i in idx:
        out[i] = styles[rng.integers(len(styles))](out[i])
    return out


def generate_customers(n_samples: int = 25_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_samples

    # --- Account structure -----------------------------------------------------------
    contract = rng.choice(CATEGORY_LEVELS["contract_type"], p=[0.55, 0.24, 0.21], size=n)
    is_m2m = contract == "month_to_month"
    is_2y = contract == "two_year"
    shape = np.select([is_m2m, contract == "one_year"], [1.1, 2.2], 3.5)
    scale = np.select([is_m2m, contract == "one_year"], [14.5, 14.5], 13.7)
    tenure = np.clip(np.round(rng.gamma(shape, scale)), 0, 72).astype(int)

    age = np.clip(np.round(rng.normal(44, 14, n)), 18, 85).astype(int)
    internet = rng.choice(CATEGORY_LEVELS["internet_service"], p=[0.44, 0.36, 0.20], size=n)
    is_fiber = internet == "fiber_optic"
    has_internet = internet != "no_internet"
    plan_tier = rng.choice(CATEGORY_LEVELS["plan_tier"], p=[0.42, 0.38, 0.20], size=n)
    region = rng.choice(CATEGORY_LEVELS["region"], p=[0.20, 0.22, 0.20, 0.16, 0.22], size=n)
    has_partner = rng.random(n) < 0.48
    paperless = rng.random(n) < np.where(has_internet, 0.66, 0.40)

    # --- Latent drivers (never observed directly) ------------------------------------
    engagement = rng.normal(0, 1, n)
    service_issues = rng.normal(0, 1, n)
    financial_stress = rng.normal(0, 1, n)
    competitor_pressure = rng.normal(0, 1, n)

    # --- Pricing: current bill vs. historical average reveals hidden price increases ---
    addons = np.clip(rng.poisson(np.where(has_internet, 1.6, 0.4)), 0, 6)
    base_price = (
        np.select([is_fiber, internet == "dsl"], [72.0, 48.0], 22.0)
        + np.select([plan_tier == "standard", plan_tier == "premium"], [12.0, 28.0], 0.0)
        + 6.0 * addons
        + rng.normal(0, 4, n)
    )
    base_price = np.clip(base_price, 15, None)
    price_increase = (rng.random(n) < 0.20) * rng.uniform(0.08, 0.35, n)
    monthly_charges = np.round(base_price * (1 + price_increase), 2)
    historical_avg = base_price * rng.uniform(0.97, 1.03, n)
    total_charges = np.round(tenure * historical_avg, 2)

    # --- Behaviour -------------------------------------------------------------------
    expected_log_usage = np.select(
        [is_fiber, internet == "dsl"], [np.log(180), np.log(90)], np.log(6)
    )
    relative_usage = 0.55 * engagement + rng.normal(0, 0.35, n)  # vs. plan expectation
    log_usage = expected_log_usage + relative_usage
    usage = np.round(np.exp(log_usage), 1)
    days_since_login = np.clip(
        np.round(rng.exponential(7 * np.exp(-0.9 * engagement))), 0, 365
    ).astype(int)

    # Month-to-month customers skew heavily towards manual electronic-check payments.
    pay_probs = np.where(is_m2m[:, None], [0.45, 0.20, 0.17, 0.18], [0.18, 0.20, 0.31, 0.31])
    pay_idx = (rng.random(n)[:, None] > pay_probs.cumsum(axis=1)).sum(axis=1).clip(0, 3)
    payment = np.array(CATEGORY_LEVELS["payment_method"])[pay_idx]
    is_ec = payment == "electronic_check"

    late_payments = np.clip(rng.poisson(np.exp(-1.2 + 0.7 * financial_stress + 0.5 * is_ec)), 0, 12)
    tickets = np.clip(rng.poisson(np.exp(-0.75 + 0.35 * is_fiber + 0.45 * service_issues)), 0, 50)
    satisfaction_latent = (
        3.5
        - 0.30 * tickets
        - 3.0 * price_increase
        - 0.45 * service_issues
        + 0.25 * engagement
        + rng.normal(0, 0.6, n)
    )
    satisfaction = np.clip(np.round(satisfaction_latent), 1, 5)

    # --- Churn mechanism -------------------------------------------------------------
    # Contract lock-in dampens how strongly dissatisfaction and price hikes translate
    # into churn; new customers react more to support friction; usage is U-shaped relative
    # to what the customer's plan implies (disengaged users and capped power users leave).
    logit = (
        -3.70
        + 1.20 * is_m2m
        - 0.60 * is_2y
        + 1.40 * np.exp(-tenure / 6)
        - 0.015 * tenure
        + 0.80 * (is_m2m & (tenure < 12))
        + 0.35 * is_fiber
        + 0.030 * np.clip(monthly_charges - 80, 0, None) * is_fiber
        + 5.0 * price_increase * (1 + is_m2m)
        + 1.10 * (tickets >= 3)
        + 0.10 * tickets
        + 0.50 * ((tickets >= 2) & (tenure < 12))
        - np.where(is_m2m, 1.10, 0.25) * (satisfaction_latent - 3)
        + 1.60 * (days_since_login > 30)
        + 0.60 * (1 - np.exp(-days_since_login / 14))
        + 0.95 * (relative_usage < -0.6)
        + 0.80 * (relative_usage > 0.9)
        + 0.40 * is_ec
        + 0.12 * late_payments
        + 0.35 * late_payments * is_ec
        - 0.20 * addons
        + 0.75 * (age < 28)
        + 0.65 * (age >= 70)
        - 0.20 * has_partner
        + 0.10 * paperless
        + 0.35 * competitor_pressure
    )
    churned = (rng.random(n) < _sigmoid(logit)).astype(int)

    df = pd.DataFrame(
        {
            ID_COLUMN: [f"CUST-{i:06d}" for i in range(1, n + 1)],
            "age": age,
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": tickets,
            "avg_monthly_usage_gb": usage,
            "days_since_last_login": days_since_login,
            "num_addon_services": addons,
            "satisfaction_score": satisfaction,
            "late_payments_12m": late_payments,
            "contract_type": _mangle_case(contract, rng, 0.003),
            "payment_method": _mangle_case(payment, rng, 0.006),
            "internet_service": internet,
            "plan_tier": plan_tier,
            "region": region,
            "paperless_billing": paperless,
            "has_partner": has_partner,
            TARGET: churned,
        }
    )

    # --- Data-quality issues ---------------------------------------------------------
    # Disengaged and brand-new customers skip satisfaction surveys (missing-not-at-random).
    p_skip_survey = _sigmoid(-2.0 - 0.8 * engagement + 0.6 * (tenure < 3))
    df.loc[rng.random(n) < p_skip_survey, "satisfaction_score"] = np.nan
    df.loc[rng.random(n) < 0.04, "avg_monthly_usage_gb"] = np.nan  # telemetry gaps
    df.loc[rng.random(n) < 0.012, "total_charges"] = np.nan  # billing-sync gaps
    return df[[ID_COLUMN, *RAW_FEATURES, TARGET]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-samples", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    df = generate_customers(args.n_samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    missing = df[RAW_FEATURES].isna().mean()
    print(f"Wrote {len(df):,} customers -> {args.output}")
    print(f"Churn rate: {df[TARGET].mean():.1%}")
    print("Missing values: " + ", ".join(f"{c}={v:.1%}" for c, v in missing[missing > 0].items()))


if __name__ == "__main__":
    main()

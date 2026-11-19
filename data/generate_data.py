"""
Generate a realistic synthetic telecom customer dataset for churn analysis.

The data mimics the structure of real-world telco churn datasets (contracts,
services, billing, support tickets) and embeds realistic churn drivers so the
analysis and models have genuine signal to discover. It also injects a few
data-quality issues (blank values, duplicates, inconsistent casing) so the
cleaning step is meaningful.

Usage:
    python data/generate_data.py --rows 7000 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).resolve().parent / "raw" / "telco_customers.csv"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int = 7000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n_rows)
    senior = rng.choice([0, 1], n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_rows, p=[0.48, 0.52])
    dependents = np.where(
        partner == "Yes",
        rng.choice(["Yes", "No"], n_rows, p=[0.55, 0.45]),
        rng.choice(["Yes", "No"], n_rows, p=[0.10, 0.90]),
    )

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.21, 0.24]
    )
    # Longer contracts -> longer tenure
    tenure_scale = np.select(
        [contract == "Month-to-month", contract == "One year"], [14, 35], default=52
    )
    tenure = np.clip(rng.exponential(tenure_scale), 0, 72).round().astype(int)

    internet = rng.choice(["DSL", "Fiber optic", "No"], n_rows, p=[0.34, 0.44, 0.22])
    has_net = internet != "No"

    def addon(p_yes: float) -> np.ndarray:
        vals = rng.choice(["Yes", "No"], n_rows, p=[p_yes, 1 - p_yes])
        return np.where(has_net, vals, "No internet service")

    online_security = addon(0.37)
    online_backup = addon(0.44)
    tech_support = addon(0.37)
    streaming_tv = addon(0.49)
    streaming_movies = addon(0.50)

    phone_service = rng.choice(["Yes", "No"], n_rows, p=[0.90, 0.10])
    paperless = rng.choice(["Yes", "No"], n_rows, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_rows,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    base_price = np.select([internet == "Fiber optic", internet == "DSL"], [70, 45], default=20)
    monthly = (
        base_price
        + (phone_service == "Yes") * 5
        + (online_security == "Yes") * 5
        + (online_backup == "Yes") * 5
        + (tech_support == "Yes") * 5
        + (streaming_tv == "Yes") * 10
        + (streaming_movies == "Yes") * 10
        + rng.normal(0, 3, n_rows)
    ).round(2)
    total = (monthly * tenure * rng.uniform(0.97, 1.03, n_rows)).round(2)

    tickets = rng.poisson(
        np.where(internet == "Fiber optic", 1.6, 0.9) + (tech_support == "No") * 0.6
    )

    # ---- Churn mechanism (log-odds) ----
    logit = (
        -2.3
        + 1.3 * (contract == "Month-to-month")
        - 1.4 * (contract == "Two year")
        + 0.6 * (internet == "Fiber optic")
        + 0.5 * (payment == "Electronic check")
        - 0.035 * tenure
        + 0.35 * (online_security == "No")
        + 0.35 * (tech_support == "No")
        + 0.28 * tickets
        + 0.30 * senior
        + 0.012 * (monthly - 65)
        + 0.20 * (paperless == "Yes")
        - 0.20 * (partner == "Yes")
        + rng.normal(0, 0.35, n_rows)
    )
    churn = np.where(rng.uniform(size=n_rows) < _sigmoid(logit), "Yes", "No")

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n_rows + 1)],
            "gender": gender,
            "senior_citizen": senior,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure,
            "phone_service": phone_service,
            "internet_service": internet,
            "online_security": online_security,
            "online_backup": online_backup,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless,
            "payment_method": payment,
            "monthly_charges": monthly,
            "total_charges": total.astype(str),
            "num_support_tickets": tickets,
            "churn": churn,
        }
    )

    # ---- Inject realistic data-quality issues ----
    # 1) New customers (tenure 0) have blank total charges
    df.loc[df["tenure_months"] == 0, "total_charges"] = " "
    # 2) Inconsistent casing / whitespace in a categorical column
    idx = rng.choice(df.index, size=n_rows // 50, replace=False)
    df.loc[idx, "payment_method"] = df.loc[idx, "payment_method"].str.lower() + " "
    # 3) A handful of exact duplicate rows
    dupes = df.sample(n=max(1, n_rows // 200), random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out}  (churn rate: {(df['churn'] == 'Yes').mean():.1%})")


if __name__ == "__main__":
    main()

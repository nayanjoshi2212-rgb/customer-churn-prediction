"""Batch scoring: add churn probability and risk segment to a CSV of customers.

Usage:
    python -m src.predict --input data/raw/telco_customers.csv --output reports/scored_customers.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import ID_COL, MODEL_PATH, TARGET
from src.data_prep import clean


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python main.py` first.")
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle.get("threshold", 0.5)


def risk_segment(p: float) -> str:
    if p >= 0.7:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


def score(df: pd.DataFrame) -> pd.DataFrame:
    pipe, threshold = load_model()
    data = clean(df)
    X = data.drop(columns=[TARGET], errors="ignore")
    proba = pipe.predict_proba(X)[:, 1]
    out = pd.DataFrame(
        {
            ID_COL: data.get(ID_COL, pd.Series(range(len(data)))),
            "churn_probability": proba.round(4),
            "risk_segment": [risk_segment(p) for p in proba],
            "target_for_retention": proba >= threshold,
        }
    )
    return out.sort_values("churn_probability", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/scored_customers.csv"))
    args = parser.parse_args()
    scored = score(pd.read_csv(args.input))
    scored.to_csv(args.output, index=False)
    print(scored["risk_segment"].value_counts().to_string())
    print(f"Saved {len(scored):,} scored customers -> {args.output}")


if __name__ == "__main__":
    main()

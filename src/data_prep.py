"""Data loading, cleaning and feature engineering."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, ID_COL, TARGET

SERVICE_COLS = [
    "phone_service",
    "online_security",
    "online_backup",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]


def load_raw(path: Path = DATA_RAW) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_data.py` first."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, missing values, duplicates and inconsistent categories."""
    df = df.copy()
    report = {"rows_in": len(df)}

    # Normalise text columns: trim whitespace, consistent casing for payment method
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    if "payment_method" in df:
        df["payment_method"] = df["payment_method"].str.capitalize()

    # total_charges arrives as text with blanks for brand-new customers
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    report["missing_total_charges"] = int(df["total_charges"].isna().sum())
    df["total_charges"] = df["total_charges"].fillna(
        df["monthly_charges"] * df["tenure_months"]
    )

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    if TARGET in df and df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)

    report["rows_out"] = len(df)
    df.attrs["cleaning_report"] = report
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven features. Works on raw-cleaned rows (used inside the model pipeline)."""
    df = df.copy()
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)
    df["avg_monthly_spend"] = np.where(
        df["tenure_months"] > 0,
        df["total_charges"] / df["tenure_months"].clip(lower=1),
        df["monthly_charges"],
    )
    df["is_new_customer"] = (df["tenure_months"] <= 6).astype(int)
    df["tickets_per_year"] = df["num_support_tickets"] / (df["tenure_months"].clip(lower=1) / 12)
    df["tenure_group"] = pd.cut(
        df["tenure_months"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    ).astype(str)
    df["has_family"] = ((df["partner"] == "Yes") | (df["dependents"] == "Yes")).astype(int)
    return df.drop(columns=[ID_COL], errors="ignore")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET]


def run() -> pd.DataFrame:
    df = clean(load_raw())
    df.to_csv(DATA_PROCESSED, index=False)
    print("Cleaning report:", df.attrs["cleaning_report"])
    return df


if __name__ == "__main__":
    run()

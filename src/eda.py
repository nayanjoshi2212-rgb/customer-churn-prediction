"""Exploratory Data Analysis: saves charts to reports/figures and key insights to reports/eda_insights.md."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR, TARGET
from src.data_prep import add_features, clean, load_raw

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#2a6fdb", "#e8743b"


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby(col)[TARGET].mean().sort_values(ascending=False)


def plot_rate_bar(df: pd.DataFrame, col: str, fname: str, title: str) -> pd.Series:
    rates = churn_rate_by(df, col)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(rates.index.astype(str), rates.values * 100, color=BLUE)
    ax.axhline(df[TARGET].mean() * 100, ls="--", color="grey", lw=1, label="Overall churn")
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=15)
    _save(fig, fname)
    return rates


def plot_distribution(df: pd.DataFrame, col: str, fname: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(df.loc[df[TARGET] == 0, col], bins=30, alpha=0.65, color=BLUE, label="Stayed", density=True)
    ax.hist(df.loc[df[TARGET] == 1, col], bins=30, alpha=0.65, color=ORANGE, label="Churned", density=True)
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save(fig, fname)


def plot_correlation(df: pd.DataFrame, fname: str) -> None:
    num = df.select_dtypes("number")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix (numeric features)")
    _save(fig, fname)


def run() -> None:
    raw = load_raw()
    df = clean(raw)
    cr = df.attrs["cleaning_report"]
    feat = add_features(df)

    overall = df[TARGET].mean()

    # Target balance
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar(["Stayed", "Churned"], counts.values, color=[BLUE, ORANGE])
    ax.bar_label(bars, labels=[f"{v:,}\n({v / len(df):.0%})" for v in counts.values], padding=2)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_title("Target distribution")
    _save(fig, "01_target_distribution.png")

    contract = plot_rate_bar(feat, "contract", "02_churn_by_contract.png", "Churn rate by contract type")
    internet = plot_rate_bar(feat, "internet_service", "03_churn_by_internet.png", "Churn rate by internet service")
    payment = plot_rate_bar(feat, "payment_method", "04_churn_by_payment.png", "Churn rate by payment method")
    tenure = plot_rate_bar(feat, "tenure_group", "05_churn_by_tenure_group.png", "Churn rate by tenure group")
    support = plot_rate_bar(feat, "tech_support", "06_churn_by_tech_support.png", "Churn rate by tech support")
    plot_distribution(feat, "tenure_months", "07_tenure_distribution.png", "Tenure: churned vs stayed")
    plot_distribution(feat, "monthly_charges", "08_monthly_charges_distribution.png", "Monthly charges: churned vs stayed")
    plot_correlation(feat, "09_correlation_matrix.png")

    tickets = feat.groupby(TARGET)["num_support_tickets"].mean()
    revenue_at_risk = df.loc[df[TARGET] == 1, "monthly_charges"].sum()

    md = f"""# EDA Insights

## Data quality
| Check | Result |
|---|---|
| Rows loaded | {cr['rows_in']:,} |
| Blank `total_charges` fixed (new customers) | {cr['missing_total_charges']} |
| Duplicate rows removed | {cr['duplicates_removed']} |
| Inconsistent `payment_method` labels | normalised (trimmed + recased) |
| Final rows | {cr['rows_out']:,} |

## Key findings
- **Overall churn rate: {overall:.1%}** - the classes are imbalanced, so accuracy alone is misleading; ROC-AUC, recall and F1 are used instead.
- **Contract type is the strongest driver:** month-to-month customers churn at **{contract.get('Month-to-month', 0):.1%}** vs **{contract.get('Two year', 0):.1%}** on two-year contracts.
- **Early tenure is the riskiest period:** customers in their first 12 months churn at **{tenure.get('0-12m', 0):.1%}**, compared with **{tenure.get('49-72m', 0):.1%}** after 4+ years.
- **Fiber optic users churn more** (**{internet.get('Fiber optic', 0):.1%}**) than DSL (**{internet.get('DSL', 0):.1%}**) - possible price/quality issue worth investigating.
- **Electronic check payers churn at {payment.get('Electronic check', 0):.1%}**, the highest of any payment method; auto-pay customers are stickier.
- **No tech support -> {support.get('No', 0):.1%} churn** vs {support.get('Yes', 0):.1%} with tech support.
- Churned customers raised **{tickets.get(1, 0):.2f}** support tickets on average vs **{tickets.get(0, 0):.2f}** for retained customers.
- **Monthly revenue lost to churn: ${revenue_at_risk:,.0f}**.

## Business recommendations
1. Offer discounts to move month-to-month customers onto 1-2 year contracts.
2. Build a structured onboarding programme for the first 12 months.
3. Bundle free tech support / online security for high-risk fiber customers.
4. Incentivise switching from electronic check to auto-pay.
5. Flag customers with rising support tickets for proactive outreach.

![Churn by contract](figures/02_churn_by_contract.png)
![Churn by tenure group](figures/05_churn_by_tenure_group.png)
"""
    (REPORTS_DIR / "eda_insights.md").write_text(md, encoding="utf-8")
    print(f"EDA complete -> {REPORTS_DIR / 'eda_insights.md'}")


if __name__ == "__main__":
    run()

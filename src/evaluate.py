"""Model evaluation: ROC curves, confusion matrix, feature importance and business-driven threshold tuning."""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    AVG_CUSTOMER_VALUE,
    FIGURES_DIR,
    MODEL_PATH,
    OFFER_SUCCESS_RATE,
    RANDOM_STATE,
    REPORTS_DIR,
    RETENTION_OFFER_COST,
)

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def business_profit(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    """Net savings of targeting customers above `threshold` with a retention offer."""
    target = proba >= threshold
    tp = np.sum(target & (y_true == 1))
    n_target = target.sum()
    return tp * OFFER_SUCCESS_RATE * AVG_CUSTOMER_VALUE - n_target * RETENTION_OFFER_COST


def run() -> dict:
    bundle = joblib.load(MODEL_PATH)
    pipe, X_test, y_test = bundle["pipeline"], bundle["X_test"], bundle["y_test"].to_numpy()
    proba = pipe.predict_proba(X_test)[:, 1]

    # ROC curves for every model
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, m in bundle["all_models"].items():
        RocCurveDisplay.from_estimator(m, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC curves - test set")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_roc_curves.png")
    plt.close(fig)

    # Threshold optimisation (business profit)
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": [precision_score(y_test, proba >= t, zero_division=0) for t in thresholds],
            "recall": [recall_score(y_test, proba >= t) for t in thresholds],
            "f1": [f1_score(y_test, proba >= t) for t in thresholds],
            "profit": [business_profit(y_test, proba, t) for t in thresholds],
        }
    )
    best = curve.loc[curve["profit"].idxmax()]
    best_t = float(best["threshold"])

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    for col, c in [("precision", "#2a6fdb"), ("recall", "#e8743b"), ("f1", "#3a9d5d")]:
        ax1.plot(curve["threshold"], curve[col], label=col.title(), color=c)
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Score")
    ax2 = ax1.twinx()
    ax2.plot(curve["threshold"], curve["profit"] / 1000, "k--", label="Net savings ($k)")
    ax2.set_ylabel("Net savings on test set ($k)")
    ax1.axvline(best_t, color="grey", ls=":", lw=1)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax1.set_title(f"Threshold vs metrics - best business threshold = {best_t:.2f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_threshold_tuning.png")
    plt.close(fig)

    # Confusion matrix at the optimal threshold
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (proba >= best_t).astype(int), display_labels=["Stayed", "Churned"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion matrix (threshold {best_t:.2f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_confusion_matrix.png")
    plt.close(fig)

    # Permutation importance on original (pre-engineering) columns
    imp = permutation_importance(
        pipe, X_test, y_test, scoring="roc_auc", n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = (
        pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean, "std": imp.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#2a6fdb")
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_feature_importance.png")
    plt.close(fig)

    default_profit = business_profit(y_test, proba, 0.5)
    summary = {
        "model": bundle["model_name"],
        "optimal_threshold": best_t,
        "at_optimal_threshold": {
            "precision": round(float(best["precision"]), 4),
            "recall": round(float(best["recall"]), 4),
            "f1": round(float(best["f1"]), 4),
            "net_savings_test_set_usd": round(float(best["profit"]), 0),
        },
        "net_savings_at_0.5_usd": round(float(default_profit), 0),
        "top_features": importance.head(5)["feature"].tolist(),
    }
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["business_evaluation"] = summary
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Save threshold alongside the model for the app / CLI
    bundle["threshold"] = best_t
    joblib.dump(bundle, MODEL_PATH)
    print("Evaluation summary:", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()

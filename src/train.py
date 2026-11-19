"""Train, cross-validate and compare ML models; persist the best pipeline."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import MODEL_PATH, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from src.data_prep import add_features, clean, load_raw, split_xy


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model) -> Pipeline:
    """Full pipeline: feature engineering -> preprocessing -> model. Accepts cleaned raw rows."""
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("prep", build_preprocessor()),
            ("model", model),
        ]
    )


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_depth=4,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }


TUNING_GRIDS = {
    "Logistic Regression": {"model__C": np.logspace(-2, 1, 12)},
    "Random Forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 8, 12, 16],
        "model__min_samples_leaf": [1, 3, 5, 10],
        "model__max_features": ["sqrt", 0.5],
    },
    "Gradient Boosting": {
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__max_depth": [3, 4, 6],
        "model__max_iter": [200, 300, 500],
        "model__l2_regularization": [0.0, 0.5, 1.0],
    },
}


def test_metrics(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
    }


def run(tune: bool = True) -> dict:
    df = clean(load_raw())
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1) Baseline comparison with 5-fold CV
    rows, fitted = [], {}
    for name, model in candidate_models().items():
        pipe = build_pipeline(model)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["roc_auc", "f1", "recall"])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        rows.append(
            {
                "model": name,
                "cv_roc_auc": scores["test_roc_auc"].mean(),
                "cv_roc_auc_std": scores["test_roc_auc"].std(),
                "cv_f1": scores["test_f1"].mean(),
                "cv_recall": scores["test_recall"].mean(),
                **{f"test_{k}": v for k, v in test_metrics(pipe, X_test, y_test).items()},
            }
        )
        print(f"{name:<20} CV ROC-AUC = {scores['test_roc_auc'].mean():.4f} +/- {scores['test_roc_auc'].std():.4f}")

    comparison = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).round(4)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["model"]
    best_pipe = fitted[best_name]

    # 2) Hyperparameter tuning of the best model family
    best_params = {}
    if tune:
        search = RandomizedSearchCV(
            build_pipeline(candidate_models()[best_name]),
            TUNING_GRIDS[best_name],
            n_iter=12, scoring="roc_auc", cv=cv, n_jobs=-1, random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)
        print(f"Tuned {best_name}: CV ROC-AUC = {search.best_score_:.4f}")
        if search.best_score_ > comparison.iloc[0]["cv_roc_auc"]:
            best_pipe = search.best_estimator_
            best_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}

    final = test_metrics(best_pipe, X_test, y_test)
    metrics = {"best_model": best_name, "best_params": best_params, "test_metrics": final}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    joblib.dump({"pipeline": best_pipe, "model_name": best_name, "X_test": X_test, "y_test": y_test,
                 "all_models": fitted}, MODEL_PATH)
    print(f"Best model: {best_name}  |  test metrics: {final}")
    print(f"Saved -> {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    run()

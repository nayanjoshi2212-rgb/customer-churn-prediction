import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from generate_data import generate  # noqa: E402
from src.data_prep import add_features, clean, split_xy  # noqa: E402
from src.train import build_pipeline  # noqa: E402


def _data(n=600):
    return clean(generate(n_rows=n, seed=0))


def test_clean_fixes_quality_issues():
    df = _data()
    assert df["total_charges"].isna().sum() == 0
    assert df.duplicated().sum() == 0
    assert set(df["churn"].unique()) <= {0, 1}
    assert not df["payment_method"].str.endswith(" ").any()
    assert df["payment_method"].nunique() == 4


def test_feature_engineering():
    feat = add_features(_data())
    for col in ["num_services", "avg_monthly_spend", "is_new_customer", "tenure_group", "has_family"]:
        assert col in feat
    assert "customer_id" not in feat
    assert feat["num_services"].between(0, 6).all()


def test_pipeline_trains_and_predicts_probabilities():
    X, y = split_xy(_data())
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    proba = pipe.predict_proba(X.head(20))[:, 1]
    assert proba.shape == (20,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_model_beats_random():
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y = split_xy(clean(generate(n_rows=3000, seed=1)))
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    assert roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]) > 0.75

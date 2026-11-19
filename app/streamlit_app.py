"""Interactive churn-prediction dashboard.

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import FIGURES_DIR, MODEL_PATH, REPORTS_DIR  # noqa: E402
from src.predict import risk_segment, score  # noqa: E402

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python main.py` first.")
    st.stop()

bundle = load_bundle()
pipe, threshold = bundle["pipeline"], bundle.get("threshold", 0.5)

st.title("📉 Customer Churn Predictor")
st.caption(f"Model: **{bundle['model_name']}**  ·  decision threshold: **{threshold:.2f}** (optimised for retention ROI)")

tab_predict, tab_batch, tab_insights = st.tabs(["Single customer", "Batch scoring", "Model & insights"])

# ---------------- Single prediction ----------------
with tab_predict:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Profile")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 6)
    with c2:
        st.subheader("Services")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        opts = ["No internet service"] if internet == "No" else ["No", "Yes"]
        security = st.selectbox("Online security", opts)
        backup = st.selectbox("Online backup", opts)
        support = st.selectbox("Tech support", opts)
        tv = st.selectbox("Streaming TV", opts)
        movies = st.selectbox("Streaming movies", opts)
    with c3:
        st.subheader("Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        monthly = st.number_input("Monthly charges ($)", 15.0, 150.0, 85.0, step=1.0)
        tickets = st.slider("Support tickets raised", 0, 10, 2)

    row = pd.DataFrame([{
        "customer_id": "NEW", "gender": gender, "senior_citizen": int(senior == "Yes"),
        "partner": partner, "dependents": dependents, "tenure_months": tenure,
        "phone_service": phone, "internet_service": internet, "online_security": security,
        "online_backup": backup, "tech_support": support, "streaming_tv": tv,
        "streaming_movies": movies, "contract": contract, "paperless_billing": paperless,
        "payment_method": payment, "monthly_charges": monthly,
        "total_charges": round(monthly * tenure, 2), "num_support_tickets": tickets,
    }])

    p = float(pipe.predict_proba(row)[:, 1][0])
    seg = risk_segment(p)
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{p:.1%}")
    m2.metric("Risk segment", seg)
    m3.metric("Target for retention?", "Yes" if p >= threshold else "No")
    st.progress(min(p, 1.0))

    tips = []
    if contract == "Month-to-month":
        tips.append("Offer a discounted 1- or 2-year contract.")
    if support == "No":
        tips.append("Bundle free tech support for 3 months.")
    if security == "No":
        tips.append("Add online security as a free trial.")
    if payment == "Electronic check":
        tips.append("Incentivise switching to auto-pay (card / bank transfer).")
    if tenure <= 12:
        tips.append("Enrol in the new-customer onboarding programme.")
    if tickets >= 3:
        tips.append("Escalate to a senior support agent for a check-in call.")
    if tips and p >= 0.4:
        st.subheader("Recommended retention actions")
        for t in tips:
            st.write("•", t)

# ---------------- Batch scoring ----------------
with tab_batch:
    st.write("Upload a CSV with the same columns as `data/raw/telco_customers.csv`.")
    up = st.file_uploader("Customer CSV", type="csv")
    if up is not None:
        scored = score(pd.read_csv(up))
        st.dataframe(scored, use_container_width=True)
        st.bar_chart(scored["risk_segment"].value_counts())
        st.download_button("Download scored CSV", scored.to_csv(index=False), "scored_customers.csv")

# ---------------- Insights ----------------
with tab_insights:
    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        tm = metrics["test_metrics"]
        cols = st.columns(len(tm))
        for col, (k, v) in zip(cols, tm.items()):
            col.metric(k.upper().replace("_", "-"), f"{v:.3f}")
    comp = REPORTS_DIR / "model_comparison.csv"
    if comp.exists():
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(comp), use_container_width=True)
    figs = sorted(FIGURES_DIR.glob("*.png"))
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, f in zip(cols, figs[i:i + 2]):
            col.image(str(f), caption=f.stem[3:].replace("_", " ").capitalize())

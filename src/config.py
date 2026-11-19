"""Central project paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "telco_customers.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "telco_clean.csv"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "churn"
ID_COL = "customer_id"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Business assumptions used for threshold optimisation (in USD)
AVG_CUSTOMER_VALUE = 1200   # revenue lost when a customer churns
RETENTION_OFFER_COST = 150  # cost of a retention offer (discount / call)
OFFER_SUCCESS_RATE = 0.40   # share of targeted churners the offer saves

for _d in (DATA_PROCESSED.parent, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

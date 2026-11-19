"""Run the full project pipeline: data -> cleaning -> EDA -> training -> evaluation -> scoring."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from generate_data import OUT_PATH, generate  # noqa: E402

from src import data_prep, eda, evaluate, train  # noqa: E402
from src.config import REPORTS_DIR  # noqa: E402
from src.predict import score  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true", help="skip hyper-parameter search (faster)")
    args = parser.parse_args()

    if not OUT_PATH.exists():
        print("[1/6] Generating dataset ...")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(OUT_PATH, index=False)
    else:
        print("[1/6] Dataset found")

    print("[2/6] Cleaning data ...")
    raw = data_prep.load_raw()
    data_prep.run()
    print("[3/6] Exploratory data analysis ...")
    eda.run()
    print("[4/6] Training & comparing models ...")
    train.run(tune=not args.no_tune)
    print("[5/6] Evaluating best model ...")
    evaluate.run()
    print("[6/6] Scoring all customers ...")
    score(raw).to_csv(REPORTS_DIR / "scored_customers.csv", index=False)
    print("\nDone! See the reports/ folder. Launch the app with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()

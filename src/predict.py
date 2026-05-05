import os
import joblib
import pandas as pd

from features import build_prediction_data
from api import get_current_gw, get_next_deadline

MODEL_PATH = "outputs/model.joblib"
FEATURES_PATH = "outputs/features.joblib"


def main():
    os.makedirs("outputs", exist_ok=True)

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    df = build_prediction_data()

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df["predicted_points"] = model.predict(df[feature_cols])
    df["deadline_utc"] = get_next_deadline()
    df["gameweek"] = get_current_gw()

    out = df.sort_values("predicted_points", ascending=False)

    csv_path = f"outputs/predictions_gw{get_current_gw()}.csv"
    out.to_csv(csv_path, index=False)

    print(f"Predictions saved to {csv_path}")
    print(out[["web_name", "team", "now_cost", "predicted_points"]].head(30))


if __name__ == "__main__":
    main()
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from features import build_training_data

MODEL_PATH = "outputs/model.joblib"
FEATURES_PATH = "outputs/features.joblib"


def main():
    os.makedirs("outputs", exist_ok=True)

    df = build_training_data()

    drop_cols = ["target", "player_id", "round"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Validation MAE: {mae:.3f}")


if __name__ == "__main__":
    main()
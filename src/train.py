import os
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from features import build_training_data

MODEL_PATH = "outputs/model.joblib"
FEATURES_PATH = "outputs/features.joblib"


def main():
    os.makedirs("outputs", exist_ok=True)

    df, feature_cols = build_training_data()

    y = df["points"]
    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        shuffle=True,
        random_state=0,
    )

    model = GradientBoostingRegressor(random_state=0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(f"MAE: {mean_absolute_error(y_test, preds):.3f}")
    print(f"R2: {r2_score(y_test, preds):.3f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved features to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
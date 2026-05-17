import os
import joblib
import numpy as np
import pandas as pd

from features import build_future_data
from api import (
    get_current_gw,
    get_bootstrap_data,
)

MODEL_PATH = "outputs/model.joblib"
FEATURES_PATH = "outputs/features.joblib"


def fixture_multiplier(fdr: int) -> float:
    """
    Fixture adjustment.

    Easy fixture => boost.
    Hard fixture => reduction.
    """
    return {
        1: 1.25,
        2: 1.12,
        3: 1.00,
        4: 0.88,
        5: 0.75,
    }.get(int(fdr), 1.0)


def main():
    os.makedirs("outputs", exist_ok=True)

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    fut_df = build_future_data(feature_cols)

    bootstrap = get_bootstrap_data()

    teams_df = pd.DataFrame(bootstrap["teams"])
    team_map = teams_df.set_index("id")["short_name"].to_dict()

    # -----------------------------
    # PURE ML PREDICTION
    # -----------------------------

    ml_pred = model.predict(fut_df[feature_cols])

    # -----------------------------
    # STRONG BASELINE
    # -----------------------------

    mins = fut_df["mins_fpf"].fillna(0)

    expected_minutes = np.clip(
        (
            0.7 * mins +
            0.3 * fut_df["ave_mins_fpf"].fillna(0)
        ),
        0,
        90,
    )

    minute_factor = expected_minutes / 90

    # attacking production
    goals = fut_df["ave_goals_fpf"].fillna(0)
    assists = fut_df["ave_assists_fpf"].fillna(0)

    xg = fut_df["ave_xG_fpf"].fillna(0)
    xa = fut_df["ave_xA_fpf"].fillna(0)

    ict = fut_df["ave_ict_fpf"].fillna(0)

    recent_points = fut_df["ave_points_fpf"].fillna(0)

    # stronger fantasy baseline
    baseline = (
        recent_points * 0.45 +
        goals * 4.5 +
        assists * 3.0 +
        xg * 5.0 +
        xa * 3.0 +
        ict * 0.03
    )

    fixture_adj = fut_df["oppo_difficulty"].apply(fixture_multiplier)

    # -----------------------------
    # FINAL BLEND
    # -----------------------------

    fut_df["xP"] = (
        (
            baseline * 0.70 +
            ml_pred * 0.30
        )
        * fixture_adj
        * minute_factor
    )

    # realistic floor
    fut_df["xP"] = fut_df["xP"].clip(lower=0)

    # round nicely
    fut_df["xP"] = fut_df["xP"].round(2)

    # -----------------------------
    # ADD TEAM / OPPONENT
    # -----------------------------

    fut_df["team_name"] = fut_df["id"].map(team_map)
    fut_df["opponent_name"] = fut_df["opponent_team"].map(team_map)

    current_gw = get_current_gw()

    preds = (
        fut_df[
            [
                "element",
                "name",
                "team_name",
                "opponent_name",
                "GW",
                "xP",
            ]
        ]
        .groupby(
            [
                "element",
                "name",
                "team_name",
                "opponent_name",
                "GW",
            ],
            as_index=False,
        )
        .agg({"xP": "sum"})
    )

    gw_preds = (
        preds[preds["GW"] == current_gw]
        .sort_values("xP", ascending=False)
        .reset_index(drop=True)
    )

    out_path = f"outputs/predictions_gw{current_gw}.csv"

    gw_preds.to_csv(out_path, index=False)

    print(gw_preds.head(40))

    print(f"\nSaved predictions to {out_path}")


if __name__ == "__main__":
    main()
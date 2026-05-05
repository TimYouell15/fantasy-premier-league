import pandas as pd
from api import (
    get_bootstrap_data,
    get_fixture_data,
    collate_player_hist,
    get_current_gw
)

ROLLING_WINDOWS = [3, 5]

def build_training_data() -> pd.DataFrame:
    hist = collate_player_hist()

    hist = hist.sort_values(["player_id", "round"]).copy()
    hist["target"] = hist["total_points"]

    base_cols = [
        "player_id",
        "round",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "saves",
        "bonus",
        "bps",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "value",
        "was_home",
        "team_h_score",
        "team_a_score",
        "target",
    ]

    hist = hist[[c for c in base_cols if c in hist.columns]].copy()

    numeric_cols = [
        c for c in hist.columns
        if c not in ["player_id", "round", "target", "was_home"]
    ]

    for col in numeric_cols:
        hist[col] = pd.to_numeric(hist[col], errors="coerce").fillna(0)

    hist["was_home"] = hist["was_home"].astype(int)

    for window in ROLLING_WINDOWS:
        for col in ["minutes", "target", "bps", "ict_index", "threat", "creativity", "influence"]:
            if col in hist.columns:
                hist[f"{col}_roll_{window}"] = (
                    hist.groupby("player_id")[col]
                    .shift(1)
                    .rolling(window)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

    hist = hist.fillna(0)
    return hist


def build_prediction_data() -> pd.DataFrame:
    bootstrap = get_bootstrap_data()
    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])
    fixtures = pd.DataFrame(get_fixture_data())

    next_gw = get_current_gw()

    upcoming = fixtures[
        (fixtures["event"] == next_gw) & (fixtures["finished"] == False)
    ].copy()

    rows = []

    for _, p in players.iterrows():
        team_id = p["team"]
        team_fixtures = upcoming[
            (upcoming["team_h"] == team_id) | (upcoming["team_a"] == team_id)
        ]

        if team_fixtures.empty:
            continue

        for _, f in team_fixtures.iterrows():
            was_home = f["team_h"] == team_id
            rows.append({
                "player_id": p["id"],
                "web_name": p["web_name"],
                "team": teams.set_index("id").loc[team_id, "short_name"],
                "position": p["element_type"],
                "now_cost": p["now_cost"] / 10,
                "selected_by_percent": float(p["selected_by_percent"]),
                "form": float(p["form"] or 0),
                "chance_of_playing_next_round": p["chance_of_playing_next_round"] or 100,
                "minutes": p["minutes"],
                "goals_scored": p["goals_scored"],
                "assists": p["assists"],
                "clean_sheets": p["clean_sheets"],
                "goals_conceded": p["goals_conceded"],
                "own_goals": p["own_goals"],
                "penalties_saved": p["penalties_saved"],
                "penalties_missed": p["penalties_missed"],
                "yellow_cards": p["yellow_cards"],
                "red_cards": p["red_cards"],
                "saves": p["saves"],
                "bonus": p["bonus"],
                "bps": p["bps"],
                "influence": float(p["influence"] or 0),
                "creativity": float(p["creativity"] or 0),
                "threat": float(p["threat"] or 0),
                "ict_index": float(p["ict_index"] or 0),
                "value": p["now_cost"],
                "was_home": int(was_home),
                "fixture_difficulty": (
                    f["team_h_difficulty"] if was_home else f["team_a_difficulty"]
                ),
                "gameweek": next_gw,
            })

    pred = pd.DataFrame(rows)

    train = build_training_data()
    latest = (
        train.sort_values(["player_id", "round"])
        .groupby("player_id")
        .tail(1)
    )

    rolling_cols = [c for c in latest.columns if "_roll_" in c]
    pred = pred.merge(
        latest[["player_id"] + rolling_cols],
        on="player_id",
        how="left",
    )

    return pred.fillna(0)
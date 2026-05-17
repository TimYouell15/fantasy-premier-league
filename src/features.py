import pandas as pd
import numpy as np

from api import (
    get_bootstrap_data,
    get_fixture_data,
    collate_player_hist,
    get_current_gw,
)

from model_config import RENAMED_COLS, BASE_FEATURES


TEAM_STRENGTH_COLS = [
    "id", "name", "strength", "strength_overall_home",
    "strength_overall_away", "strength_attack_home",
    "strength_attack_away", "strength_defence_home",
    "strength_defence_away",
]

TEAM_COLS = [
    "id", "team", "team_str", "team_str_h", "team_str_a",
    "team_str_att_h", "team_str_att_a", "team_str_def_h", "team_str_def_a",
]

OPPO_COLS = [
    "opponent_team", "oppo_name", "oppo_str", "oppo_str_h",
    "oppo_str_a", "oppo_str_att_h", "oppo_str_att_a",
    "oppo_str_def_h", "oppo_str_def_a",
]

STRING_COLS = ["position", "team", "oppo_name", "was_home"]

RAW_STAT_COLS = [
    "points", "assists", "goals", "cs", "xA", "xGI", "xG",
    "bps", "xGC", "i", "c", "t", "ict", "saves", "ps", "yc",
    "rc", "mins",
]

EVENT_STAT_COLS = RAW_STAT_COLS + ["gc", "og", "pm", "prop_mins"]

TEAM_STAT_COLS = ["goals", "xG", "xA"]


def _position_map():
    data = get_bootstrap_data()
    return pd.DataFrame(data["element_types"]).set_index("id")["singular_name_short"].to_dict()


def _prepare_reference_tables():
    data = get_bootstrap_data()
    teams = pd.DataFrame(data["teams"])
    fixtures = pd.DataFrame(get_fixture_data()).rename(columns={"id": "fixture"})

    team_cut = teams[TEAM_STRENGTH_COLS].copy()
    oppo_cut = teams[TEAM_STRENGTH_COLS].copy()

    team_cut.columns = TEAM_COLS
    oppo_cut.columns = OPPO_COLS

    fixt_cut = fixtures[
        ["fixture", "event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
    ].copy()

    return teams, fixtures, team_cut, oppo_cut, fixt_cut


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in EVENT_STAT_COLS:
        if col not in df.columns:
            df[col] = 0

    for col in RAW_STAT_COLS:
        if col not in df.columns:
            df[col] = 0

    for col in EVENT_STAT_COLS + RAW_STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["games_avail"] = df.groupby("p_id").cumcount() + 1
    df["prop_mins"] = (
        df.groupby("p_id")["mins"].transform(lambda x: x.cumsum())
        / (df["games_avail"] * 90)
    )

    for col in TEAM_STAT_COLS:
        df[f"team_{col}"] = df.groupby(["team", "fixture"])[col].transform("sum")

    ave_cols = RAW_STAT_COLS + [f"team_{c}" for c in TEAM_STAT_COLS]

    for col in ave_cols:
        df[f"ave_{col}"] = (
            df.groupby("p_id")[col].cumsum() / df["games_avail"]
        )

    fpf_source_cols = (
        EVENT_STAT_COLS
        + [f"ave_{c}" for c in RAW_STAT_COLS]
        + [f"ave_team_{c}" for c in TEAM_STAT_COLS]
    )

    rename_map = {c: f"{c}_fpf" for c in fpf_source_cols}
    df = df.rename(columns=rename_map)

    fpf_cols = list(rename_map.values())

    df["points"] = df["points_fpf"]
    df[fpf_cols] = df.groupby("p_id")[fpf_cols].shift(1)

    df = df[df["points_fpf"].notna()].copy()
    df = df[df["points"].notna()].copy()

    return df


def build_training_data() -> tuple[pd.DataFrame, list[str]]:
    bootstrap = get_bootstrap_data()
    elements = pd.DataFrame(bootstrap["elements"])

    teams, fixtures, team_cut, oppo_cut, fixt_cut = _prepare_reference_tables()
    pos_map = _position_map()

    hist = collate_player_hist()
    hist = hist.rename(columns=RENAMED_COLS)

    elements_cut = elements.copy()
    elements_cut = elements_cut.rename(columns={"id": "element"})
    elements_cut["name"] = elements_cut["first_name"] + " " + elements_cut["second_name"]
    elements_cut["position"] = elements_cut["element_type"].map(pos_map)
    elements_cut = elements_cut[["element", "name", "position", "team"]]

    df = hist.merge(elements_cut, on="element", how="left")
    df["fixture"] = df["fixture"].astype(int)
    df["GW"] = df["round"]
    df["p_id"] = df["element"].astype(str)

    df = df.rename(columns={"team": "id"})
    df = df.merge(team_cut, on="id", how="left")
    df = df.merge(oppo_cut, on="opponent_team", how="left")
    df = df.merge(
        fixt_cut[["fixture", "team_h_difficulty", "team_a_difficulty"]],
        on="fixture",
        how="left",
    )

    df["team"] = df["team"].fillna(df["id"])

    df = df.sort_values(["p_id", "fixture"])
    df = _add_lag_features(df)

    df.loc[df["was_home"] == True, "oppo_difficulty"] = df["team_h_difficulty"]
    df.loc[df["was_home"] == False, "oppo_difficulty"] = df["team_a_difficulty"]

    df["position"] = df["position"].replace({"GK": "GKP"})

    dummy = pd.get_dummies(df, columns=STRING_COLS)

    feature_cols = list(BASE_FEATURES)

    dynamic_cols = [
        c for c in dummy.columns
        if c.startswith("team_")
        or c.startswith("oppo_name_")
    ]

    for c in dynamic_cols:
        if c not in feature_cols:
            feature_cols.append(c)

    for c in feature_cols:
        if c not in dummy.columns:
            dummy[c] = 0

    dummy[feature_cols] = dummy[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    return dummy, feature_cols


def build_future_data(feature_cols: list[str]) -> pd.DataFrame:
    bootstrap = get_bootstrap_data()
    elements = pd.DataFrame(bootstrap["elements"])
    teams, fixtures, team_cut, oppo_cut, fixt_cut = _prepare_reference_tables()
    pos_map = _position_map()

    current_gw = get_current_gw()

    hist = collate_player_hist()
    hist = hist.rename(columns=RENAMED_COLS)

    elements_hist = elements.copy()
    elements_hist = elements_hist.rename(columns={"id": "element"})
    elements_hist["name"] = elements_hist["first_name"] + " " + elements_hist["second_name"]
    elements_hist["position"] = elements_hist["element_type"].map(pos_map)
    elements_hist = elements_hist[["element", "name", "position", "team"]]

    curr = hist.merge(elements_hist, on="element", how="left")
    curr["fixture"] = curr["fixture"].astype(int)
    curr["p_id"] = curr["element"].astype(str)
    curr = curr.rename(columns={"team": "id"})
    curr = curr.merge(team_cut, on="id", how="left")
    curr = curr.merge(oppo_cut, on="opponent_team", how="left")
    curr = curr.merge(
        fixt_cut[["fixture", "team_h_difficulty", "team_a_difficulty"]],
        on="fixture",
        how="left",
    )
    curr["team"] = curr["team"].fillna(curr["id"])
    curr = curr.sort_values(["p_id", "fixture"])

    curr = _add_lag_features(curr)

    fpf_cols = [c for c in curr.columns if c.endswith("_fpf")]
    latest_player_features = (
        curr.sort_values("fixture")
        .groupby("element")
        .tail(1)[["element"] + fpf_cols]
    )

    future_fixtures = fixtures[
        (fixtures["event"] >= current_gw) & (fixtures["finished"] == False)
    ].copy()

    rows = []

    for _, p in elements.iterrows():
        team_id = p["team"]

        player_fixtures = future_fixtures[
            (future_fixtures["team_h"] == team_id)
            | (future_fixtures["team_a"] == team_id)
        ]

        for _, f in player_fixtures.iterrows():
            was_home = f["team_h"] == team_id
            opponent = f["team_a"] if was_home else f["team_h"]
            oppo_difficulty = (
                f["team_h_difficulty"] if was_home else f["team_a_difficulty"]
            )

            rows.append({
                "element": p["id"],
                "name": f"{p['first_name']} {p['second_name']}",
                "position": pos_map.get(p["element_type"]),
                "id": team_id,
                "fixture": f["fixture"],
                "GW": f["event"],
                "value": p["now_cost"],
                "transfers_in": p["transfers_in_event"],
                "transfers_out": p["transfers_out_event"],
                "transfers_balance": p["transfers_in_event"] - p["transfers_out_event"],
                "was_home": was_home,
                "opponent_team": opponent,
                "oppo_difficulty": oppo_difficulty,
            })

    fut = pd.DataFrame(rows)

    fut = fut.merge(team_cut, on="id", how="left")
    fut = fut.merge(oppo_cut, on="opponent_team", how="left")
    fut = fut.merge(latest_player_features, on="element", how="left")

    fut["position"] = fut["position"].replace({"GK": "GKP"})

    fut_dummy = pd.get_dummies(fut, columns=STRING_COLS)

    for c in feature_cols:
        if c not in fut_dummy.columns:
            fut_dummy[c] = 0

    fut_dummy[feature_cols] = (
        fut_dummy[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    fut_dummy = fut_dummy[fut_dummy["points_fpf"].notna()].copy()

    return fut_dummy
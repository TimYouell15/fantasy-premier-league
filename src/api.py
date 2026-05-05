######################################################################################
#
# FPL API utility functions for fetching and transforming Fantasy Premier League data.
#
# Fantasy Premier League
#
######################################################################################

import pandas as pd
import requests
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

BASE_URL = "https://fantasy.premierleague.com/api/"


def _get(endpoint: str) -> dict:
    """
    Internal helper function to make GET requests to the FPL API.

    Args:
        endpoint (str): API endpoint string (appended to BASE_URL)

    Returns:
        dict: endpoint JSON response
    """
    resp = requests.get(f"{BASE_URL}{endpoint}")
    if resp.status_code != 200:
        raise Exception(f"Request to {endpoint} failed with status {resp.status_code}")
    return resp.json()


def get_bootstrap_data() -> dict:
    """
    Fetch general FPL data including players, teams, and game settings.

    Returns:
        dict: containing bootstrap data with keys such as:
          'elements', 'teams', 'events', 'element_types', etc.
    """
    return _get("bootstrap-static/")


def get_fixture_data() -> list:
    """
    Fetch all fixture data for the current FPL season.

    Returns:
        list: List of fixture dictionaries
    """
    return _get("fixtures/")


def get_player_data(player_id: int) -> dict:
    """
    Fetch detailed historical and fixture data for a specific player.

    Args:
        player_id (int): Unique FPL player ID

    Returns:
        dict: Dictionary containing:
          'history', 'fixtures', and 'history_past'
    """
    return _get(f"element-summary/{player_id}/")


def get_manager_details(manager_id: int) -> dict:
    """
    Fetch general information about a specific FPL manager.

    Args:
        manager_id (int): FPL manager ID

    Returns:
        dict: Dictionary containing manager profile details
    """
    return _get(f"entry/{manager_id}/")


def get_manager_history_data(manager_id: int) -> dict:
    """
    Fetch historical performance data for a specific FPL manager.

    Args:
        manager_id (int): FPL manager ID

    Returns:
        dict: Dictionary containing season and gameweek history
    """
    return _get(f"entry/{manager_id}/history/")


def get_manager_team_data(manager_id: int, gw: int) -> dict:
    """
    Fetch a manager's team selection for a given gameweek.

    Args:
        manager_id (int): FPL manager ID
        gw (int): Gameweek number

    Returns:
        dict: Dictionary containing picks, chips, and entry history
    """
    return _get(f"entry/{manager_id}/event/{gw}/picks/")


def get_total_fpl_players() -> int:
    """
    Fetch the total number of registered FPL players.

    Returns:
        int: Integer representing total number of players
    """
    return get_bootstrap_data()["total_players"]


def get_player_id_dict(web_name: bool = True) -> Dict[int, str]:
    """
    Create a mapping of player IDs to player names.

    Args:
        web_name (bool): If True, use short web name; otherwise use full name with team

    Returns:
        Dict[int, str]: Dictionary mapping player_id to player name
    """
    data = get_bootstrap_data()
    ele_df = pd.DataFrame(data["elements"])
    teams_df = pd.DataFrame(data["teams"])

    ele_df["team_name"] = ele_df["team"].map(
        teams_df.set_index("id")["short_name"]
    )

    if web_name:
        return dict(zip(ele_df["id"], ele_df["web_name"]))
    else:
        ele_df["full_name"] = (
            ele_df["first_name"] + " " +
            ele_df["second_name"] + " (" +
            ele_df["team_name"] + ")"
        )
        return dict(zip(ele_df["id"], ele_df["full_name"]))

def _fetch_player_history(player_id: int, player_name: str, retries: int = 5) -> pd.DataFrame:
    """
    Fetch one player's gameweek history with retry/backoff.

    Args:
        player_id (int): FPL player ID
        player_name (str): Player's name (for logging)
        retries (int): Number of retry attempts before giving up

    Returns:
        pd.DataFrame: DataFrame containing the player's historical gameweek data
    """
    for attempt in range(1, retries + 1):
        try:
            print(f"Fetching history for {player_name}")
            player_data = get_player_data(player_id)

            player_df = pd.DataFrame(player_data["history"])
            player_df["player_id"] = player_id
            player_df["player_name"] = player_name

            return player_df

        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(
                f"Failed fetching {player_name} "
                f"(attempt {attempt}/{retries}): {e}. Retrying in {wait}s..."
            )
            time.sleep(wait)

    print(f"Skipping {player_name} after {retries} failed attempts.")
    return pd.DataFrame()


def collate_player_hist(max_workers: int = 20) -> pd.DataFrame:
    """
    Fetch and combine historical gameweek data for all players concurrently.

    Args:
        max_workers: Number of concurrent requests. Keep this modest to avoid
                     hammering the FPL API.

    Returns:
        pd.DataFrame: Combined player history.
    """
    player_dict = get_player_id_dict()
    player_dfs = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_player_history, player_id, player_name): (
                player_id,
                player_name,
            )
            for player_id, player_name in player_dict.items()
        }

        for future in as_completed(futures):
            _, player_name = futures[future]

            try:
                df = future.result()
                if not df.empty:
                    player_dfs.append(df)
            except Exception as e:
                print(f"Unexpected error for {player_name}: {e}")

    if not player_dfs:
        return pd.DataFrame()

    return pd.concat(player_dfs, ignore_index=True)


def get_league_table() -> pd.DataFrame:
    """
    Generate a league table based on completed fixtures.

    Returns:
        pd.DataFrame: DataFrame containing team standings and statistics
    """
    fixt_df = pd.DataFrame(get_fixture_data())
    data = get_bootstrap_data()
    teams_df = pd.DataFrame(data["teams"])

    teams_id_list = teams_df["id"].unique().tolist()
    df_list = []

    for t_id in teams_id_list:
        home = fixt_df[fixt_df["team_h"] == t_id].copy()
        away = fixt_df[fixt_df["team_a"] == t_id].copy()

        home["was_home"] = True
        away["was_home"] = False

        df = pd.concat([home, away])
        df = df[df["finished"] == True].sort_values("event")

        df["gf"] = df.apply(
            lambda x: x["team_h_score"] if x["was_home"] else x["team_a_score"], axis=1
        )
        df["ga"] = df.apply(
            lambda x: x["team_a_score"] if x["was_home"] else x["team_h_score"], axis=1
        )

        df["win"] = (
            ((df["was_home"]) & (df["team_h_score"] > df["team_a_score"])) |
            ((~df["was_home"]) & (df["team_a_score"] > df["team_h_score"]))
        )
        df["draw"] = df["team_h_score"] == df["team_a_score"]
        df["loss"] = ~(df["win"] | df["draw"])

        ws = df["win"].sum()
        ds = df["draw"].sum()

        team_data = {
            "id": t_id,
            "GP": len(df),
            "W": ws,
            "D": ds,
            "L": df["loss"].sum(),
            "GF": df["gf"].sum(),
            "GA": df["ga"].sum(),
            "GD": df["gf"].sum() - df["ga"].sum(),
            "Pts": ws * 3 + ds,
        }

        df_list.append(pd.DataFrame([team_data]))

    league_df = pd.concat(df_list)
    league_df = league_df.sort_values(["Pts", "GD"], ascending=False)

    league_df["team"] = league_df["id"].map(
        teams_df.set_index("id")["short_name"]
    )

    return league_df


def get_current_gw() -> int:
    """
    Get the upcoming (next) gameweek number.

    Returns:
        int: The next gameweek ID
    """
    events_df = pd.DataFrame(get_bootstrap_data()["events"])
    return int(events_df.loc[events_df["is_next"] == True, "id"].iloc[0])


def get_next_deadline() -> pd.Timestamp:
    """
    Get the deadline timestamp for the next gameweek.

    Returns:
        pd.Timestamp: Pandas Timestamp of the next gameweek deadline (UTC)
    """
    events_df = pd.DataFrame(get_bootstrap_data()["events"])
    deadline = events_df.loc[events_df["is_next"] == True, "deadline_time"].iloc[0]
    return pd.to_datetime(deadline, utc=True)


def get_fixture_dfs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate fixture and fixture difficulty rating (FDR) tables for all teams.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of:
            1. FDR DataFrame (teams vs upcoming difficulty)
            2. Fixture DataFrame (teams vs upcoming opponents)
    """
    fixt_df = pd.DataFrame(get_fixture_data())
    data = get_bootstrap_data()
    teams_df = pd.DataFrame(data["teams"])

    fixt_df["team_h"] = fixt_df["team_h"].map(
        teams_df.set_index("id")["short_name"]
    )
    fixt_df["team_a"] = fixt_df["team_a"].map(
        teams_df.set_index("id")["short_name"]
    )

    teams = teams_df["short_name"].unique().tolist()

    team_fdr_data = []
    team_fixt_data = []

    for team in teams:
        home = fixt_df[fixt_df["team_h"] == team].copy()
        away = fixt_df[fixt_df["team_a"] == team].copy()

        home["was_home"] = True
        away["was_home"] = False

        df = pd.concat([home, away]).sort_values("event")

        df["next"] = df.apply(
            lambda x: f"{x['team_a']} (H)" if x["was_home"]
            else f"{x['team_h']} (A)",
            axis=1
        )

        df["next_fdr"] = df.apply(
            lambda x: x["team_h_difficulty"] if x["was_home"]
            else x["team_a_difficulty"],
            axis=1
        )

        team_fixt_data.append(pd.DataFrame([[team] + df["next"].tolist()]))
        team_fdr_data.append(pd.DataFrame([[team] + df["next_fdr"].tolist()]))

    team_fixt_df = pd.concat(team_fixt_data).set_index(0)
    team_fdr_df = pd.concat(team_fdr_data).set_index(0)

    return team_fdr_df, team_fixt_df


if __name__ == "__main__":
    # Example usage: Fetch and display the league table
    league_table = get_league_table()
    print(league_table)
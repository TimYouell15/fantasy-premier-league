import json
import os

STATE_PATH = "outputs/email_state.json"


def already_sent(gameweek: int) -> bool:
    if not os.path.exists(STATE_PATH):
        return False

    with open(STATE_PATH, "r") as f:
        state = json.load(f)

    return int(state.get("last_emailed_gameweek", -1)) == int(gameweek)


def mark_sent(gameweek: int):
    os.makedirs("outputs", exist_ok=True)

    with open(STATE_PATH, "w") as f:
        json.dump({"last_emailed_gameweek": int(gameweek)}, f)
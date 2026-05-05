import os
import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone

import pandas as pd

from api import get_current_gw, get_next_deadline
from state import already_sent, mark_sent


def hours_until_deadline() -> float:
    deadline = get_next_deadline()
    now = datetime.now(timezone.utc)
    return (deadline.to_pydatetime() - now).total_seconds() / 3600


def build_email_body(predictions: pd.DataFrame, gw: int) -> str:
    top = predictions.head(25)

    lines = [
        f"FPL GW{gw} predicted points",
        "",
        f"Deadline UTC: {get_next_deadline()}",
        "",
        "Top predicted players:",
        "",
    ]

    for _, row in top.iterrows():
        lines.append(
            f"{row['web_name']} | {row['team']} | "
            f"£{row['now_cost']:.1f}m | "
            f"{row['predicted_points']:.2f} pts"
        )

    return "\n".join(lines)


def send_email(subject: str, body: str, attachment_path: str):
    sender = os.environ["EMAIL_FROM"]
    recipient = os.environ["EMAIL_TO"]
    password = os.environ["EMAIL_APP_PASSWORD"]

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    with open(attachment_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="text",
            subtype="csv",
            filename=os.path.basename(attachment_path),
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)


def main():
    gw = get_current_gw()
    hrs = hours_until_deadline()

    if hrs > 36:
        print(f"Deadline is {hrs:.1f} hours away. No email sent.")
        return

    if already_sent(gw):
        print(f"Email already sent for GW{gw}. No duplicate sent.")
        return

    predictions_path = f"outputs/predictions_gw{gw}.csv"

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Missing {predictions_path}. Run predictions first.")

    predictions = pd.read_csv(predictions_path)

    subject = f"FPL GW{gw} predictions"
    body = build_email_body(predictions, gw)

    send_email(subject, body, predictions_path)
    mark_sent(gw)

    print(f"Email sent for GW{gw}.")


if __name__ == "__main__":
    main()
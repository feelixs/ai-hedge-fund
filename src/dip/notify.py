"""Post a Discord alert for the /watch-dips loop.

Used by the watcher to ping the user the moment a tracked dip record crosses an
actionable level (entry confirmation, take-profit, stop-loss, or a broken-down
watch). The webhook URL is read from the DISCORD_WEBHOOK_ALERT env var inside
this module so it never has to appear on a command line or in shell logs.

Usage:
    python -m src.dip.notify --message "..."
    echo "..." | python -m src.dip.notify --stdin
"""

import argparse
import os
import sys

import httpx
from dotenv import load_dotenv

DISCORD_CONTENT_LIMIT = 2000  # Discord rejects webhook payloads over this many chars.


def post_alert(message: str, *, timeout: float = 10.0) -> None:
    """Post `message` to the DISCORD_WEBHOOK_ALERT webhook.

    Raises KeyError if the env var is unset (no silent fallback) and
    httpx.HTTPStatusError if Discord rejects the post.
    """
    load_dotenv()
    webhook_url = os.environ["DISCORD_WEBHOOK_ALERT"]  # KeyError if missing — fail loud.

    content = message if len(message) <= DISCORD_CONTENT_LIMIT else message[: DISCORD_CONTENT_LIMIT - 1] + "…"
    response = httpx.post(webhook_url, json={"content": content}, timeout=timeout)
    response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Post a Discord alert for the dip watcher.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--message", help="The alert text to post.")
    source.add_argument("--stdin", action="store_true", help="Read the alert text from stdin.")
    args = parser.parse_args()

    message = sys.stdin.read() if args.stdin else args.message
    if not message.strip():
        parser.error("refusing to post an empty alert")

    post_alert(message)
    print("posted Discord alert", file=sys.stderr)


if __name__ == "__main__":
    main()

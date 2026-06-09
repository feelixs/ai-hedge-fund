"""Persist a hedge-fund run's final analysis to a dated file in the repo.

Writes two artifacts per run under ``analysis/<YYYY-MM-DD>/``:
- ``<TICKERS>_<HHMMSS>.json`` — structured decisions + analyst signals (+ run
  metadata), for later programmatic/automated use.
- ``<TICKERS>_<HHMMSS>.txt`` — the human-readable table (same as the console),
  with ANSI color codes stripped.
"""

import contextlib
import io
import json
import os
import re
import sys
from datetime import datetime

from src.utils.display import print_trading_output

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def save_analysis(result: dict, *, tickers: list[str], start_date: str, end_date: str, model_name: str | None = None, model_provider: str | None = None) -> str:
    """Save the run's final analysis to analysis/<date>/. Returns the JSON path."""
    now = datetime.now()
    day_dir = os.path.join(ANALYSIS_DIR, now.strftime("%Y-%m-%d"))
    os.makedirs(day_dir, exist_ok=True)

    base = f"{'-'.join(tickers)}_{now.strftime('%H%M%S')}"
    json_path = os.path.join(day_dir, base + ".json")
    txt_path = os.path.join(day_dir, base + ".txt")

    payload = {
        "timestamp": now.isoformat(timespec="seconds"),
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "model": {"name": model_name, "provider": model_provider},
        "decisions": result.get("decisions"),
        "analyst_signals": result.get("analyst_signals"),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # Human-readable table (ANSI stripped) matching the console output. A
    # rendering hiccup here must not lose the JSON or crash a finished run.
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_trading_output(result)
        with open(txt_path, "w") as f:
            f.write(_ANSI_RE.sub("", buf.getvalue()))
    except Exception as e:  # noqa: BLE001
        print(f"[analysis_output] Could not write text report: {e}", file=sys.stderr)

    print(f"\nSaved analysis to {os.path.relpath(json_path, PROJECT_ROOT)} (+ .txt)")
    return json_path

"""Load and clean a Strava bulk-export activities file.

The Strava export contains ``activities.csv`` with one row per activity.
Only the columns needed downstream are kept:
date, sport, duration (min), average HR, average power.
"""

from __future__ import annotations

import pandas as pd

# Strava activity types -> project disciplines
SPORT_MAP = {
    "Swim": "swim",
    "Ride": "bike",
    "Virtual Ride": "bike",
    "VirtualRide": "bike",
    "Gravel Ride": "bike",
    "Mountain Bike Ride": "bike",
    "Run": "run",
    "Trail Run": "run",
    "Virtual Run": "run",
    "VirtualRun": "run",
}

# Candidate column names in Strava exports (varies by export locale/version)
COLUMNS = {
    "date": ["Activity Date"],
    "type": ["Activity Type"],
    "moving_time_s": ["Moving Time"],
    "elapsed_time_s": ["Elapsed Time"],
    "avg_hr": ["Average Heart Rate"],
    "avg_power": ["Average Watts", "Average Power"],
}


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_activities(path: str) -> pd.DataFrame:
    """Return cleaned activities: date, sport, duration_min, avg_hr, avg_power.

    Non-triathlon activities (gym, hike, ...) are dropped; a summary of
    dropped rows is printed so nothing disappears silently.
    """
    raw = pd.read_csv(path)

    col = {key: _pick(raw, names) for key, names in COLUMNS.items()}
    missing = [k for k in ("date", "type") if col[k] is None]
    if missing:
        raise ValueError(f"Export is missing required columns: {missing}")

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw[col["date"]], format="mixed").dt.normalize()
    df["sport"] = raw[col["type"]].map(SPORT_MAP)

    # Prefer moving time; fall back to elapsed time. Strava stores seconds.
    time_col = col["moving_time_s"] or col["elapsed_time_s"]
    if time_col is None:
        raise ValueError("Export has neither Moving Time nor Elapsed Time")
    df["duration_min"] = pd.to_numeric(raw[time_col], errors="coerce") / 60.0

    df["avg_hr"] = (
        pd.to_numeric(raw[col["avg_hr"]], errors="coerce") if col["avg_hr"] else pd.NA
    )
    df["avg_power"] = (
        pd.to_numeric(raw[col["avg_power"]], errors="coerce") if col["avg_power"] else pd.NA
    )

    n_total = len(df)
    dropped = df["sport"].isna() | df["duration_min"].isna() | (df["duration_min"] <= 0)
    df = df[~dropped].reset_index(drop=True)
    print(f"[load_strava] kept {len(df)}/{n_total} activities "
          f"(dropped {int(dropped.sum())} non-swim/bike/run or invalid rows)")

    return df.sort_values("date").reset_index(drop=True)

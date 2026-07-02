"""TRIMP (Training Impulse) per activity, plus LTHR intensity bucketing.

TRIMP follows Banister's exponential formulation:

    TRIMP = t [min] * HRr * 0.64 * exp(1.92 * HRr)

where HRr = (HR_avg - HR_rest) / (HR_max - HR_rest) is the heart-rate
reserve fraction (male coefficients; Banister et al., 1975).

Activities without HR data get a conservative fallback based on a
sport-specific default intensity (fraction of the sport's LTHR).

Each activity is labelled with the intensity bucket used by Model 1,
relative to the *sport-specific* LTHR (HR reads lower in cycling and
swimming than running at equal effort):
    easy      avg HR <  0.85 * LTHR_sport   (Z1-Z2)
    moderate  0.85 <= avg HR/LTHR_sport < 0.95   (Z3)
    hard      avg HR >= 0.95 * LTHR_sport  (Z4+)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default avg-HR as a fraction of sport LTHR when HR is missing (typical easy session)
DEFAULT_HR_FRACTION = {"swim": 0.80, "bike": 0.78, "run": 0.82}

BUCKET_EDGES = (0.85, 0.95)   # fractions of LTHR separating easy/mod/hard
BUCKETS = ["easy", "moderate", "hard"]

# Representative HR (fraction of sport LTHR) at each bucket's midpoint,
# used for theoretical TRIMP/hour when a (sport, bucket) cell has no data.
BUCKET_MID_FRACTION = {"easy": 0.78, "moderate": 0.90, "hard": 1.00}

MIN_OBSERVATIONS = 5          # cells with fewer activities use theoretical rates


def _lthr_by_sport(profile: dict) -> dict:
    thr = profile["thresholds"]
    default = thr.get("lthr_bpm", 173)
    return thr.get("lthr_by_sport") or {s: default for s in ("swim", "bike", "run")}


def banister_trimp(duration_min, avg_hr, hr_rest: float, hr_max: float):
    hrr = ((avg_hr - hr_rest) / (hr_max - hr_rest)).clip(0.0, 1.0)
    return duration_min * hrr * 0.64 * np.exp(1.92 * hrr)


def bucket_of(avg_hr: pd.Series, lthr: pd.Series | float) -> pd.Series:
    frac = avg_hr / lthr
    return pd.cut(frac, bins=[0, BUCKET_EDGES[0], BUCKET_EDGES[1], np.inf],
                  labels=BUCKETS)


def add_trimp(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Add ``trimp`` and ``bucket`` columns to the cleaned activities frame."""
    hr = profile["heart_rate"]
    lthr = df["sport"].map(_lthr_by_sport(profile))

    out = df.copy()

    # Fill missing HR with sport-specific defaults (fraction of sport LTHR)
    default_hr = out["sport"].map(DEFAULT_HR_FRACTION) * lthr
    n_missing = int(out["avg_hr"].isna().sum())
    out["avg_hr_filled"] = out["avg_hr"].fillna(default_hr)
    if n_missing:
        print(f"[trimp] {n_missing} activities without HR -> sport-default intensity")

    out["trimp"] = banister_trimp(
        out["duration_min"], out["avg_hr_filled"],
        hr_rest=hr["hr_rest_bpm"], hr_max=hr["hr_max_bpm"],
    )
    out["bucket"] = bucket_of(out["avg_hr_filled"], lthr)
    return out


def theoretical_rate(sport: str, bucket: str, profile: dict) -> float:
    """TRIMP/hour from the Banister formula at the bucket's representative HR."""
    hr = profile["heart_rate"]
    rep_hr = BUCKET_MID_FRACTION[bucket] * _lthr_by_sport(profile)[sport]
    hrr = min(max((rep_hr - hr["hr_rest_bpm"]) / (hr["hr_max_bpm"] - hr["hr_rest_bpm"]), 0.0), 1.0)
    return 60.0 * hrr * 0.64 * float(np.exp(1.92 * hrr))


def trimp_rates(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Complete 3x3 grid of TRIMP-per-hour by (sport, bucket) — Model 1's r_db.

    Cells with >= MIN_OBSERVATIONS activities use the observed rate;
    sparse or empty cells fall back to the theoretical Banister rate.
    The ``source`` column records which was used.
    """
    g = df.groupby(["sport", "bucket"], observed=False)
    observed = g["trimp"].sum() / (g["duration_min"].sum() / 60.0)
    counts = g.size()

    rows = []
    for sport in sorted(df["sport"].unique()):
        for bucket in BUCKETS:
            n = int(counts.get((sport, bucket), 0))
            if n >= MIN_OBSERVATIONS:
                rows.append((sport, bucket, float(observed[(sport, bucket)]), n, "observed"))
            else:
                rows.append((sport, bucket, theoretical_rate(sport, bucket, profile), n, "theoretical"))
    return pd.DataFrame(rows, columns=["sport", "bucket", "trimp_per_hour", "n_activities", "source"])

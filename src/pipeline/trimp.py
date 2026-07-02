"""TRIMP (Training Impulse) per activity, plus LTHR intensity bucketing.

TRIMP follows Banister's exponential formulation:

    TRIMP = t [min] * HRr * 0.64 * exp(1.92 * HRr)

where HRr = (HR_avg - HR_rest) / (HR_max - HR_rest) is the heart-rate
reserve fraction (male coefficients; Banister et al., 1975).

Activities without HR data get a conservative fallback based on a
sport-specific default intensity (fraction of LTHR).

Each activity is also labelled with the intensity bucket used by Model 1:
    easy      avg HR <  0.85 * LTHR   (Z1-Z2)
    moderate  0.85 <= avg HR/LTHR < 0.95   (Z3)
    hard      avg HR >= 0.95 * LTHR  (Z4+)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default avg-HR as a fraction of LTHR when HR is missing (typical easy session)
DEFAULT_HR_FRACTION = {"swim": 0.80, "bike": 0.78, "run": 0.82}

BUCKET_EDGES = (0.85, 0.95)  # fractions of LTHR separating easy/mod/hard


def banister_trimp(duration_min: pd.Series, avg_hr: pd.Series,
                   hr_rest: float, hr_max: float) -> pd.Series:
    hrr = ((avg_hr - hr_rest) / (hr_max - hr_rest)).clip(lower=0.0, upper=1.0)
    return duration_min * hrr * 0.64 * np.exp(1.92 * hrr)


def bucket_of(avg_hr: pd.Series, lthr: float) -> pd.Series:
    frac = avg_hr / lthr
    return pd.cut(frac, bins=[0, BUCKET_EDGES[0], BUCKET_EDGES[1], np.inf],
                  labels=["easy", "moderate", "hard"])


def add_trimp(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Add ``trimp`` and ``bucket`` columns to the cleaned activities frame."""
    hr = profile["heart_rate"]
    lthr = profile["thresholds"]["lthr_bpm"]

    out = df.copy()

    # Fill missing HR with sport-specific defaults (fraction of LTHR)
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


def trimp_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate TRIMP-per-hour by (sport, bucket) — the r_db parameters of Model 1."""
    g = df.groupby(["sport", "bucket"], observed=True)
    rates = (g["trimp"].sum() / (g["duration_min"].sum() / 60.0)).rename("trimp_per_hour")
    return rates.reset_index()

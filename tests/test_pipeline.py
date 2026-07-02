"""End-to-end and unit tests for the data pipeline (synthetic Strava export)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.pipeline.banister import ctl_atl, daily_load
from src.pipeline.load_strava import load_activities
from src.pipeline.trimp import add_trimp, banister_trimp, bucket_of

PROFILE = {
    "thresholds": {"lthr_bpm": 173,
                   "lthr_by_sport": {"run": 173, "bike": 166, "swim": 164}},
    "heart_rate": {"hr_max_bpm": 195, "hr_rest_bpm": 50},
    "banister": {"tau_ctl_days": 42, "tau_atl_days": 7},
}


@pytest.fixture
def synthetic_csv(tmp_path):
    """Strava-format activities.csv: 60 days of swim/bike/run + one gym session."""
    rng = np.random.default_rng(42)
    rows = []
    start = pd.Timestamp("2026-01-01")
    sports = ["Swim", "Ride", "Run"]
    for day in range(60):
        sport = sports[day % 3]
        rows.append({
            "Activity Date": (start + pd.Timedelta(days=day)).strftime("%b %d, %Y, %I:%M:%S %p"),
            "Activity Type": sport,
            "Moving Time": int(rng.uniform(30, 120) * 60),   # seconds
            "Elapsed Time": 0,
            "Average Heart Rate": float(rng.uniform(120, 175)),
            "Average Watts": 200.0 if sport == "Ride" else "",
        })
    rows.append({  # must be dropped
        "Activity Date": "Jan 05, 2026, 06:00:00 PM",
        "Activity Type": "Weight Training",
        "Moving Time": 3600, "Elapsed Time": 0,
        "Average Heart Rate": 110.0, "Average Watts": "",
    })
    path = tmp_path / "activities.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_loader_maps_and_drops(synthetic_csv):
    df = load_activities(synthetic_csv)
    assert len(df) == 60                                  # gym session dropped
    assert set(df["sport"]) == {"swim", "bike", "run"}
    assert (df["duration_min"] > 0).all()


def test_trimp_formula_scalar():
    # 60 min at HRr = 0.5 -> TRIMP = 60 * 0.5 * 0.64 * e^0.96
    got = banister_trimp(pd.Series([60.0]), pd.Series([122.5]), hr_rest=50, hr_max=195)
    assert math.isclose(got.iloc[0], 60 * 0.5 * 0.64 * math.exp(1.92 * 0.5), rel_tol=1e-9)


def test_buckets():
    hr = pd.Series([120.0, 150.0, 170.0])  # 69%, 87%, 98% of LTHR 173
    assert list(bucket_of(hr, 173)) == ["easy", "moderate", "hard"]


def test_ctl_atl_recursion():
    # constant load L: CTL_t converges to L; check one step exactly
    load = pd.Series([100.0, 100.0], index=pd.date_range("2026-01-01", periods=2))
    traj = ctl_atl(load, tau_ctl=42, tau_atl=7)
    lam_c = math.exp(-1 / 42)
    step1 = (1 - lam_c) * 100
    assert math.isclose(traj["ctl"].iloc[0], step1, rel_tol=1e-9)
    assert math.isclose(traj["ctl"].iloc[1], lam_c * step1 + (1 - lam_c) * 100, rel_tol=1e-9)
    assert traj["tsb"].iloc[1] == pytest.approx(traj["ctl"].iloc[0] - traj["atl"].iloc[0])


def test_trimp_rates_complete_grid(synthetic_csv):
    from src.pipeline.trimp import trimp_rates
    activities = add_trimp(load_activities(synthetic_csv), PROFILE)
    rates = trimp_rates(activities, PROFILE)
    assert len(rates) == 9                                # full 3x3 grid
    assert (rates["trimp_per_hour"] > 0).all()
    assert set(rates["source"]) <= {"observed", "theoretical"}
    # harder buckets must cost more TRIMP/hour within each sport
    for sport, grp in rates.groupby("sport"):
        by = grp.set_index("bucket")["trimp_per_hour"]
        assert by["easy"] < by["moderate"] < by["hard"]


def test_end_to_end(synthetic_csv):
    activities = add_trimp(load_activities(synthetic_csv), PROFILE)
    assert (activities["trimp"] > 0).all()
    traj = ctl_atl(daily_load(activities))
    assert len(traj) == 60                                # continuous daily index
    assert (traj["ctl"] >= 0).all() and (traj["atl"] >= 0).all()
    # ATL reacts faster than CTL early in the block
    assert traj["atl"].iloc[6] > traj["ctl"].iloc[6]

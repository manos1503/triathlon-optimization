"""Tests for Model 1: feasibility, constraint satisfaction, solution shape."""

import math

import pandas as pd
import pytest

from src.models.model1_training import BUCKETS, DISCIPLINES, build_model, extract_solution, solve

PROFILE = {
    "constraints": {
        "weekly_hours_max": 12.0,
        "weekly_hours_min_per_discipline": {"swim": 1.0, "bike": 2.0, "run": 1.5},
    },
    "banister": {"tau_ctl_days": 42, "tau_atl_days": 7, "k1": 1.0, "k2": 2.0},
    "macrocycle": {
        "weeks": 16,
        "phases": {
            "base": {"weeks": [1, 6]},
            "build": {"weeks": [7, 12]},
            "peak": {"weeks": [13, 14]},
            "taper": {"weeks": [15, 16]},
        },
    },
    "model1": {
        "atl_max": 110,
        "polarization_max": 0.20,
        "ramp_max": 0.10,
        "hours_multiplier": {"base": 0.90, "build": 1.00, "peak": 1.00, "taper": 0.65},
        "taper_load_fraction": {15: 0.60, 16: 0.40},
    },
}

STATE = {"ctl": 78.0, "atl": 91.0}

RATES = pd.DataFrame(
    [(d, b, r) for d, rr in
     {"swim": (44, 94, 137), "bike": (68, 87, 143), "run": (82, 103, 166)}.items()
     for b, r in zip(BUCKETS, rr)],
    columns=["sport", "bucket", "trimp_per_hour"],
)


@pytest.fixture(scope="module")
def solution():
    prob, v = build_model(PROFILE, STATE, RATES)
    assert solve(prob) == "Optimal"
    return extract_solution(prob, v, STATE)


def test_hours_constraints(solution):
    weekly = solution["plan"].groupby("week")["hours"].sum()
    assert (weekly <= 12.0 + 1e-6).all()
    per_disc = solution["plan"].groupby(["week", "discipline"])["hours"].sum()
    for d in DISCIPLINES:
        mins = per_disc.xs(d, level="discipline")
        assert (mins >= PROFILE["constraints"]["weekly_hours_min_per_discipline"][d] - 1e-6).all()


def test_polarization(solution):
    plan = solution["plan"]
    for w, grp in plan.groupby("week"):
        total = grp["hours"].sum()
        hard = grp.loc[grp["bucket"].isin(["moderate", "hard"]), "hours"].sum()
        assert hard <= 0.20 * total + 1e-6


def test_atl_ceiling_and_dynamics(solution):
    traj = solution["trajectory"]
    assert (traj["atl"] <= 110 + 1e-6).all()
    # re-check Banister recursion on the solved trajectory
    lam_c = math.exp(-7 / 42)
    prev = STATE["ctl"]
    for _, row in traj.iterrows():
        expect = lam_c * prev + (1 - lam_c) * row["load"] / 7.0
        assert row["ctl"] == pytest.approx(expect, rel=1e-5)
        prev = row["ctl"]


def test_taper_shape(solution):
    traj = solution["trajectory"].set_index("week")
    avg_build = traj.loc[7:12, "load"].mean()
    assert traj.loc[15, "load"] <= 0.60 * avg_build + 1e-4
    assert traj.loc[16, "load"] <= 0.40 * avg_build + 1e-4
    # race-day form must improve on the starting state
    race_tsb = traj.loc[16, "ctl"] - traj.loc[16, "atl"]
    assert race_tsb > STATE["ctl"] - STATE["atl"]


def test_infeasible_when_minimums_exceed_ceiling():
    bad = {**PROFILE, "constraints": {
        "weekly_hours_max": 3.0,   # below sum of minimums (4.5)
        "weekly_hours_min_per_discipline": {"swim": 1.0, "bike": 2.0, "run": 1.5},
    }}
    prob, _ = build_model(bad, STATE, RATES)
    assert solve(prob) != "Optimal"

"""Tests for Model 2: constraint satisfaction, readiness link, relaxation bound."""

import pandas as pd
import pulp
import pytest

from src.models.model2_calendar import (build_model, extract_solution,
                                        project_season_ctl, readiness, solve)

PROFILE = {
    "banister": {"tau_ctl_days": 42, "tau_atl_days": 7},
    "model1": {"ramp_max": 0.10},
    "model2": {
        "budget_eur": 2500,
        "max_races_per_month": 2,
        "recovery_weeks": {"sprint": 1, "olympic": 2, "70.3": 4},
        "value_weights": {"points": 1.0, "strategic": 3.0, "preference": 2.0},
        "ctl_required": {"sprint": 60, "olympic": 75, "70.3": 90},
        "offseason_ctl": 60,
        "offseason_weekly_load": 400,
        "sustained_weekly_load": 770,
    },
}

RACES = pd.DataFrame([
    # id, name, date, week, class, fee, travel, pts, strat, pref
    ("A", "Early 70.3", "2027-03-07", 9, "70.3", 350, 700, 85, 7, 7),
    ("B", "Home Sprint", "2027-03-14", 11, "sprint", 45, 0, 20, 3, 6),
    ("C", "May Olympic", "2027-05-09", 19, "olympic", 70, 140, 35, 4, 8),
    ("D", "May 70.3", "2027-05-30", 22, "70.3", 340, 520, 80, 8, 7),
    ("E", "June Olympic", "2027-06-06", 23, "olympic", 60, 0, 35, 5, 7),
    ("F", "Sept Champs", "2027-09-12", 37, "olympic", 65, 0, 60, 9, 9),
    ("G", "Sept 70.3", "2027-09-26", 39, "70.3", 330, 250, 85, 9, 10),
], columns=["id", "name", "date", "week", "distance_class",
            "entry_fee_eur", "travel_cost_eur", "points", "strategic", "preference"])


@pytest.fixture(scope="module")
def solution():
    prob, v = build_model(RACES, PROFILE)
    assert solve(prob) == "Optimal"
    return extract_solution(RACES, v), pulp.value(prob.objective)


def test_ctl_projection_monotone_to_steady_state():
    ctl = project_season_ctl(PROFILE)
    assert all(b >= a - 1e-9 for a, b in zip(ctl, ctl[1:]))
    assert ctl[-1] == pytest.approx(770 / 7, rel=0.02)   # converges to L/7


def test_readiness_gates_early_703():
    ready = readiness(RACES, PROFILE)
    assert not ready[0]        # week-9 70.3 requires CTL 90, projection ~88
    assert ready[1:].all()     # everything else is reachable


def test_budget_and_recovery(solution):
    sol, _ = solution
    sel = sol[sol["selected"]]
    assert sel["cost"].sum() <= PROFILE["model2"]["budget_eur"]
    rec = PROFILE["model2"]["recovery_weeks"]
    weeks = sel.sort_values("week")[["week", "distance_class"]].to_numpy()
    for (w1, c1), (w2, _) in zip(weeks, weeks[1:]):
        assert w2 - w1 >= rec[str(c1)] + 1


def test_champs_vs_703_conflict(solution):
    sol, _ = solution
    sel = set(sol.loc[sol["selected"], "id"])
    assert not {"F", "G"} <= sel   # 2-week gap < olympic recovery + 1


def test_lp_relaxation_upper_bounds_mip(solution):
    _, z_mip = solution
    prob_lp, _ = build_model(RACES, PROFILE, relax=True)
    assert solve(prob_lp) == "Optimal"
    assert pulp.value(prob_lp.objective) >= z_mip - 1e-6

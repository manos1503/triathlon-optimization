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
        "value_weights": {"points": 1.0, "strategic": 3.0, "preference": 2.0, "fit": 2.0},
        "preference_weights": {"organization": 0.3, "scenery": 0.3, "swim_quality": 0.4},
        "terrain_preference": {"bike": "flat", "run": "flat"},
        "weather_risk_adjust": True,
        "vacation_days_budget": 12,
        "require_a_race": True,
        "ctl_required": {"sprint": 60, "olympic": 75, "70.3": 90},
        "offseason_ctl": 60,
        "offseason_weekly_load": 400,
        "sustained_weekly_load": 770,
    },
}

RACES = pd.DataFrame([
    # id, name, date, week, class, fee, travel, days, pts, strat, a_race, org, scenery, swim, bike_gain, run_gain, weather
    ("A", "Early 70.3", "2027-03-07", 9, "70.3", 350, 700, 3, 85, 7, 0, 9, 6, 8, 300, 50, 0.90),
    ("B", "Home Sprint", "2027-03-14", 11, "sprint", 45, 0, 0, 20, 3, 0, 6, 5, 6, 100, 30, 0.85),
    ("C", "May Olympic", "2027-05-09", 19, "olympic", 70, 140, 2, 35, 4, 0, 7, 9, 9, 550, 120, 0.85),
    ("D", "May 70.3", "2027-05-30", 22, "70.3", 340, 520, 3, 80, 8, 0, 9, 7, 6, 1000, 150, 0.75),
    ("E", "June Olympic", "2027-06-06", 23, "olympic", 60, 0, 0, 35, 5, 0, 6, 6, 7, 200, 40, 0.90),
    ("F", "Sept Champs", "2027-09-12", 37, "olympic", 65, 0, 0, 60, 9, 1, 7, 6, 7, 300, 60, 0.85),
    ("G", "Sept 70.3", "2027-09-26", 39, "70.3", 330, 250, 2, 85, 9, 1, 9, 10, 10, 900, 200, 0.85),
], columns=["id", "name", "date", "week", "distance_class",
            "entry_fee_eur", "travel_cost_eur", "vacation_days", "points", "strategic",
            "a_race", "organization", "scenery", "swim_quality",
            "bike_elev_gain_m", "run_elev_gain_m", "weather_prob"])


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


def test_vacation_days_budget(solution):
    sol, _ = solution
    sel = sol[sol["selected"]]
    assert sel["vacation_days"].sum() <= PROFILE["model2"]["vacation_days_budget"]


def test_tight_days_budget_forces_domestic():
    import copy
    tight = copy.deepcopy(PROFILE)
    tight["model2"]["vacation_days_budget"] = 2
    prob, v = build_model(RACES, tight)
    assert solve(prob) == "Optimal"
    sel = extract_solution(RACES, v)
    chosen = sel[sel["selected"]]
    assert chosen["vacation_days"].sum() <= 2
    # the cheap-in-euros/expensive-in-days trade-off must bite:
    # fewer away races than with the default 12-day budget
    assert (chosen["vacation_days"] > 0).sum() <= 1


def test_suitability_flat_preference():
    from src.models.model2_calendar import suitability_score
    fit = suitability_score(RACES, {"bike": "flat", "run": "flat"})
    # flat Dubai-like 70.3 (A) must fit better than mountainous 70.3 (D)
    assert fit[0] > fit[3]
    # inverting the preference inverts the ordering
    fit_hilly = suitability_score(RACES, {"bike": "hilly", "run": "hilly"})
    assert fit_hilly[3] > fit_hilly[0]
    assert fit.between(0, 10).all()


def test_a_race_requirement(solution):
    sol, _ = solution
    sel = sol[sol["selected"]]
    assert (sel["a_race"] == 1).any()


def test_a_race_forced_even_when_low_value():
    # zero out the A-races' points so they are unattractive; constraint must still pick one
    races = RACES.copy()
    races.loc[races["a_race"] == 1, ["points", "strategic"]] = 0
    prob, v = build_model(races, PROFILE)
    assert solve(prob) == "Optimal"
    sel = extract_solution(races, v)
    assert (sel.loc[sel["selected"], "a_race"] == 1).any()


def test_weather_risk_lowers_value():
    from src.models.model2_calendar import race_value
    m2 = PROFILE["model2"]
    risky = race_value(RACES, m2)
    m2_no = {**m2, "weather_risk_adjust": False}
    raw = race_value(RACES, m2_no)
    assert (risky <= raw + 1e-9).all()
    assert (risky < raw).any()


def test_lp_relaxation_upper_bounds_mip(solution):
    _, z_mip = solution
    prob_lp, _ = build_model(RACES, PROFILE, relax=True)
    assert solve(prob_lp) == "Optimal"
    assert pulp.value(prob_lp.objective) >= z_mip - 1e-6

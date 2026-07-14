"""Tests for Model 3: feasibility, constraint satisfaction, dual signs, monotonicity."""

import pulp
import pytest
import yaml

from src.models.model3_pacing import (HARD, LEGS, ZONES, build_model,
                                      extract_solution, solve, zone_parameters)

with open("docs/athlete_profile.yaml") as f:
    PROFILE = yaml.safe_load(f)

CTL = 81.6


@pytest.fixture(scope="module")
def solution():
    prob, v = build_model(PROFILE, CTL)
    assert solve(prob) == "Optimal"
    return extract_solution(prob, v)


def test_zone_parameters_monotone():
    zp = zone_parameters(PROFILE)
    for leg in LEGS:
        sub = zp[zp.leg == leg].set_index("zone").loc[ZONES]
        assert sub["speed_km_min"].is_monotonic_increasing
        assert sub["energy_kj_min"].is_monotonic_increasing


def test_distances_covered(solution):
    d = PROFILE["model3"]["distances_km"]
    km = solution["plan"].groupby("leg")["km"].sum()
    phi = PROFILE["model3"]["bike_run_coupling_km_per_kj"]
    hard_bike_kj = solution["plan"].query("leg=='bike' and zone in @HARD")["kj"].sum()
    assert km["swim"] >= d["swim"] - 1e-6
    assert km["bike"] >= d["bike"] - 1e-6
    assert km["run"] - phi * hard_bike_kj >= d["run"] - 1e-6


def test_energy_budget_and_hard_caps(solution):
    assert solution["energy_used_kj"] <= solution["e_tot"] + 1e-4
    plan = solution["plan"]
    for leg in LEGS:
        leg_rows = plan[plan.leg == leg]
        hard = leg_rows[leg_rows.zone.isin(HARD)]["minutes"].sum()
        cap = PROFILE["model3"]["max_hard_fraction"][leg]
        assert hard <= cap * leg_rows["minutes"].sum() + 1e-6


def test_dual_signs(solution):
    duals = solution["duals"]
    # covering constraints (>= distance) must have positive prices in a min problem
    assert duals["dist_swim"] > 0 and duals["dist_bike"] > 0 and duals["dist_run"] > 0
    # more energy can only help: non-positive dual on <= budget
    assert duals["energy_budget"] <= 1e-9


def test_more_fitness_never_slower():
    times = []
    for ctl in (75.0, 82.0, 90.0):
        prob, _ = build_model(PROFILE, ctl)
        assert solve(prob) == "Optimal"
        times.append(pulp.value(prob.objective))
    assert times[0] >= times[1] >= times[2] - 1e-6


def test_higher_ftp_not_slower_below_saturation():
    prob_lo, _ = build_model(PROFILE, CTL, ftp_watts=240)
    prob_hi, _ = build_model(PROFILE, CTL, ftp_watts=255)
    assert solve(prob_lo) == "Optimal" and solve(prob_hi) == "Optimal"
    assert pulp.value(prob_hi.objective) <= pulp.value(prob_lo.objective) + 1e-6


def test_infeasible_when_energy_too_low():
    prob, _ = build_model(PROFILE, ctl_race_day=30.0)   # tiny energy budget
    assert solve(prob) != "Optimal"

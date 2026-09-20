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


def test_fueling_decision_mode():
    import copy
    p = copy.deepcopy(PROFILE)
    p["model3"]["distances_km"] = {"swim": 1.9, "bike": 90.0, "run": 21.1}
    p["model3"]["energy_budget_kj_per_ctl"] = 142
    prob, v = build_model(p, ctl_race_day=71.4, fueling_decision=True)
    assert solve(prob) == "Optimal"
    sol = extract_solution(prob, v)
    # no eating while swimming; absorption ceilings respected per leg
    assert v["F"]["swim"].varValue == pytest.approx(0.0, abs=1e-6)
    rbar = p["model3"]["fueling_max_kj_per_min"]
    for leg in LEGS:
        assert v["F"][leg].varValue <= rbar[leg] * sol["leg_times"][leg] + 1e-4
    # at 70.3 the gut is a bottleneck: absorption constraints bind on bike & run
    duals = sol["duals"]
    assert duals["absorption_bike"] < -1e-6
    assert duals["absorption_run"] < -1e-6


def test_fueling_extension():
    import copy
    p = copy.deepcopy(PROFILE)
    p["model3"]["distances_km"] = {"swim": 1.9, "bike": 90.0, "run": 21.1}
    p["model3"]["energy_budget_kj_per_ctl"] = 142
    # 70.3 infeasible on stored energy alone, feasible with standard fueling
    prob0, _ = build_model(p, ctl_race_day=71.4, fueling_kj_min=0.0)
    assert solve(prob0) != "Optimal"
    prob21, _ = build_model(p, ctl_race_day=71.4, fueling_kj_min=21.0)
    assert solve(prob21) == "Optimal"
    # more fueling can only help
    prob25, _ = build_model(p, ctl_race_day=71.4, fueling_kj_min=25.0)
    assert solve(prob25) == "Optimal"
    assert pulp.value(prob25.objective) <= pulp.value(prob21.objective) + 1e-6
    # r = 0 (default) reproduces the baseline model exactly
    prob_a, _ = build_model(PROFILE, ctl_race_day=81.6)
    prob_b, _ = build_model(PROFILE, ctl_race_day=81.6, fueling_kj_min=0.0)
    assert solve(prob_a) == solve(prob_b) == "Optimal"
    assert pulp.value(prob_a.objective) == pytest.approx(pulp.value(prob_b.objective))


def test_gradient_slows_the_bike():
    """A hilly course must be slower than a flat one, all else equal."""
    from src.models.model3_pacing import zone_parameters
    flat = zone_parameters(PROFILE, gradient_m_per_km=0.0)
    hilly = zone_parameters(PROFILE, gradient_m_per_km=15.0)
    fb = flat[flat.leg == "bike"]["speed_km_min"].to_numpy()
    hb = hilly[hilly.leg == "bike"]["speed_km_min"].to_numpy()
    assert (hb < fb).all()
    # a bike gradient must not touch the swim or the run
    for leg in ("swim", "run"):
        a = flat[flat.leg == leg]["speed_km_min"].to_numpy()
        b = hilly[hilly.leg == leg]["speed_km_min"].to_numpy()
        assert (a == b).all()


def test_flat_reference_course_unchanged():
    """G = 0 is the baseline flat course: the shipped results must not move."""
    prob, _ = build_model(PROFILE, ctl_race_day=81.6, gradient_m_per_km=0.0)
    assert solve(prob) == "Optimal"
    assert pulp.value(prob.objective) == pytest.approx(140.46, abs=0.1)


def test_grade_factor_physics():
    """The factor is derived, not fitted: check its qualitative properties."""
    from src.models.model3_pacing import grade_factor
    assert grade_factor(PROFILE, 0.0) == 1.0
    f5, f15, f20 = (grade_factor(PROFILE, g) for g in (5.0, 15.0, 20.0))
    assert 1.0 > f5 > f15 > f20 > 0.5          # monotone, never absurd
    # convex in G: the marginal penalty grows with gradient
    assert (f5 - f15) / 10.0 < (f15 - f20) / 5.0
    # sanity against an independent hand calculation at 15 m/km (~8-9% loss)
    assert 0.90 < f15 < 0.93

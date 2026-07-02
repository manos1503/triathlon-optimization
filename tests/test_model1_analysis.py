"""Tests for the Model 1 duality/sensitivity analysis modules."""

import numpy as np
import pandas as pd
import pulp
import pytest

from src.analysis.alt_optima import secondary_solve
from src.analysis.heuristic_benchmark import heuristic_trajectory
from src.analysis.rhs_ranging import sweep
from src.analysis.shadow_prices import shadow_price_table
from src.models.model1_training import build_model, extract_solution, solve

from .test_model1 import PROFILE, RATES, STATE


@pytest.fixture(scope="module")
def base():
    prob, v = build_model(PROFILE, STATE, RATES)
    assert solve(prob) == "Optimal"
    return prob, extract_solution(prob, v, STATE)


def test_rhs_value_function_concave_nondecreasing():
    df = sweep(PROFILE, STATE, RATES, grid=np.arange(9.0, 15.01, 1.0))
    obj = df.loc[df.status == "Optimal", "objective"].to_numpy()
    assert (np.diff(obj) >= -1e-6).all()          # non-decreasing in RHS
    assert (np.diff(np.diff(obj)) <= 1e-4).all()  # concave (decreasing slopes)


def test_shadow_price_table_structure(base):
    _, sol = base
    table = shadow_price_table(sol["duals"])
    assert not table.empty
    assert {"constraint", "type", "week", "shadow_price", "meaning"} <= set(table.columns)
    # no equality-definition constraints leak through
    assert not table["constraint"].str.startswith("def_").any()
    # taper-week discipline minimums must hurt (negative pi), if binding
    taper_mins = table[(table["type"].str.startswith("min_")) & (table["week"] == 16)]
    assert (taper_mins["shadow_price"] < 0).all()


def test_alternative_optima_same_performance():
    prob, _ = build_model(PROFILE, STATE, RATES)
    assert solve(prob) == "Optimal"
    p_star = pulp.value(prob.objective)
    lean = secondary_solve(PROFILE, STATE, RATES, p_star, pulp.LpMinimize)
    bulky = secondary_solve(PROFILE, STATE, RATES, p_star, pulp.LpMaximize)
    assert lean["performance"] == pytest.approx(p_star, abs=1e-3)
    assert bulky["performance"] == pytest.approx(p_star, abs=1e-3)
    assert bulky["total_hours"] >= lean["total_hours"]


def test_heuristic_never_beats_optimal(base):
    prob, _ = base
    p_opt = pulp.value(prob.objective)
    heur = heuristic_trajectory(PROFILE, STATE, RATES)
    ban = PROFILE["banister"]
    p_heur = ban["k1"] * heur["ctl"].iloc[-1] - ban["k2"] * heur["atl"].iloc[-1]
    # heuristic is a feasible-or-worse point of the same dynamics: cannot beat LP optimum
    assert p_heur <= p_opt + 1e-6

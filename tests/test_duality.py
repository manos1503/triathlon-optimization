"""Formal verification of LP duality theory on the solved models.

Two theorems are checked numerically on Models 1 and 3:

Strong duality: at the optimum, the primal objective equals the dual
objective sum_i y_i b_i (all decision variables here have lower bound 0 and
no finite upper bound, so variable-bound duals vanish and constraint duals
suffice).

Complementary slackness: for every constraint, y_i * slack_i = 0 --- a
constraint can only have a nonzero shadow price if it is binding.
"""

import pandas as pd
import pulp
import pytest
import yaml

from src.models.model1_training import build_model as m1_build
from src.models.model1_training import solve as m1_solve
from src.models.model3_pacing import build_model as m3_build
from src.models.model3_pacing import solve as m3_solve

from .test_model1 import PROFILE as M1_PROFILE
from .test_model1 import RATES, STATE

with open("docs/athlete_profile.yaml") as f:
    FULL_PROFILE = yaml.safe_load(f)


def dual_objective(prob: pulp.LpProblem) -> float:
    """sum_i y_i * b_i, with b_i = -constant of the pulp constraint expression."""
    return sum(c.pi * (-c.constant) for c in prob.constraints.values())


def max_cs_violation(prob: pulp.LpProblem) -> float:
    """max_i |y_i * slack_i| over all constraints."""
    return max(abs((c.pi or 0.0) * (c.slack or 0.0)) for c in prob.constraints.values())


@pytest.fixture(scope="module")
def solved_m1():
    prob, _ = m1_build(M1_PROFILE, STATE, RATES)
    assert m1_solve(prob) == "Optimal"
    return prob


@pytest.fixture(scope="module")
def solved_m3():
    prob, _ = m3_build(FULL_PROFILE, 81.6)
    assert m3_solve(prob) == "Optimal"
    return prob


def test_strong_duality_model1(solved_m1):
    primal = pulp.value(solved_m1.objective)
    assert primal == pytest.approx(dual_objective(solved_m1), rel=1e-6, abs=1e-6)


def test_strong_duality_model3(solved_m3):
    primal = pulp.value(solved_m3.objective)
    # Model 3's objective includes constant transition times, which the dual
    # objective (over constraints only) does not carry
    transitions = (FULL_PROFILE["model3"]["transitions_min"]["t1"]
                   + FULL_PROFILE["model3"]["transitions_min"]["t2"])
    assert primal - transitions == pytest.approx(dual_objective(solved_m3),
                                                 rel=1e-6, abs=1e-6)


def test_complementary_slackness_model1(solved_m1):
    assert max_cs_violation(solved_m1) < 1e-6


def test_complementary_slackness_model3(solved_m3):
    assert max_cs_violation(solved_m3) < 1e-6

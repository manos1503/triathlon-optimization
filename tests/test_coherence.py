"""Tests for pipeline coherence and the Model 1 <-> Model 2 fixed point."""

import pandas as pd
import pytest
import yaml

from src.analysis.coherence import (LAMBDA, candidate_plans, evaluate,
                                    rankings_agree)
from src.analysis.pipeline_loop import iterate
from src.models.model3_pacing import build_model as m3_build
from src.models.model3_pacing import energy_budget
from src.models.model3_pacing import solve as m3_solve

from .test_model1 import PROFILE as M1_PROFILE
from .test_model1 import RATES, STATE

with open("docs/athlete_profile.yaml") as f:
    FULL = yaml.safe_load(f)


# --- energy budget ---------------------------------------------------------

def test_energy_budget_backward_compatible():
    """lam = 0 reproduces the validated fitness-only budget exactly."""
    k_e = FULL["model3"]["energy_budget_kj_per_ctl"]
    assert energy_budget(FULL, 81.6, atl=50.0, lam=0.0) == pytest.approx(k_e * 81.6)
    # ATL must be ignored when lam = 0
    assert energy_budget(FULL, 81.6, atl=0.0, lam=0.0) == \
           energy_budget(FULL, 81.6, atl=99.0, lam=0.0)


def test_form_energy_decreases_with_fatigue():
    hi = energy_budget(FULL, 81.6, atl=40.0, lam=1.5)
    lo = energy_budget(FULL, 81.6, atl=60.0, lam=1.5)
    assert hi > lo


def test_model3_unchanged_by_default():
    """The default profile must still produce the validated 140.4-min race."""
    prob, _ = m3_build(FULL, ctl_race_day=81.6)
    assert m3_solve(prob) == "Optimal"
    import pulp
    assert pulp.value(prob.objective) == pytest.approx(140.46, abs=0.1)


# --- the coherence theorem -------------------------------------------------

@pytest.fixture(scope="module")
def plans():
    return candidate_plans(M1_PROFILE, STATE, RATES)


def test_plans_are_distinct(plans):
    assert len(plans) >= 3
    assert plans.duplicated(subset=["ctl_T", "atl_T"]).sum() == 0


def test_coherent_lambda_aligns_rankings(plans):
    """lam = k2/k1  =>  ordering by p_T matches ordering by race time."""
    k1 = M1_PROFILE["banister"]["k1"]
    coh = evaluate(FULL, plans, lam=LAMBDA, k1=k1, k2=LAMBDA * k1)
    assert coh["race_time_min"].notna().all()
    assert rankings_agree(coh)


def test_fitness_only_budget_is_not_coherent(plans):
    """With lam = 0 the race time cannot discriminate between plans."""
    k1 = M1_PROFILE["banister"]["k1"]
    inc = evaluate(FULL, plans, lam=0.0, k1=k1, k2=2.0)
    assert not rankings_agree(inc)


def test_positivity_window():
    """The assumed k2 = 2.0 makes the coherent budget negative; 1.5 does not."""
    ctl, atl = 81.641, 46.539
    assert ctl - 2.0 * atl < 0
    assert ctl - LAMBDA * atl > 0


# --- the fixed point -------------------------------------------------------

def test_loop_converges():
    races = pd.read_csv("data/candidate_races.csv")
    hist = iterate(FULL, STATE, RATES, races, start_class="olympic")
    assert len(hist) <= 4
    assert hist.iloc[-1]["status"] == "FIXED POINT"
    # at the fixed point the macrocycle matches the selected A-race class
    assert hist.iloc[-1]["macrocycle_class"] == hist.iloc[-1]["a_race_class"]


def test_loop_is_a_fixed_point():
    """Starting from the converged class reproduces it in one iteration."""
    races = pd.read_csv("data/candidate_races.csv")
    hist = iterate(FULL, STATE, RATES, races, start_class="70.3")
    assert hist.iloc[0]["status"] == "FIXED POINT"

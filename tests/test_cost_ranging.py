"""Tests for objective-coefficient (cost) ranging."""

import pandas as pd
import pytest

from src.analysis.cost_ranging import (m1_coefficient_range, m2_value_ranges,
                                       _m2_selection)
from src.models.model2_calendar import race_value

from .test_model1 import PROFILE as M1_PROFILE
from .test_model1 import RATES, STATE
from .test_model2 import PROFILE as M2_PROFILE
from .test_model2 import RACES


def test_m1_range_brackets_baseline():
    r = m1_coefficient_range(M1_PROFILE, STATE, RATES, "k2", 0.5, 6.0)
    assert r["lower"] <= r["baseline"] <= r["upper"]
    assert r["lower"] < r["upper"]


@pytest.fixture(scope="module")
def m2_ranges():
    return m2_value_ranges(RACES, M2_PROFILE, span=300.0)


def test_m2_roles_match_selection(m2_ranges):
    selected = _m2_selection(RACES, M2_PROFILE)
    for _, row in m2_ranges.iterrows():
        is_selected = row["id"] in selected
        assert (row["role"].startswith("selected")) == is_selected


def test_m2_perturbation_directions(m2_ranges):
    """Selected races are tested downward, rejected ones upward."""
    for _, row in m2_ranges.iterrows():
        if pd.isna(row["allowed_change"]):
            continue
        if row["role"].startswith("selected"):
            assert row["allowed_change"] <= 1e-9
        else:
            assert row["allowed_change"] >= -1e-9


def test_m2_critical_value_actually_flips(m2_ranges):
    """Just past the critical change the optimal calendar must differ."""
    base = _m2_selection(RACES, M2_PROFILE)
    candidates = m2_ranges.dropna(subset=["allowed_change"])
    candidates = candidates[candidates["allowed_change"].abs() > 1e-6]
    assert not candidates.empty
    row = candidates.iloc[candidates["allowed_change"].abs().argmin()]
    idx = RACES.index[RACES["id"] == row["id"]][0]

    delta = row["allowed_change"]
    inside = _m2_selection(RACES, M2_PROFILE, {idx: delta * 0.9})
    outside = _m2_selection(RACES, M2_PROFILE, {idx: delta * 1.5})
    assert inside == base          # within the range: unchanged
    assert outside != base         # beyond it: the calendar changes

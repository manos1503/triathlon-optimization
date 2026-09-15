"""Objective-coefficient ranging (perturbation of the cost vector c).

Course reference: "Ανάλυση Ευαισθησίας — Διαταραχή σε αντικειμενικούς
συντελεστές", distinguishing

    A. coefficient of a BASIC variable      -> how far c_j may move before the
                                               current optimal basis changes
    B. coefficient of a NON-BASIC variable  -> how far c_j must improve before
                                               the variable enters the basis

This module computes both, on the two models where objective coefficients
carry meaning:

Model 1 (LP): the objective p_T = k1*CTL_T - k2*ATL_T has exactly two nonzero
    cost coefficients. Their stability intervals say how robust the optimal
    training plan is to the (hard-to-measure) Banister weights.

Model 2 (MIP): the cost coefficients are the race values v_i. Selected races
    play the role of basic variables (how far can v_i FALL before the race
    drops out of the optimal calendar?); rejected races play the role of
    non-basic ones (how far must v_i RISE before the race enters?). The
    critical values are found by bisection on a perturbation of v_i.

Usage:
    python -m src.analysis.cost_ranging
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pulp

from src.models.model1_training import build_model as m1_build
from src.models.model1_training import extract_solution as m1_extract
from src.models.model1_training import solve as m1_solve
from src.models.model2_calendar import build_model as m2_build
from src.models.model2_calendar import extract_solution as m2_extract
from src.models.model2_calendar import race_value
from src.models.model2_calendar import solve as m2_solve

from .common import TABLES, load_inputs

TOL = 1e-4          # plans closer than this are treated as identical
BISECT_STEPS = 24   # ~1e-5 relative precision on the critical value


# --------------------------------------------------------------------------
# Model 1 — ranging the Banister objective weights k1, k2
# --------------------------------------------------------------------------

def _m1_plan(profile: dict, state: dict, rates: pd.DataFrame) -> np.ndarray | None:
    prob, v = m1_build(profile, state, rates)
    if m1_solve(prob) != "Optimal":
        return None
    return m1_extract(prob, v, state)["plan"]["hours"].to_numpy()


def m1_coefficient_range(profile: dict, state: dict, rates: pd.DataFrame,
                         key: str, lo: float, hi: float) -> dict:
    """Interval of the objective weight ``key`` over which the optimal plan is
    unchanged. Found by bisecting outward from the baseline value."""
    base_val = profile["banister"][key]
    base_plan = _m1_plan(profile, state, rates)

    def unchanged(val: float) -> bool:
        p = copy.deepcopy(profile)
        p["banister"][key] = val
        plan = _m1_plan(p, state, rates)
        return plan is not None and np.abs(plan - base_plan).max() < TOL

    def edge(direction: int, limit: float) -> float:
        """Last value in `direction` for which the plan is unchanged."""
        if unchanged(limit):
            return limit
        good, bad = base_val, limit
        for _ in range(BISECT_STEPS):
            mid = 0.5 * (good + bad)
            if unchanged(mid):
                good = mid
            else:
                bad = mid
        return good

    return {"coefficient": key,
            "baseline": base_val,
            "lower": round(edge(-1, lo), 3),
            "upper": round(edge(+1, hi), 3)}


# --------------------------------------------------------------------------
# Model 2 — ranging the race values v_i (basic vs non-basic analogue)
# --------------------------------------------------------------------------

def _m2_selection(races: pd.DataFrame, profile: dict, delta: dict | None = None) -> set:
    """Set of selected race ids, optionally perturbing one race's value.

    The perturbation is applied through an explicit objective override so the
    model's own value function stays untouched.
    """
    prob, v = m2_build(races, profile)
    if delta:
        vals = race_value(races, profile["model2"]).copy()
        for idx, d in delta.items():
            vals[idx] += d
        prob.setObjective(pulp.lpSum(vals[i] * v["x"][i] for i in races.index))
    if m2_solve(prob) != "Optimal":
        return set()
    sol = m2_extract(races, v)
    return set(sol.loc[sol["selected"], "id"])


def m2_value_ranges(races: pd.DataFrame, profile: dict,
                    span: float = 400.0) -> pd.DataFrame:
    """For every race: how far its value may fall (if selected) or must rise
    (if rejected) before the optimal calendar changes."""
    base = _m2_selection(races, profile)
    values = race_value(races, profile["model2"])
    rows = []

    for i in races.index:
        rid = races.loc[i, "id"]
        selected = rid in base
        # selected -> perturb downward; rejected -> perturb upward
        sign = -1.0 if selected else +1.0

        if _m2_selection(races, profile, {i: sign * span}) == base:
            critical = None           # no flip within the scanned span
        else:
            good, bad = 0.0, sign * span
            for _ in range(BISECT_STEPS):
                mid = 0.5 * (good + bad)
                if _m2_selection(races, profile, {i: mid}) == base:
                    good = mid
                else:
                    bad = mid
            critical = good

        rows.append({
            "id": rid,
            "name": races.loc[i, "name"],
            "role": "selected (basic)" if selected else "rejected (non-basic)",
            "value": round(float(values[i]), 1),
            "allowed_change": None if critical is None else round(critical, 2),
            "critical_value": None if critical is None else round(float(values[i]) + critical, 1),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    profile, state, rates = load_inputs()
    races = pd.read_csv("data/candidate_races.csv")

    # --- Model 1: objective weights ---
    m1_rows = [
        m1_coefficient_range(profile, state, rates, "k1", 0.1, 6.0),
        m1_coefficient_range(profile, state, rates, "k2", 0.1, 8.0),
    ]
    m1 = pd.DataFrame(m1_rows)
    m1.to_csv(TABLES / "model1_cost_ranging.csv", index=False)
    print("Model 1 — objective-coefficient ranging (optimal plan unchanged):")
    print(m1.to_string(index=False))

    # --- Model 2: race values ---
    m2 = m2_value_ranges(races, profile)
    m2.to_csv(TABLES / "model2_cost_ranging.csv", index=False)
    print("\nModel 2 — race-value ranging (optimal calendar unchanged):")
    print(m2.to_string(index=False))

    tight = m2.dropna(subset=["allowed_change"]).reindex(
        m2.dropna(subset=["allowed_change"])["allowed_change"].abs().sort_values().index)
    if not tight.empty:
        r = tight.iloc[0]
        print(f"\n[cost_ranging] most fragile choice: {r['id']} ({r['name']}) - "
              f"a change of only {r['allowed_change']:+.2f} in its value flips the calendar")

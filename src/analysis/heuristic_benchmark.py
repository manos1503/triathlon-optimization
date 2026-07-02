"""Benchmark: optimal LP plan vs. a fixed 80/20 coaching-rule plan.

The heuristic follows common triathlon coaching practice with NO optimization:
    - fill the weekly hours ceiling every week (phase multipliers applied)
    - fixed discipline split: swim 20%, bike 45%, run 35%
    - fixed intensity split: easy 80%, moderate 15%, hard 5%
Its CTL/ATL trajectory follows from the Banister recursion, and its
performance p_T is compared to the LP optimum. Constraint violations
(e.g. the fatigue ceiling) are reported — a plan a coach might write can
be infeasible for THIS athlete's recovery capacity.

Usage:
    python -m src.analysis.heuristic_benchmark
"""

from __future__ import annotations

import math

import pandas as pd
import pulp

from src.models.model1_training import build_model, phase_of, solve

from .common import TABLES, load_inputs

DISCIPLINE_SPLIT = {"swim": 0.20, "bike": 0.45, "run": 0.35}
INTENSITY_SPLIT = {"easy": 0.80, "moderate": 0.15, "hard": 0.05}


def heuristic_trajectory(profile: dict, state: dict, rates: pd.DataFrame) -> pd.DataFrame:
    m1, mc, ban = profile["model1"], profile["macrocycle"], profile["banister"]
    r = {(row.sport, row.bucket): row.trimp_per_hour for row in rates.itertuples()}
    lam_c = math.exp(-7.0 / ban["tau_ctl_days"])
    lam_a = math.exp(-7.0 / ban["tau_atl_days"])

    rows, ctl, atl = [], state["ctl"], state["atl"]
    for w in range(1, mc["weeks"] + 1):
        hours = profile["constraints"]["weekly_hours_max"] * m1["hours_multiplier"][phase_of(w, mc["phases"])]
        load = sum(hours * DISCIPLINE_SPLIT[d] * INTENSITY_SPLIT[b] * r[d, b]
                   for d in DISCIPLINE_SPLIT for b in INTENSITY_SPLIT)
        ctl = lam_c * ctl + (1 - lam_c) * load / 7.0
        atl = lam_a * atl + (1 - lam_a) * load / 7.0
        rows.append({"week": w, "hours": hours, "load": load, "ctl": ctl, "atl": atl})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    profile, state, rates = load_inputs()
    ban, m1 = profile["banister"], profile["model1"]

    heur = heuristic_trajectory(profile, state, rates)
    p_heur = ban["k1"] * heur["ctl"].iloc[-1] - ban["k2"] * heur["atl"].iloc[-1]
    atl_violations = int((heur["atl"] > m1["atl_max"] + 1e-6).sum())

    prob, _ = build_model(profile, state, rates)
    assert solve(prob) == "Optimal"
    p_opt = pulp.value(prob.objective)

    summary = pd.DataFrame({
        "plan": ["LP optimal", "80/20 heuristic"],
        "performance_pT": [round(p_opt, 2), round(p_heur, 2)],
        "total_hours": [None, round(heur["hours"].sum(), 1)],
        "atl_ceiling_violations": [0, atl_violations],
        "race_day_ctl": [None, round(heur["ctl"].iloc[-1], 1)],
        "race_day_atl": [None, round(heur["atl"].iloc[-1], 1)],
    })
    heur.round(2).to_csv(TABLES / "model1_heuristic_trajectory.csv", index=False)
    summary.to_csv(TABLES / "model1_heuristic_benchmark.csv", index=False)

    print(heur.round(1).to_string(index=False))
    print(f"\n[heuristic] p_heuristic = {p_heur:.2f}  vs  p_optimal = {p_opt:.2f}  "
          f"(gap {p_opt - p_heur:+.2f})")
    print(f"[heuristic] ATL ceiling violated in {atl_violations}/16 weeks")

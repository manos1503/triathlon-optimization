"""Alternative optima: two visibly different plans with identical performance.

Method: solve Model 1 for p*, add the constraint p_T >= p* - eps, then
re-solve with two different secondary objectives (minimize vs. maximize
total training hours). If the two plans differ, the LP has multiple optima
— the optimal face is not a single vertex.

Usage:
    python -m src.analysis.alt_optima
"""

from __future__ import annotations

import pandas as pd
import pulp

from src.models.model1_training import BUCKETS, DISCIPLINES, build_model, extract_solution, solve

from .common import TABLES, load_inputs

EPS = 1e-4


def secondary_solve(profile, state, rates, p_star: float, sense: int) -> dict:
    """Re-solve with total hours as secondary objective, performance fixed at p*."""
    prob, v = build_model(profile, state, rates)
    T = v["T"]
    perf = prob.objective  # k1*CTL_T - k2*ATL_T expression
    prob += perf >= p_star - EPS, "fix_performance"
    total_hours = pulp.lpSum(
        v["h"][d][b][w] for d in DISCIPLINES for b in BUCKETS for w in range(1, T + 1)
    )
    prob.sense = sense
    prob.setObjective(total_hours)
    assert solve(prob) == "Optimal"
    sol = extract_solution(prob, v, state)
    sol["performance"] = pulp.value(perf)
    sol["total_hours"] = pulp.value(total_hours)
    return sol


if __name__ == "__main__":
    profile, state, rates = load_inputs()

    prob, v = build_model(profile, state, rates)
    assert solve(prob) == "Optimal"
    p_star = pulp.value(prob.objective)

    lean = secondary_solve(profile, state, rates, p_star, pulp.LpMinimize)
    bulky = secondary_solve(profile, state, rates, p_star, pulp.LpMaximize)

    cmp = pd.DataFrame({
        "plan": ["min_hours", "max_hours"],
        "performance_pT": [round(lean["performance"], 3), round(bulky["performance"], 3)],
        "total_hours": [round(lean["total_hours"], 1), round(bulky["total_hours"], 1)],
    })
    cmp.to_csv(TABLES / "model1_alt_optima_summary.csv", index=False)
    lean["plan"].round(2).to_csv(TABLES / "model1_plan_min_hours.csv", index=False)
    bulky["plan"].round(2).to_csv(TABLES / "model1_plan_max_hours.csv", index=False)

    print(f"[alt_optima] p* = {p_star:.3f}")
    print(cmp.to_string(index=False))
    gap = bulky["total_hours"] - lean["total_hours"]
    print(f"[alt_optima] same performance achievable with a {gap:.1f}h season-hour spread "
          f"-> optimal face has dimension >= 1 (degenerate alternative optima)")

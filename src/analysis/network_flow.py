"""Network-flow extension: relaxed Model 2 as a longest path in a DAG.

Drop the two knapsack constraints and the monthly cap from Model 2, keeping
only recovery spacing (and the readiness pre-filter). Feasible race
selections are then exactly the PATHS of a DAG:

    nodes: race-ready races, ordered by week (+ virtual source S, sink T)
    arc i -> j: week_j - week_i >= recovery(class_i) + 1
    node weight: race value v_i

Recovery chains compose (if i->j and j->k are feasible, so is i->k), so any
path through the DAG is a feasible calendar and vice versa. Maximizing
selected value = node-weighted longest path.

Three solution methods are compared and must agree:
    1. dynamic programming in topological (week) order,   O(n^2)
    2. the path LP: a unit of flow from S to T on arc variables —
       the constraint matrix is a network matrix, hence TOTALLY UNIMODULAR,
       so the plain LP (no integrality constraints!) has an integral optimum,
    3. the restricted 0-1 MIP (Model 2 with the side constraints disabled).

This is the course's network-flow topic meeting integer programming: the
knapsack constraints are what break the network structure and force B&B.

Usage:
    python -m src.analysis.network_flow
"""

from __future__ import annotations

import copy

import pandas as pd
import pulp

from src.models.model2_calendar import build_model, extract_solution, race_value, readiness, solve

from .common import TABLES, load_inputs


def feasible_arc(races: pd.DataFrame, rec: dict, i, j) -> bool:
    gap = int(races.loc[j, "week"]) - int(races.loc[i, "week"])
    return gap >= rec[str(races.loc[i, "distance_class"])] + 1


def longest_path_dp(races: pd.DataFrame, values: pd.Series, rec: dict) -> tuple[float, list]:
    """Node-weighted longest path via DP in week order."""
    order = races.sort_values("week").index.to_list()
    best = {i: float(values[i]) for i in order}      # best value of a path ending at i
    pred = {i: None for i in order}
    for pos, j in enumerate(order):
        for i in order[:pos]:
            if feasible_arc(races, rec, i, j) and best[i] + values[j] > best[j]:
                best[j] = best[i] + float(values[j])
                pred[j] = i
    end = max(best, key=best.get)
    path, node = [], end
    while node is not None:
        path.append(races.loc[node, "id"])
        node = pred[node]
    return best[end], list(reversed(path))


def longest_path_lp(races: pd.DataFrame, values: pd.Series, rec: dict) -> tuple[float, dict]:
    """Path LP: continuous flow variables; integrality comes free (TU matrix)."""
    idx = races.sort_values("week").index.to_list()
    arcs = [("S", j) for j in idx] + [(i, "T") for i in idx]
    arcs += [(i, j) for pi, i in enumerate(idx) for j in idx[pi + 1:]
             if feasible_arc(races, rec, i, j)]

    prob = pulp.LpProblem("longest_path_lp", pulp.LpMaximize)
    f = {a: pulp.LpVariable(f"f_{a[0]}_{a[1]}", lowBound=0) for a in arcs}   # continuous!

    prob += pulp.lpSum(values[j] * f[a] for a in arcs for j in [a[1]] if j != "T")
    prob += pulp.lpSum(f[a] for a in arcs if a[0] == "S") == 1, "source"
    for n in idx:
        prob += (pulp.lpSum(f[a] for a in arcs if a[1] == n)
                 == pulp.lpSum(f[a] for a in arcs if a[0] == n)), f"balance_{n}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    flows = {a: f[a].varValue for a in arcs if (f[a].varValue or 0) > 1e-6}
    return pulp.value(prob.objective), flows


def restricted_mip(races: pd.DataFrame, profile: dict) -> tuple[float, list]:
    p = copy.deepcopy(profile)
    p["model2"]["budget_eur"] = 10 ** 9
    p["model2"]["vacation_days_budget"] = 10 ** 9
    p["model2"]["max_races_per_month"] = 99
    p["model2"]["require_a_race"] = False
    prob, v = build_model(races, p)
    assert solve(prob) == "Optimal"
    sel = extract_solution(races, v)
    return pulp.value(prob.objective), sel.loc[sel["selected"], "id"].to_list()


if __name__ == "__main__":
    profile, _, _ = load_inputs()
    races = pd.read_csv("data/candidate_races.csv")

    ready = readiness(races, profile)
    ready_races = races[ready].copy()
    values = race_value(ready_races, profile["model2"])

    rec = profile["model2"]["recovery_weeks"]
    z_dp, path = longest_path_dp(ready_races, values, rec)
    z_lp, flows = longest_path_lp(ready_races, values, rec)
    z_mip, mip_sel = restricted_mip(races, profile)

    frac = [a for a, val in flows.items() if 1e-6 < val < 1 - 1e-6]
    summary = pd.DataFrame([{
        "z_dp_longest_path": round(z_dp, 2),
        "z_lp_flow": round(z_lp, 2),
        "z_restricted_mip": round(z_mip, 2),
        "lp_solution_integral": len(frac) == 0,
        "n_races_selected": len(path),
    }])
    summary.to_csv(TABLES / "model2_network_flow.csv", index=False)

    print(f"[network_flow] DP longest path  z = {z_dp:.2f}  path: {' -> '.join(path)}")
    print(f"[network_flow] flow LP          z = {z_lp:.2f}  "
          f"({'integral' if not frac else 'FRACTIONAL'} without integrality constraints)")
    print(f"[network_flow] restricted MIP   z = {z_mip:.2f}  selected: {mip_sel}")
    assert abs(z_dp - z_lp) < 1e-4 and abs(z_dp - z_mip) < 1e-4, "methods disagree!"
    print("[network_flow] all three methods agree (total unimodularity confirmed empirically)")

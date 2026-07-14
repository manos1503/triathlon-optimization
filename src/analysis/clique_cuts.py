"""Valid inequalities: clique cuts strengthen Model 2's LP relaxation.

The base formulation encodes recovery conflicts pairwise, x_i + x_j <= 1.
If a set C of races is PAIRWISE conflicting, the clique inequality

    sum_{i in C} x_i <= 1

is valid for the integer polytope and strictly stronger than its pairwise
pieces in the LP relaxation (three variables at 1/2 satisfy all pairwise
constraints but violate the clique cut). This module enumerates maximal
cliques of the conflict graph (Bron--Kerbosch; the graph has 17 nodes),
adds cliques of size >= 3 to the relaxation, and measures how much the
integrality gap closes --- a cutting-planes mini-study.

Usage:
    python -m src.analysis.clique_cuts
"""

from __future__ import annotations

import pandas as pd
import pulp

from src.models.model2_calendar import build_model, solve

from .common import TABLES, load_inputs


def conflict_graph(races: pd.DataFrame, rec: dict) -> dict[int, set]:
    """Adjacency: i ~ j iff the pair violates recovery spacing (as in the MIP)."""
    adj = {i: set() for i in races.index}
    for i in races.index:
        for j in races.index:
            if i >= j:
                continue
            wi, wj = int(races.loc[i, "week"]), int(races.loc[j, "week"])
            earlier = i if wi <= wj else j
            need = rec[str(races.loc[earlier, "distance_class"])]
            if abs(wj - wi) < need + 1:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def maximal_cliques(adj: dict[int, set]) -> list[set]:
    """Bron--Kerbosch with pivoting (fine for 17 nodes)."""
    cliques: list[set] = []

    def bk(r: set, p: set, x: set):
        if not p and not x:
            cliques.append(r)
            return
        pivot = max(p | x, key=lambda v: len(adj[v] & p))
        for v in list(p - adj[pivot]):
            bk(r | {v}, p & adj[v], x & adj[v])
            p.remove(v)
            x.add(v)

    bk(set(), set(adj.keys()), set())
    return cliques


def relaxation_value(races: pd.DataFrame, profile: dict,
                     cliques: list[set] | None = None) -> float:
    prob, v = build_model(races, profile, relax=True)
    if cliques:
        for k, c in enumerate(cliques):
            prob += pulp.lpSum(v["x"][i] for i in c) <= 1, f"clique_{k}"
    assert solve(prob) == "Optimal"
    return pulp.value(prob.objective)


if __name__ == "__main__":
    profile, _, _ = load_inputs()
    races = pd.read_csv("data/candidate_races.csv")
    rec = profile["model2"]["recovery_weeks"]

    adj = conflict_graph(races, rec)
    cliques = [c for c in maximal_cliques(adj) if len(c) >= 3]

    prob_mip, _ = build_model(races, profile)
    assert solve(prob_mip) == "Optimal"
    z_mip = pulp.value(prob_mip.objective)

    z_lp = relaxation_value(races, profile)
    z_lp_cuts = relaxation_value(races, profile, cliques)

    summary = pd.DataFrame([{
        "z_mip": round(z_mip, 1),
        "z_lp_pairwise": round(z_lp, 1),
        "z_lp_with_clique_cuts": round(z_lp_cuts, 1),
        "gap_pairwise_pct": round(100 * (z_lp - z_mip) / z_mip, 2),
        "gap_with_cuts_pct": round(100 * (z_lp_cuts - z_mip) / z_mip, 2),
        "n_clique_cuts": len(cliques),
        "max_clique_size": max((len(c) for c in cliques), default=0),
    }])
    summary.to_csv(TABLES / "model2_clique_cuts.csv", index=False)

    print(f"[clique_cuts] {len(cliques)} maximal cliques of size >= 3 "
          f"(largest: {max((len(c) for c in cliques), default=0)} races)")
    for c in cliques:
        print("   ", " + ".join(sorted(races.loc[i, 'id'] for i in c)), "<= 1")
    print(f"[clique_cuts] z_MIP = {z_mip:.1f}")
    print(f"[clique_cuts] LP relaxation: pairwise {z_lp:.1f} "
          f"(gap {100*(z_lp-z_mip)/z_mip:.2f}%) -> with cuts {z_lp_cuts:.1f} "
          f"(gap {100*(z_lp_cuts-z_mip)/z_mip:.2f}%)")

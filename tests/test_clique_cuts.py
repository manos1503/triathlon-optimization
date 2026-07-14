"""Tests for the clique-cut strengthening of Model 2's LP relaxation."""

import pulp

from src.analysis.clique_cuts import conflict_graph, maximal_cliques, relaxation_value
from src.models.model2_calendar import build_model, solve

from .test_model2 import PROFILE, RACES


def test_cliques_are_pairwise_conflicting():
    adj = conflict_graph(RACES, PROFILE["model2"]["recovery_weeks"])
    for clique in maximal_cliques(adj):
        for i in clique:
            for j in clique:
                if i != j:
                    assert j in adj[i]


def test_cuts_tighten_but_stay_valid():
    adj = conflict_graph(RACES, PROFILE["model2"]["recovery_weeks"])
    cliques = [c for c in maximal_cliques(adj) if len(c) >= 3]

    prob_mip, _ = build_model(RACES, PROFILE)
    assert solve(prob_mip) == "Optimal"
    z_mip = pulp.value(prob_mip.objective)

    z_lp = relaxation_value(RACES, PROFILE)
    z_cuts = relaxation_value(RACES, PROFILE, cliques)

    # cuts can only tighten the relaxation, and never cut off the integer optimum
    assert z_cuts <= z_lp + 1e-6
    assert z_cuts >= z_mip - 1e-6

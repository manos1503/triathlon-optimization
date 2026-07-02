"""Model 2 — Optimal race-calendar selection (0-1 MIP).

Knapsack core (budget) plus temporal constraints:

    max  sum_i v_i x_i
    s.t. sum_i cost_i x_i <= budget
         races per calendar month <= M
         recovery spacing: x_i + x_j <= 1 for race pairs closer than the
             recovery requirement of the earlier race (same-week pairs
             are excluded by the same constraint)
         readiness pre-filter: x_i = 0 if projected CTL at race week is
             below the class requirement (Model 1 -> Model 2 link)
         x_i in {0, 1}

Also solves the LP relaxation (0 <= x <= 1) to demonstrate the
integrality gap closed by Branch & Bound.
"""

from __future__ import annotations

import math

import pandas as pd
import pulp


def race_value(races: pd.DataFrame, weights: dict) -> pd.Series:
    return (weights["points"] * races["points"]
            + weights["strategic"] * races["strategic"]
            + weights["preference"] * races["preference"])


def project_season_ctl(profile: dict, n_weeks: int = 45) -> list[float]:
    """Projected CTL per season week: off-season start, load ramps at
    ramp_max toward Model 1's sustained build load (sequential pipeline link)."""
    m2, ban = profile["model2"], profile["banister"]
    lam_c = math.exp(-7.0 / ban["tau_ctl_days"])
    ramp = 1.0 + profile["model1"]["ramp_max"]

    ctl, load = m2["offseason_ctl"], m2["offseason_weekly_load"]
    out = []
    for _ in range(1, n_weeks + 1):
        load = min(m2["sustained_weekly_load"], load * ramp)
        ctl = lam_c * ctl + (1 - lam_c) * load / 7.0
        out.append(ctl)
    return out


def readiness(races: pd.DataFrame, profile: dict) -> pd.Series:
    """True if the athlete can be fit in time for the race."""
    ctl_proj = project_season_ctl(profile)
    req = profile["model2"]["ctl_required"]
    return pd.Series(
        [ctl_proj[int(row.week) - 1] >= req[str(row.distance_class)]
         for row in races.itertuples()],
        index=races.index,
    )


def build_model(races: pd.DataFrame, profile: dict,
                relax: bool = False) -> tuple[pulp.LpProblem, dict]:
    m2 = profile["model2"]
    v = race_value(races, m2["value_weights"])
    cost = races["entry_fee_eur"] + races["travel_cost_eur"]
    ready = readiness(races, profile)
    rec = m2["recovery_weeks"]

    cat = "Continuous" if relax else "Binary"
    x = {i: pulp.LpVariable(f"x_{races.loc[i, 'id']}", 0, 1, cat) for i in races.index}

    prob = pulp.LpProblem("model2_race_calendar", pulp.LpMaximize)
    prob += pulp.lpSum(v[i] * x[i] for i in races.index), "season_value"

    prob += pulp.lpSum(cost[i] * x[i] for i in races.index) <= m2["budget_eur"], "budget"

    months = pd.to_datetime(races["date"]).dt.month
    for mo in sorted(months.unique()):
        idx = races.index[months == mo]
        if len(idx) > 1:
            prob += pulp.lpSum(x[i] for i in idx) <= m2["max_races_per_month"], f"month_{mo}"

    # recovery spacing (also excludes same-week clashes)
    for i in races.index:
        for j in races.index:
            if i >= j:
                continue
            wi, wj = races.loc[i, "week"], races.loc[j, "week"]
            earlier, later = (i, j) if wi <= wj else (j, i)
            gap = abs(int(wj) - int(wi))
            need = rec[str(races.loc[earlier, "distance_class"])]
            if gap < need + 1:
                prob += x[i] + x[j] <= 1, f"recovery_{races.loc[i,'id']}_{races.loc[j,'id']}"

    # fitness-readiness pre-filter (Model 1 link)
    for i in races.index:
        if not ready[i]:
            prob += x[i] == 0, f"not_ready_{races.loc[i, 'id']}"

    return prob, {"x": x, "value": v, "cost": cost, "ready": ready}


def solve(prob: pulp.LpProblem) -> str:
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.LpStatus[prob.status]


def extract_solution(races: pd.DataFrame, v: dict) -> pd.DataFrame:
    out = races.copy()
    out["value"] = v["value"]
    out["cost"] = v["cost"]
    out["ready"] = v["ready"]
    out["x"] = [round(v["x"][i].varValue, 4) for i in races.index]
    out["selected"] = out["x"] > 0.99
    return out

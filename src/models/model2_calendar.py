"""Model 2 — Optimal race-calendar selection (0-1 MIP).

Two-dimensional knapsack core (money + vacation days) plus temporal constraints:

    max  sum_i v_i x_i          v_i = weather_prob_i * (weighted score)
    s.t. sum_i cost_i x_i <= budget
         sum_i days_i x_i <= vacation_days_budget
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


# leg distances (km) per class, for elevation-gain normalization
BIKE_KM = {"sprint": 20.0, "olympic": 40.0, "70.3": 90.0}
RUN_KM = {"sprint": 5.0, "olympic": 10.0, "70.3": 21.1}
GPK_CAP = 15.0  # m/km treated as maximally hilly


def preference_score(races: pd.DataFrame, pref_weights: dict) -> pd.Series:
    """Composite enjoyment from decomposed subjective scores."""
    return sum(w * races[col] for col, w in pref_weights.items())


def _terrain_score(gain_per_km: pd.Series, preference: str) -> pd.Series:
    flat = 10.0 * (1.0 - gain_per_km.clip(upper=GPK_CAP) / GPK_CAP)
    if preference == "flat":
        return flat
    if preference == "hilly":
        return 10.0 - flat
    return pd.Series(7.0, index=gain_per_km.index)  # 'any'


def suitability_score(races: pd.DataFrame, terrain_pref: dict) -> pd.Series:
    """Athlete-course fit (0-10) from OBJECTIVE elevation data per leg."""
    bike_km = races["distance_class"].map(BIKE_KM)
    run_km = races["distance_class"].map(RUN_KM)
    bike = _terrain_score(races["bike_elev_gain_m"] / bike_km, terrain_pref["bike"])
    run = _terrain_score(races["run_elev_gain_m"] / run_km, terrain_pref["run"])
    return 0.5 * bike + 0.5 * run


def race_value(races: pd.DataFrame, m2: dict) -> pd.Series:
    weights = m2["value_weights"]
    enjoyment = preference_score(races, m2["preference_weights"])
    fit = suitability_score(races, m2["terrain_preference"])
    raw = (weights["points"] * races["points"]
           + weights["strategic"] * races["strategic"]
           + weights["preference"] * enjoyment
           + weights["fit"] * fit)
    if m2.get("weather_risk_adjust", False):
        return races["weather_prob"] * raw   # expected value: a ruined race delivers nothing
    return raw


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
    v = race_value(races, m2)
    cost = races["entry_fee_eur"] + races["travel_cost_eur"]
    ready = readiness(races, profile)
    rec = m2["recovery_weeks"]

    cat = "Continuous" if relax else "Binary"
    x = {i: pulp.LpVariable(f"x_{races.loc[i, 'id']}", 0, 1, cat) for i in races.index}

    prob = pulp.LpProblem("model2_race_calendar", pulp.LpMaximize)
    prob += pulp.lpSum(v[i] * x[i] for i in races.index), "season_value"

    prob += pulp.lpSum(cost[i] * x[i] for i in races.index) <= m2["budget_eur"], "budget"

    # second knapsack dimension: vacation days off work
    if "vacation_days_budget" in m2:
        prob += (pulp.lpSum(races.loc[i, "vacation_days"] * x[i] for i in races.index)
                 <= m2["vacation_days_budget"]), "vacation_days"

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

    # at least one championship-quality target (set-covering side constraint)
    if m2.get("require_a_race", False) and "a_race" in races.columns:
        a_idx = races.index[races["a_race"] == 1]
        if len(a_idx):
            prob += pulp.lpSum(x[i] for i in a_idx) >= 1, "a_race_requirement"

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

"""Model 1 — Optimal weekly training-load allocation (LP).

Decision variables
    h[d, b, w] >= 0   hours of discipline d at intensity bucket b in week w
                      (3 x 3 x 16 = 144 variables)

Auxiliary variables (tied by equality constraints, so the model stays LP)
    L[w]      weekly TRIMP load
    CTL[w]    chronic training load (fitness), daily-scale units
    ATL[w]    acute training load (fatigue), daily-scale units

Banister dynamics (weekly step of the daily exponential smoothing; weekly
TRIMP is converted to average daily load L/7 so CTL/ATL stay comparable
to the data pipeline's daily values):

    CTL_w = lam_c * CTL_{w-1} + (1 - lam_c) * L_w / 7      lam_c = exp(-7/42)
    ATL_w = lam_a * ATL_{w-1} + (1 - lam_a) * L_w / 7      lam_a = exp(-7/7)

Objective: maximize predicted race-day performance
    p_T = k1 * CTL_T - k2 * ATL_T
"""

from __future__ import annotations

import math

import pandas as pd
import pulp

DISCIPLINES = ["swim", "bike", "run"]
BUCKETS = ["easy", "moderate", "hard"]


def phase_of(week: int, phases: dict) -> str:
    for name, spec in phases.items():
        lo, hi = spec["weeks"]
        if lo <= week <= hi:
            return name
    raise ValueError(f"week {week} not in any phase")


def build_model(profile: dict, state: dict, rates: pd.DataFrame) -> tuple[pulp.LpProblem, dict]:
    """Construct the LP. Returns (problem, variables-dict)."""
    m1 = profile["model1"]
    mc = profile["macrocycle"]
    ban = profile["banister"]
    T = mc["weeks"]
    weeks = range(1, T + 1)

    r = {(row.sport, row.bucket): row.trimp_per_hour for row in rates.itertuples()}
    lam_c = math.exp(-7.0 / ban["tau_ctl_days"])
    lam_a = math.exp(-7.0 / ban["tau_atl_days"])
    h_min = profile["constraints"]["weekly_hours_min_per_discipline"]
    h_ceiling = profile["constraints"]["weekly_hours_max"]
    taper_frac = {int(k): v for k, v in m1["taper_load_fraction"].items()}

    prob = pulp.LpProblem("model1_training_load", pulp.LpMaximize)

    h = pulp.LpVariable.dicts("h", (DISCIPLINES, BUCKETS, weeks), lowBound=0)
    L = pulp.LpVariable.dicts("L", weeks, lowBound=0)
    CTL = pulp.LpVariable.dicts("CTL", weeks, lowBound=0)
    ATL = pulp.LpVariable.dicts("ATL", weeks, lowBound=0)

    # --- dynamics ---
    for w in weeks:
        prob += L[w] == pulp.lpSum(r[d, b] * h[d][b][w] for d in DISCIPLINES for b in BUCKETS), f"def_load_w{w}"
        prev_c = CTL[w - 1] if w > 1 else state["ctl"]
        prev_a = ATL[w - 1] if w > 1 else state["atl"]
        prob += CTL[w] == lam_c * prev_c + (1 - lam_c) * L[w] / 7.0, f"def_ctl_w{w}"
        prob += ATL[w] == lam_a * prev_a + (1 - lam_a) * L[w] / 7.0, f"def_atl_w{w}"

    # --- constraints ---
    for w in weeks:
        phase = phase_of(w, mc["phases"])
        h_max = h_ceiling * m1["hours_multiplier"][phase]
        total = pulp.lpSum(h[d][b][w] for d in DISCIPLINES for b in BUCKETS)

        prob += total <= h_max, f"hours_cap_w{w}"

        for d in DISCIPLINES:
            prob += pulp.lpSum(h[d][b][w] for b in BUCKETS) >= h_min[d], f"min_{d}_w{w}"

        prob += (
            pulp.lpSum(h[d][b][w] for d in DISCIPLINES for b in ("moderate", "hard"))
            <= m1["polarization_max"] * total
        ), f"polarization_w{w}"

        prob += ATL[w] <= m1["atl_max"], f"atl_cap_w{w}"

        if w >= 2:
            prob += L[w] <= (1 + m1["ramp_max"]) * L[w - 1], f"ramp_w{w}"

    # taper: cap vs. average build-phase load
    build_weeks = range(mc["phases"]["build"]["weeks"][0], mc["phases"]["build"]["weeks"][1] + 1)
    avg_build = pulp.lpSum(L[w] for w in build_weeks) / len(list(build_weeks))
    for w, frac in taper_frac.items():
        prob += L[w] <= frac * avg_build, f"taper_w{w}"

    # --- objective: race-day performance ---
    prob += ban["k1"] * CTL[T] - ban["k2"] * ATL[T], "race_day_performance"

    return prob, {"h": h, "L": L, "CTL": CTL, "ATL": ATL, "T": T}


def solve(prob: pulp.LpProblem) -> str:
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.LpStatus[prob.status]


def extract_solution(prob: pulp.LpProblem, v: dict, state: dict) -> dict:
    """Weekly plan and trajectories as DataFrames, plus duals of named constraints."""
    T = v["T"]
    weeks = range(1, T + 1)

    plan = pd.DataFrame(
        [
            {"week": w, "discipline": d, "bucket": b, "hours": v["h"][d][b][w].varValue}
            for w in weeks for d in DISCIPLINES for b in BUCKETS
        ]
    )

    ctl = {w: v["CTL"][w].varValue for w in weeks}
    atl = {w: v["ATL"][w].varValue for w in weeks}
    traj = pd.DataFrame(
        [
            {
                "week": w,
                "load": v["L"][w].varValue,
                "hours": plan.loc[plan.week == w, "hours"].sum(),
                "ctl": ctl[w],
                "atl": atl[w],
                "tsb": (ctl[w - 1] if w > 1 else state["ctl"]) - (atl[w - 1] if w > 1 else state["atl"]),
            }
            for w in weeks
        ]
    )

    duals = {name: c.pi for name, c in prob.constraints.items() if c.pi is not None}
    return {
        "objective": pulp.value(prob.objective),
        "plan": plan,
        "trajectory": traj,
        "duals": duals,
    }

"""Model 3 — Optimal race pacing strategy (LP with zone linearization).

Decision variables
    t[l, z] >= 0   minutes spent in intensity zone z during leg l
                   (3 legs x 5 zones = 15 variables)

The nonlinear speed-energy relationship is linearized by discretizing
intensity into zones: within a zone, speed (km/min) and metabolic energy
rate (kJ/min) are constants, so distance and energy are linear in t.

    min  sum_lz t_lz + T1 + T2                       (total race time)
    s.t. sum_z s_swim,z t_swim,z >= d_swim           (distance, swim)
         sum_z s_bike,z t_bike,z >= d_bike           (distance, bike)
         sum_z s_run,z  t_run,z - phi*G >= d_run     (distance, run - coupling)
              G = sum_{z in Z4,Z5} e_bike,z t_bike,z (hard bike work)
         sum_lz e_lz t_lz <= E_tot                   (energy budget, from Model 1)
         sum_{z in Z4,Z5} t_lz <= rho_l sum_z t_lz   (per-leg intensity caps)

Duals: the energy constraint's shadow price is the marginal race-time value
of fitness (min/kJ); distance duals give the marginal cost of each km.
"""

from __future__ import annotations

import pandas as pd
import pulp

LEGS = ["swim", "bike", "run"]
ZONES = ["Z1", "Z2", "Z3", "Z4", "Z5"]
HARD = ["Z4", "Z5"]


def _pace_to_kmh(pace: str, meters: float) -> float:
    """'1:40' per `meters` -> km/h."""
    m, s = pace.split(":")
    return (meters / 1000.0) / ((int(m) * 60 + int(s)) / 3600.0)


def zone_parameters(profile: dict, ftp_watts: float | None = None,
                    gradient_m_per_km: float = 0.0) -> pd.DataFrame:
    """Speed (km/min) and metabolic energy rate (kJ/min) per (leg, zone).

    ``gradient_m_per_km``: average climbing of the bike course. The reference
    speed at FTP describes a FLAT time-trial course; on a hilly course the same
    power yields less speed, so the reference is scaled by (1 - c * gpk) with
    c = ``bike_gradient_penalty`` from the profile. c = 0 (default) reproduces
    the original flat-course model exactly. Calibrated jointly with k_E on the
    athlete's three real races: c = 0.014 per m/km.
    """
    m3 = profile["model3"]
    thr = profile["thresholds"]
    ftp = ftp_watts if ftp_watts is not None else thr["bike_ftp_watts"]

    v_swim_thr = _pace_to_kmh(m3["swim_threshold_pace_per_100m"], 100) / 60.0   # km/min
    v_run_thr = _pace_to_kmh(thr["run_threshold_pace"], 1000) / 60.0
    c_grad = m3.get("bike_gradient_penalty", 0.0)
    grade_factor = max(0.25, 1.0 - c_grad * gradient_m_per_km)
    v_bike_ftp_ref = m3["bike_speed_at_ftp_kmh"] * grade_factor / 60.0
    ftp_ref = thr["bike_ftp_watts"]

    rows = []
    for z in ZONES:
        cost = m3["intensity_cost_factor"][z]

        v = v_swim_thr * m3["swim_speed_fraction"][z]
        rows.append(("swim", z, v, m3["swim_energy_cost_kj_per_km"] * cost * v))

        p = ftp * m3["power_fraction"][z]
        # speed scales with cube root of power (aero drag); reference 35 km/h at profile FTP
        v = v_bike_ftp_ref * (p / ftp_ref) ** (1.0 / 3.0)
        e = p / m3["bike_metabolic_efficiency"] * 60.0 / 1000.0                 # kJ/min
        rows.append(("bike", z, v, e))

        v = v_run_thr * m3["run_speed_fraction"][z]
        rows.append(("run", z, v, m3["run_energy_cost_kj_per_km"] * cost * v))

    return pd.DataFrame(rows, columns=["leg", "zone", "speed_km_min", "energy_kj_min"])


def energy_budget(profile: dict, ctl: float, atl: float = 0.0,
                  lam: float | None = None) -> float:
    """Race-day energy budget.

    Fitness-only (default, lam = 0):   E = k_E * CTL
    Form-based (lam > 0):              E = E_base + k_E * (CTL - lam * ATL)

    The form-based version is what makes the pipeline coherent: it is an
    increasing affine function of Model 1's objective p_T = k1*CTL - k2*ATL
    exactly when lam = k2/k1, so maximizing p_T is equivalent to maximizing
    race-day energy, hence to minimizing finishing time (Model 3's objective
    is non-increasing in E). See src/analysis/coherence.py.
    """
    m3 = profile["model3"]
    lam = m3.get("fatigue_lambda", 0.0) if lam is None else lam
    k_e = m3["energy_budget_kj_per_ctl"]
    if lam == 0.0:
        return k_e * ctl
    return m3.get("energy_base_kj", 0.0) + k_e * (ctl - lam * atl)


def build_model(profile: dict, ctl_race_day: float,
                ftp_watts: float | None = None,
                fueling_kj_min: float | None = None,
                fueling_decision: bool = False,
                atl_race_day: float = 0.0,
                fatigue_lambda: float | None = None,
                gradient_m_per_km: float = 0.0) -> tuple[pulp.LpProblem, dict]:
    """In-race carbohydrate intake, two modes.

    Fixed rate (``fueling_kj_min`` = r): intake at r kJ/min reduces every
    zone's NET energy drain, keeping the model an LP:
    sum (e_lz - r) t_lz <= E_tot. r ~ 21 kJ/min corresponds to the standard
    ~75 g carbohydrate/hour guideline.

    Decision mode (``fueling_decision=True``): per-leg intake F_l >= 0 (kJ)
    becomes a variable, bounded by gut absorption per leg,
    F_l <= rbar_l * (leg time)  --- linear in both F and t. The energy budget
    becomes  sum e_lz t_lz - sum_l F_l <= E_tot. The duals of the absorption
    constraints price 'gut training': minutes saved per extra kJ/min the
    athlete could absorb in that leg. Essential for long-course races
    (a 5-6 h race is not ridden on stored glycogen alone)."""
    m3 = profile["model3"]
    r = fueling_kj_min if fueling_kj_min is not None else m3.get("in_race_fueling_kj_per_min", 0.0)
    zp = zone_parameters(profile, ftp_watts, gradient_m_per_km)
    s = {(r.leg, r.zone): r.speed_km_min for r in zp.itertuples()}
    e = {(r.leg, r.zone): r.energy_kj_min for r in zp.itertuples()}
    d = m3["distances_km"]
    e_tot = energy_budget(profile, ctl_race_day, atl_race_day, fatigue_lambda)
    phi = m3["bike_run_coupling_km_per_kj"]

    prob = pulp.LpProblem("model3_pacing", pulp.LpMinimize)
    t = pulp.LpVariable.dicts("t", (LEGS, ZONES), lowBound=0)

    prob += (pulp.lpSum(t[l][z] for l in LEGS for z in ZONES)
             + m3["transitions_min"]["t1"] + m3["transitions_min"]["t2"]), "total_time"

    hard_bike_energy = pulp.lpSum(e["bike", z] * t["bike"][z] for z in HARD)

    prob += pulp.lpSum(s["swim", z] * t["swim"][z] for z in ZONES) >= d["swim"], "dist_swim"
    prob += pulp.lpSum(s["bike", z] * t["bike"][z] for z in ZONES) >= d["bike"], "dist_bike"
    prob += (pulp.lpSum(s["run", z] * t["run"][z] for z in ZONES)
             - phi * hard_bike_energy >= d["run"]), "dist_run"

    if fueling_decision:
        rbar = m3["fueling_max_kj_per_min"]
        F = pulp.LpVariable.dicts("F", LEGS, lowBound=0)   # kJ ingested per leg
        prob += (pulp.lpSum(e[l, z] * t[l][z] for l in LEGS for z in ZONES)
                 - pulp.lpSum(F[l] for l in LEGS) <= e_tot), "energy_budget"
        for l in LEGS:
            prob += F[l] <= rbar[l] * pulp.lpSum(t[l][z] for z in ZONES), f"absorption_{l}"
    else:
        F = None
        prob += (pulp.lpSum((e[l, z] - r) * t[l][z] for l in LEGS for z in ZONES)
                 <= e_tot), "energy_budget"

    for l in LEGS:
        prob += (pulp.lpSum(t[l][z] for z in HARD)
                 <= m3["max_hard_fraction"][l] * pulp.lpSum(t[l][z] for z in ZONES)), f"hard_cap_{l}"

    return prob, {"t": t, "F": F, "zp": zp, "e_tot": e_tot}


def solve(prob: pulp.LpProblem) -> str:
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.LpStatus[prob.status]


def extract_solution(prob: pulp.LpProblem, v: dict) -> dict:
    rows = []
    for l in LEGS:
        for z in ZONES:
            minutes = v["t"][l][z].varValue or 0.0
            if minutes > 1e-6:
                zp = v["zp"]
                spd = zp.loc[(zp.leg == l) & (zp.zone == z)].iloc[0]
                rows.append({"leg": l, "zone": z, "minutes": minutes,
                             "km": minutes * spd.speed_km_min,
                             "kj": minutes * spd.energy_kj_min})
    plan = pd.DataFrame(rows)
    leg_times = plan.groupby("leg")["minutes"].sum().reindex(LEGS)
    duals = {name: c.pi for name, c in prob.constraints.items() if c.pi is not None}
    return {
        "total_time_min": pulp.value(prob.objective),
        "plan": plan,
        "leg_times": leg_times,
        "energy_used_kj": plan["kj"].sum(),
        "e_tot": v["e_tot"],
        "duals": duals,
    }

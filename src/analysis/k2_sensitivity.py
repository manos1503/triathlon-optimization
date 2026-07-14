"""Sensitivity of Model 1 to the Banister fatigue weight k2.

The objective p_T = k1*CTL_T - k2*ATL_T depends on the k2/k1 ratio, which is
athlete-specific and hard to estimate. This sweep shows how the optimal plan's
race-day TSB responds: higher k2 -> deeper taper. Coaching literature
recommends race-day TSB in roughly +15..+25; the sweep identifies which k2
values agree with that band — an honest calibration discussion for the report.

Usage:
    python -m src.analysis.k2_sensitivity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

from src.models.model1_training import build_model, extract_solution, solve

from .common import FIGURES, TABLES, load_inputs, with_param

TSB_BAND = (15.0, 25.0)   # coaching-recommended race-day form


def sweep(profile: dict, state: dict, rates: pd.DataFrame, grid=None) -> pd.DataFrame:
    grid = grid if grid is not None else np.arange(1.0, 3.01, 0.25)
    rows = []
    for k2 in grid:
        p = with_param(profile, ["banister", "k2"], float(k2))
        prob, v = build_model(p, state, rates)
        assert solve(prob) == "Optimal"
        sol = extract_solution(prob, v, state)
        traj = sol["trajectory"]
        tsb = traj["ctl"].iloc[-1] - traj["atl"].iloc[-1]
        rows.append({
            "k2": round(float(k2), 2),
            "race_day_ctl": round(traj["ctl"].iloc[-1], 1),
            "race_day_atl": round(traj["atl"].iloc[-1], 1),
            "race_day_tsb": round(tsb, 1),
            "total_hours": round(traj["hours"].sum(), 1),
            "in_coaching_band": TSB_BAND[0] <= tsb <= TSB_BAND[1],
        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["k2"], df["race_day_tsb"], "o-", color="#2563eb")
    ax.axhspan(*TSB_BAND, color="#bbf7d0", alpha=0.5, label="coaching band (+15..+25)")
    ax.axvline(2.0, color="#94a3b8", ls="--", lw=1)
    ax.annotate("baseline k2=2", (2.0, df["race_day_tsb"].min()), fontsize=9,
                textcoords="offset points", xytext=(6, 4), color="#475569")
    ax.set_xlabel("Fatigue weight $k_2$ (with $k_1 = 1$)")
    ax.set_ylabel("Optimal race-day TSB")
    ax.set_title("Model 1 — taper depth vs fatigue weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    profile, state, rates = load_inputs()
    df = sweep(profile, state, rates)
    df.to_csv(TABLES / "model1_k2_sensitivity.csv", index=False)
    plot(df, FIGURES / "model1_k2_sensitivity.png")
    print(df.to_string(index=False))
    ok = df[df["in_coaching_band"]]
    if not ok.empty:
        print(f"\n[k2] TSB lands in the +15..+25 coaching band for k2 in "
              f"[{ok['k2'].min()}, {ok['k2'].max()}]")

"""Parametric analysis: race time as a function of FTP — 'what is +5 W worth?'

Re-solves Model 3 over a grid of FTP values. Higher FTP shifts every bike
zone's power (and cube-root speed) upward. The negative of the empirical
slope answers the classic athlete question: seconds saved per extra watt.

Also sweeps race-day CTL (the Model 1 link): the energy budget dual predicts
the value of fitness; the sweep verifies it.

Usage:
    python -m src.analysis.ftp_sweep
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

from src.models.model3_pacing import build_model, solve
from src.models.run_model3 import race_day_ctl

from .common import FIGURES, TABLES, load_inputs


def sweep_ftp(profile: dict, ctl: float, grid=None) -> pd.DataFrame:
    grid = grid if grid is not None else np.arange(240, 281, 5)
    rows = []
    for ftp in grid:
        prob, _ = build_model(profile, ctl, ftp_watts=float(ftp))
        assert solve(prob) == "Optimal"
        rows.append({"ftp_watts": int(ftp), "total_min": pulp.value(prob.objective)})
    df = pd.DataFrame(rows)
    df["sec_saved_per_watt"] = -df["total_min"].diff() / df["ftp_watts"].diff() * 60.0
    return df


def sweep_ctl(profile: dict, grid=None) -> pd.DataFrame:
    grid = grid if grid is not None else np.arange(70, 96, 5)
    rows = []
    for ctl in grid:
        prob, _ = build_model(profile, float(ctl))
        assert solve(prob) == "Optimal"
        rows.append({"ctl": float(ctl), "total_min": pulp.value(prob.objective)})
    df = pd.DataFrame(rows)
    df["sec_saved_per_ctl"] = -df["total_min"].diff() / df["ctl"].diff() * 60.0
    return df


def plot(ftp_df: pd.DataFrame, ctl_df: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(ftp_df["ftp_watts"], ftp_df["total_min"], "o-", color="#2563eb")
    ax1.axvline(260, color="#94a3b8", ls="--", lw=1)
    ax1.set_xlabel("FTP (W)")
    ax1.set_ylabel("Optimal race time (min)")
    ax1.set_title("Race time vs FTP (current: 260 W)")

    ax2.plot(ctl_df["ctl"], ctl_df["total_min"], "s-", color="#16a34a")
    ax2.axvline(81.6, color="#94a3b8", ls="--", lw=1)
    ax2.set_xlabel("Race-day CTL (fitness)")
    ax2.set_ylabel("Optimal race time (min)")
    ax2.set_title("Race time vs fitness (Model 1 link)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    profile, _, _ = load_inputs()
    ctl = race_day_ctl("results/tables/model1_trajectory.csv")

    ftp_df = sweep_ftp(profile, ctl)
    ctl_df = sweep_ctl(profile)
    ftp_df.round(3).to_csv(TABLES / "model3_ftp_sweep.csv", index=False)
    ctl_df.round(3).to_csv(TABLES / "model3_ctl_sweep.csv", index=False)
    plot(ftp_df, ctl_df, FIGURES / "model3_sweeps.png")

    at_260 = ftp_df.loc[ftp_df["ftp_watts"] == 265, "sec_saved_per_watt"]
    print(ftp_df.round(2).to_string(index=False))
    print(ctl_df.round(2).to_string(index=False))
    if not at_260.empty:
        print(f"\n[ftp_sweep] around current FTP: +5 W of FTP buys "
              f"~{5 * float(at_260.iloc[0]):.0f} s of race time")

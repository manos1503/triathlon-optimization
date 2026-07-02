"""Daily CTL/ATL/TSB trajectories via Banister exponential smoothing.

    CTL_t = lam_c * CTL_{t-1} + (1 - lam_c) * L_t     lam_c = exp(-1/42)
    ATL_t = lam_a * ATL_{t-1} + (1 - lam_a) * L_t     lam_a = exp(-1/7)
    TSB_t = CTL_{t-1} - ATL_{t-1}

where L_t is total TRIMP on day t. The final (CTL, ATL) pair is the
initial fitness state CTL_0/ATL_0 consumed by Model 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_load(activities: pd.DataFrame) -> pd.Series:
    """Total TRIMP per calendar day, zero-filled over the full date range."""
    per_day = activities.groupby("date")["trimp"].sum()
    idx = pd.date_range(per_day.index.min(), per_day.index.max(), freq="D")
    return per_day.reindex(idx, fill_value=0.0).rename("load")


def ctl_atl(load: pd.Series, tau_ctl: float = 42.0, tau_atl: float = 7.0,
            ctl0: float = 0.0, atl0: float = 0.0) -> pd.DataFrame:
    """Run the exponential-smoothing recursions over a daily load series."""
    lam_c, lam_a = np.exp(-1.0 / tau_ctl), np.exp(-1.0 / tau_atl)

    ctl = np.empty(len(load))
    atl = np.empty(len(load))
    prev_c, prev_a = ctl0, atl0
    for i, l in enumerate(load.to_numpy()):
        prev_c = lam_c * prev_c + (1 - lam_c) * l
        prev_a = lam_a * prev_a + (1 - lam_a) * l
        ctl[i], atl[i] = prev_c, prev_a

    out = pd.DataFrame({"load": load, "ctl": ctl, "atl": atl}, index=load.index)
    out["tsb"] = out["ctl"].shift(1, fill_value=ctl0) - out["atl"].shift(1, fill_value=atl0)
    out.index.name = "date"
    return out


def current_state(traj: pd.DataFrame) -> dict:
    """Final fitness state — the CTL_0/ATL_0 initial condition for Model 1."""
    last = traj.iloc[-1]
    return {
        "date": str(traj.index[-1].date()),
        "ctl": round(float(last["ctl"]), 2),
        "atl": round(float(last["atl"]), 2),
        "tsb": round(float(last["ctl"] - last["atl"]), 2),
    }

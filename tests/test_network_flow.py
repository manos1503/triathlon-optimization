"""Tests for the network-flow (longest-path) extension of Model 2."""

from src.analysis.network_flow import longest_path_dp, longest_path_lp, restricted_mip
from src.models.model2_calendar import race_value, readiness

from .test_model2 import PROFILE, RACES


def test_three_methods_agree():
    ready = readiness(RACES, PROFILE)
    rr = RACES[ready].copy()
    values = race_value(rr, PROFILE["model2"])
    rec = PROFILE["model2"]["recovery_weeks"]

    z_dp, path = longest_path_dp(rr, values, rec)
    z_lp, flows = longest_path_lp(rr, values, rec)
    z_mip, sel = restricted_mip(RACES, PROFILE)

    assert abs(z_dp - z_lp) < 1e-4
    assert abs(z_dp - z_mip) < 1e-4
    assert set(path) == set(sel)


def test_flow_lp_integral():
    ready = readiness(RACES, PROFILE)
    rr = RACES[ready].copy()
    values = race_value(rr, PROFILE["model2"])
    _, flows = longest_path_lp(rr, values, PROFILE["model2"]["recovery_weeks"])
    assert all(abs(v - round(v)) < 1e-6 for v in flows.values())


def test_path_respects_recovery():
    ready = readiness(RACES, PROFILE)
    rr = RACES[ready].copy()
    values = race_value(rr, PROFILE["model2"])
    rec = PROFILE["model2"]["recovery_weeks"]
    _, path = longest_path_dp(rr, values, rec)
    weeks = rr.set_index("id").loc[path, ["week", "distance_class"]].to_numpy()
    for (w1, c1), (w2, _) in zip(weeks, weeks[1:]):
        assert w2 - w1 >= rec[str(c1)] + 1

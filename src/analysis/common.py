"""Shared helpers for Model 1 analysis modules."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import yaml

FIGURES = Path("results/figures")
TABLES = Path("results/tables")


def load_inputs(profile_path: str = "docs/athlete_profile.yaml",
                state_path: str = "data/processed/fitness_state.yaml",
                rates_path: str = "data/processed/trimp_rates.csv"):
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    with open(state_path) as f:
        state = yaml.safe_load(f)
    rates = pd.read_csv(rates_path)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    return profile, state, rates


def with_param(profile: dict, path: list[str], value) -> dict:
    """Deep-copied profile with one parameter overridden (e.g. ['constraints','weekly_hours_max'])."""
    p = copy.deepcopy(profile)
    node = p
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return p

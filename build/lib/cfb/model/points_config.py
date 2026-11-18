"""Configuration of the phase 2 points models: the home margin (spread) and the game total.
Chosen on seasons through 2025 by scripts/points_select.py; frozen before 2026 is scored
(docs/points.md)."""
from __future__ import annotations

SPREAD_ID = "spread_v1"
TOTAL_ID = "total_v1"

# Groups the selection considered. market_lines is the benchmark, never an input;
# ctx_weather holds observed weather for played games and forecasts for upcoming ones, so a
# model trained on it would be scored on better information than a live run has.
CANDIDATE_GROUPS = [
    "str_power", "str_prior", "str_elo", "str_schedule", "str_luck", "str_special_teams",
    "str_form", "str_polls",
    "matchup_style", "matchup_history",
    "coach_tenure", "coach_style",
    "ctx_schedule", "ctx_season_phase", "ctx_travel", "ctx_venue", "ctx_home_road",
    "ctx_home_field", "ctx_crowd", "ctx_portal", "ctx_qb",
]

# Team columns enter as home minus away (d) for the margin, home plus away (s) for the total.
FORMS = {"margin": ("d",), "total": ("s",)}

# Frozen 2026-11-17 on seasons through 2025 (docs/points.md has the selection record).
# Margin: forward selection over whole groups (stop: MAE gain under 0.01 or fewer than 5 of 8
# seasons better). Total: str_power plus single columns: the coach's fourth-down and run
# tendencies (rates only; the coach_style count columns grow with the data window since 2014
# and act as a calendar trend) and the pace columns of matchup_style.
GROUPS = {"margin": ["str_power", "ctx_crowd", "str_prior", "str_elo"], "total": ["str_power"]}
EXTRA: dict[str, list[str]] = {
    "margin": [],
    "total": [
        "s_coach_fourth_go_oe", "s_coach_run_oe", "s_coach_flex_slope", "s_coach_flex_shift",
        "s_mat_pace_plays", "s_mat_off_plays", "s_mat_off_sec_per_play", "s_mat_off_expl_rate",
        "s_mat_def_expl_rate", "s_mat_off_run_rate_neutral",
    ],
}
# Groups that EXTRA columns come from (they must be in the matrix).
EXTRA_GROUPS = {"margin": [], "total": ["coach_style", "matchup_style"]}
BASE = {"margin": ["site", "fcs_game"], "total": ["site", "fcs_game"]}
LAM = {"margin": 0.03, "total": 0.03}
CAP: dict[str, float | None] = {"margin": None, "total": None}
ERA_HALF_LIFE: dict[str, float | None] = {"margin": None, "total": None}
TRAIN_START = 2015


def matrix_groups(target: str) -> list[str]:
    """Every feature group the target's columns come from."""
    return GROUPS[target] + [g for g in EXTRA_GROUPS[target] if g not in GROUPS[target]]


def model_columns(target: str, groups: list[str] | None = None) -> list[str]:
    from .matrix import feature_columns, groups_by_name
    extra = EXTRA[target] if groups is None else []
    return BASE[target] + feature_columns(groups_by_name(groups or GROUPS[target]), FORMS[target]) + extra


def make_model(target: str, groups: list[str] | None = None):
    """The frozen points model for `target` ("margin" or "total"), unfitted. `groups`
    overrides the feature groups (tests)."""
    from .models import PointsModel
    return PointsModel(model_columns(target, groups), lam=LAM[target], era_half_life=ERA_HALF_LIFE[target])


def clip_target(target: str, y):
    """Training target after the configured cap (margins capped at +-CAP, totals at CAP)."""
    import numpy as np
    cap = CAP[target]
    y = np.asarray(y, float)
    if cap is None:
        return y
    return np.clip(y, -cap, cap) if target == "margin" else np.minimum(y, cap)

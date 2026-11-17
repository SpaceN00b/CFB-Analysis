"""The model configuration. Chosen on seasons through 2025 by scripts/model_select.py;
frozen before 2026 is scored (docs/model.md)."""
from __future__ import annotations

MODEL_ID = "model_v1"

# Every group the selection considered. market_lines is out by the approved default (the
# model stays independent of the line); ctx_weather is phase 2 (observed in training,
# forecast live).
CANDIDATE_GROUPS = [
    "str_power", "str_prior", "str_elo", "str_schedule", "str_luck", "str_special_teams",
    "str_form", "str_polls",
    "matchup_style", "matchup_history",
    "coach_tenure", "coach_style",
    "ctx_schedule", "ctx_season_phase", "ctx_travel", "ctx_venue", "ctx_home_road",
    "ctx_home_field", "ctx_crowd", "ctx_portal", "ctx_qb",
]

# Team columns enter as home-minus-away differences.
FORMS = ("d",)

# Frozen 2026-11-17 on seasons through 2025 (docs/model.md has the selection record).
# Margin regression on the two rating groups; every other group added less than its
# standard error in season-forward log loss.
MODEL_GROUPS = ["str_power", "str_elo"]
EXTRA_COLUMNS: list[str] = []
LAM = 0.03
ERA_HALF_LIFE = None
TRAIN_START = 2015
FAMILY = "margin"


def model_columns(groups: list[str] | None = None) -> list[str]:
    from .matrix import feature_columns, groups_by_name
    extra = EXTRA_COLUMNS if groups is None else []
    return ["site", "fcs_game"] + feature_columns(groups_by_name(groups or MODEL_GROUPS), FORMS) + extra


def make_model(groups: list[str] | None = None):
    """The frozen model, unfitted. `groups` overrides the feature groups (tests)."""
    from .models import LogitModel, MarginModel
    cls = LogitModel if FAMILY == "logit" else MarginModel
    return cls(model_columns(groups), lam=LAM, era_half_life=ERA_HALF_LIFE)

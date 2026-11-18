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


# ---- model_v2 (P2, early-season priors) ---------------------------------------------------
# Same family, penalty and training rows as model_v1. The power rating's prior is the
# player-based preseason projection (str_power2 replaces str_power's rating columns; the
# unit ratings and game count stay), plus the long-window program residual (str_program).
# Frozen on seasons through 2025 (docs/priors.md has the selection record).
MODEL_V2_ID = "model_v2"
MODEL_V2_GROUPS = ["str_power", "str_power2", "str_elo", "str_program"]
MODEL_V2_COLUMNS = ["site", "fcs_game"] + [f"d_{c}" for c in (
    "str_pwr2_off", "str_pwr2_def", "str_pwr2_net", "str_pwr_games", "str_pwr2_prior_share", "str_pwr2_net_se",
    "str_unit_off_epa", "str_unit_def_epa", "str_unit_off_success", "str_unit_def_success",
    "str_unit_off_expl", "str_unit_def_expl", "str_unit_net_epa",
    "str_elo", "str_elo_win_prob",
    "str_prog_resid",
)]

MODELS = {MODEL_ID: MODEL_GROUPS, MODEL_V2_ID: MODEL_V2_GROUPS}


def groups_for(model_id: str = MODEL_ID) -> list[str]:
    return list(MODELS[model_id])


def model_columns(groups: list[str] | None = None, model_id: str = MODEL_ID) -> list[str]:
    from .matrix import feature_columns, groups_by_name
    if model_id == MODEL_V2_ID and groups is None:
        return list(MODEL_V2_COLUMNS)
    extra = EXTRA_COLUMNS if groups is None else []
    return ["site", "fcs_game"] + feature_columns(groups_by_name(groups or MODEL_GROUPS), FORMS) + extra


def make_model(groups: list[str] | None = None, model_id: str = MODEL_ID):
    """A frozen model (model_v1 by default), unfitted. `groups` overrides the feature groups (tests)."""
    from .models import LogitModel, MarginModel
    cls = LogitModel if FAMILY == "logit" else MarginModel
    return cls(model_columns(groups, model_id), lam=LAM, era_half_life=ERA_HALF_LIFE)

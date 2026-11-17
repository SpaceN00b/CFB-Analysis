"""Evaluation: metrics, season-forward and walk-forward runs, paired bootstrap, baselines.

    from cfb.eval import evaluate, walk_forward, write_weekly, paired_bootstrap, run_baseline

    preds = walk_forward(warehouse, fit, predict, "my_model", seasons=range(2018, 2026))
    write_weekly(preds, "data")
    doc = evaluate(preds, warehouse.outcomes())

CLI: `python -m cfb.eval baselines`, `python -m cfb.eval score <model_id>`. Notes: docs/eval.md.
Outcomes come from `Warehouse.outcomes()`, which is allowed here and never in feature code.
"""
from .baselines import BASELINES, run_baseline
from .bootstrap import common_games, compare, paired_bootstrap
from .metrics import accuracy, brier, calibration, evaluate, log_loss, score_block, scored_games
from .splits import (
    DEFAULT_SEASONS, TEST_SEASON, Split, season_forward, season_forward_predict, visible_results, walk_forward,
    write_weekly,
)

__all__ = [
    "BASELINES", "DEFAULT_SEASONS", "TEST_SEASON", "Split", "accuracy", "brier", "calibration", "common_games",
    "compare", "evaluate", "log_loss", "paired_bootstrap", "run_baseline", "score_block", "scored_games",
    "season_forward", "season_forward_predict", "visible_results", "walk_forward", "write_weekly",
]

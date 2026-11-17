"""Scores for win probabilities and the metrics document (`data/eval/<model_id>.json`).

    doc = evaluate(predictions, warehouse.outcomes())     # contract in cfb.core.schemas
    write_metrics(doc, "data")

`predictions` needs `game_id`, `model_id`, `p_home_win`. Games are scored against
completed results; a game without a winner (tie) is left out. The headline scope is
FBS vs FBS: `overall`, `by_season` and `by_week` cover it. `by_segment` covers
`fbs_vs_fbs` and `fbs_vs_fcs` over all scored games, and `conference`, `non_conference`,
`neutral`, `regular`, `postseason` inside the scope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.schemas import METRICS_SCHEMA_VERSION, SchemaError, validate_metrics

# Probabilities are clipped to [EPS, 1 - EPS] for log loss, as in the exploration that set
# the baseline numbers (effects.py).
EPS = 1e-6
CALIBRATION_BINS = 10
SCOPES = ("fbs_vs_fbs", "all")
_ROUND = 6


def log_loss(y, p) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else float("nan")


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def accuracy(y, p) -> float:
    """Share of games where the side given 0.5 or more won. A 0.5 counts as a home pick."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p >= 0.5) == (y == 1))) if len(y) else float("nan")


def per_game_losses(y, p, metric: str) -> np.ndarray:
    """Per-game values whose mean is the metric; used by the paired bootstrap."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if metric == "log_loss":
        q = np.clip(p, EPS, 1 - EPS)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q))
    if metric == "brier":
        return (p - y) ** 2
    if metric == "accuracy":
        return ((p >= 0.5) == (y == 1)).astype(float)
    raise ValueError(f"unknown metric {metric!r}")


def calibration(y, p, bins: int = CALIBRATION_BINS) -> list[dict]:
    """Equal-width bins on [0, 1]; a probability of exactly 1 goes in the last bin."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1) if len(p) else np.array([], dtype=int)
    out = []
    for i in range(bins):
        hit = which == i
        out.append({
            "p_lo": round(float(edges[i]), 4), "p_hi": round(float(edges[i + 1]), 4), "n": int(hit.sum()),
            "p_mean": round(float(p[hit].mean()), _ROUND) if hit.any() else None,
            "win_rate": round(float(y[hit].mean()), _ROUND) if hit.any() else None,
        })
    return out


def score_block(y, p, with_calibration: bool = True) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = int(len(y))
    block = {"n": n, "log_loss": None, "brier": None, "accuracy": None}
    if n:
        block.update(log_loss=round(log_loss(y, p), _ROUND), brier=round(brier(y, p), _ROUND),
                     accuracy=round(accuracy(y, p), _ROUND))
    if with_calibration:
        block["calibration"] = calibration(y, p)
    return block


def scored_games(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Predictions joined to completed results, with `y` = 1 when the home team won.

    Season, week and the game flags come from `outcomes` (game_index columns)."""
    keep = [c for c in ("game_id", "model_id", "p_home_win") if c in predictions.columns]
    pred = predictions[keep]
    if pred["game_id"].duplicated().any():
        raise SchemaError("predictions hold more than one row per game; score one model at a time")
    cols = ["game_id", "season", "season_type", "week", "start_time_utc", "neutral", "conference_game",
            "home_fbs", "away_fbs", "home_points", "away_points"]
    out = outcomes[[c for c in cols if c in outcomes.columns]]
    out = out[out["home_points"].notna() & out["away_points"].notna() & (out["home_points"] != out["away_points"])]
    g = pred.merge(out, on="game_id", how="inner")
    g["y"] = (g["home_points"] > g["away_points"]).astype("float64")
    g["fbs_vs_fbs"] = g["home_fbs"].astype(bool) & g["away_fbs"].astype(bool)
    return g.sort_values(["start_time_utc", "game_id"]).reset_index(drop=True)


def evaluate(predictions: pd.DataFrame, outcomes: pd.DataFrame, model_id: str | None = None,
             scope: str = "fbs_vs_fbs", scope_label: str | None = None,
             created_utc: pd.Timestamp | None = None) -> dict:
    """Metrics document for one model's predictions. `scope` is "fbs_vs_fbs" or "all";
    `scope_label` overrides the label written to the file (e.g. to name a common game set)."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    if model_id is None:
        ids = predictions["model_id"].unique() if "model_id" in predictions else []
        if len(ids) != 1:
            raise ValueError("pass model_id when predictions hold no or several models")
        model_id = str(ids[0])
    g = scored_games(predictions, outcomes)
    s = g[g["fbs_vs_fbs"]] if scope == "fbs_vs_fbs" else g
    blk = lambda d, cal=True: score_block(d["y"].to_numpy(), d["p_home_win"].to_numpy(), cal)
    segments = {
        "fbs_vs_fbs": g[g["fbs_vs_fbs"]],
        "fbs_vs_fcs": g[~g["fbs_vs_fbs"] & (g["home_fbs"].astype(bool) | g["away_fbs"].astype(bool))],
        "conference": s[s["conference_game"].astype(bool)],
        "non_conference": s[~s["conference_game"].astype(bool)],
        "neutral": s[s["neutral"].astype(bool)],
        "regular": s[s["season_type"] == "regular"],
        "postseason": s[s["season_type"] == "postseason"],
    }
    created = created_utc if created_utc is not None else pd.Timestamp.now(tz="UTC")
    doc = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "model_id": model_id,
        "created_utc": created.isoformat(),
        "scope": scope_label or scope,
        "overall": blk(s),
        "by_season": [{"season": int(k), **blk(d)} for k, d in s.groupby("season", sort=True)],
        "by_week": [
            {"season": int(a), "season_type": str(b), "week": int(c), **blk(d, False)}
            for (a, b, c), d in s.groupby(["season", "season_type", "week"], sort=True)
        ],
        "by_segment": [{"segment": k, **blk(d, False)} for k, d in segments.items()],
    }
    return validate_metrics(doc)


__all__ = [
    "CALIBRATION_BINS", "EPS", "SCOPES", "accuracy", "brier", "calibration", "evaluate", "log_loss",
    "per_game_losses", "score_block", "scored_games",
]

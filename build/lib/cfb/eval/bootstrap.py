"""Paired bootstrap for the difference between two prediction sets on the same games.

    res = paired_bootstrap(y, p_a, p_b, metric="log_loss")
    res["diff"]       metric(a) - metric(b); negative log loss or Brier difference: a is better
    res["se"], res["ci_lo"], res["ci_hi"]
    res["p_a_better"] share of resamples in which a scores better than b

All three metrics are means of per-game values, so resampling the per-game differences
is exact. With `clusters` (e.g. season-week labels) whole clusters are resampled, which
allows for games of one week sharing conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import per_game_losses, scored_games

LOWER_IS_BETTER = {"log_loss": True, "brier": True, "accuracy": False}


def paired_bootstrap(y, p_a, p_b, metric: str = "log_loss", n_boot: int = 2000, seed: int = 0,
                     clusters=None, alpha: float = 0.05) -> dict:
    y = np.asarray(y, float)
    la, lb = per_game_losses(y, p_a, metric), per_game_losses(y, p_b, metric)
    d = la - lb
    n = len(d)
    if n == 0:
        raise ValueError("no games to compare")
    rng = np.random.default_rng(seed)
    if clusters is None:
        sums, counts = d, np.ones(n)
    else:
        codes, _ = pd.factorize(pd.Series(clusters).astype("str"))
        sums = np.bincount(codes, weights=d)
        counts = np.bincount(codes).astype(float)
    k = len(sums)
    boots = np.empty(n_boot)
    step = max(1, 2_000_000 // max(k, 1))  # bound memory of the index matrix
    for start in range(0, n_boot, step):
        stop = min(n_boot, start + step)
        idx = rng.integers(0, k, size=(stop - start, k))
        boots[start:stop] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    better = boots < 0 if LOWER_IS_BETTER[metric] else boots > 0
    return {
        "metric": metric, "n": int(n), "n_clusters": int(k),
        "a": float(la.mean()), "b": float(lb.mean()), "diff": float(d.mean()),
        "se": float(boots.std(ddof=1)),
        "ci_lo": float(np.quantile(boots, alpha / 2)), "ci_hi": float(np.quantile(boots, 1 - alpha / 2)),
        "p_a_better": float(better.mean()),
    }


def common_games(predictions: dict[str, pd.DataFrame], outcomes: pd.DataFrame,
                 fbs_only: bool = True) -> pd.DataFrame:
    """One row per completed game that every model predicted: game columns, `y`, and one
    `p_<model_id>` column per model."""
    frame = None
    for model_id, pred in predictions.items():
        g = scored_games(pred, outcomes)
        if fbs_only:
            g = g[g["fbs_vs_fbs"]]
        g = g.rename(columns={"p_home_win": f"p_{model_id}"}).drop(columns=["model_id"], errors="ignore")
        frame = g if frame is None else frame.merge(g[["game_id", f"p_{model_id}"]], on="game_id", how="inner")
    if frame is None:
        raise ValueError("no predictions given")
    return frame.reset_index(drop=True)


def compare(pred_a: pd.DataFrame, pred_b: pd.DataFrame, outcomes: pd.DataFrame, metric: str = "log_loss",
            fbs_only: bool = True, by_week: bool = False, **kwargs) -> dict:
    """Paired bootstrap of model a against model b on the completed games both predicted."""
    g = common_games({"a": pred_a, "b": pred_b}, outcomes, fbs_only)
    clusters = (g["season"].astype(str) + "_" + g["season_type"] + "_" + g["week"].astype(str)) if by_week else None
    return paired_bootstrap(g["y"], g["p_a"], g["p_b"], metric, clusters=clusters, **kwargs)


__all__ = ["LOWER_IS_BETTER", "common_games", "compare", "paired_bootstrap"]

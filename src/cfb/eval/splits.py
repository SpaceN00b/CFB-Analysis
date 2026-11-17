"""Season-forward splits and the walk-forward runner.

Season-forward (model selection on seasons through 2025):

    for split in season_forward(frame):            # S = 2018 ... 2025
        model = fit(split.train)                   # seasons before S
        p = predict(model, split.test)             # season S

Walk-forward (any model, including the 2026 test run week by week):

    preds = walk_forward(warehouse, fit, predict, "my_model", seasons=[2026], refit="cutoff")
    write_weekly(preds, "data")

    fit(view, train) -> model
        view   AsOfView at the cutoff being predicted
        train  completed games whose results are visible at that cutoff (game_index columns
               with results), oldest first. Read features of these games through `view`.
    predict(model, view, slate) -> DataFrame
        slate  schedule rows (no results) of the games at this cutoff
        returns game_id, p_home_win and optionally pred_margin, pred_total, reasons
        (JSON text or a list of dicts per row) and cutoff_utc (a later time when an input
        was known only later, e.g. a closing line at kickoff). Games left out get no row.

refit="season" fits once per season at its first cutoff (only earlier seasons are visible
then); refit="cutoff" fits again at every cutoff with every result visible at it.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from ..core.asof import Warehouse
from ..core.schemas import (
    MODEL_ID, PREDICTION_COLUMNS, SchemaError, reasons_json, validate_predictions, write_predictions,
)
from ..core.timeutil import RESULT_LAG, as_utc

DEFAULT_SEASONS = tuple(range(2018, 2026))
TEST_SEASON = 2026
REFIT = ("season", "cutoff")

FitFn = Callable[[Any, pd.DataFrame], Any]
PredictFn = Callable[[Any, Any, pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class Split:
    season: int
    train: pd.DataFrame
    test: pd.DataFrame


def season_forward(frame: pd.DataFrame, seasons: Iterable[int] = DEFAULT_SEASONS,
                   season_col: str = "season") -> Iterator[Split]:
    """Fit on seasons before S, score S. Seasons with no test rows or no training rows are skipped."""
    for s in seasons:
        train, test = frame[frame[season_col] < s], frame[frame[season_col] == s]
        if len(train) and len(test):
            yield Split(int(s), train, test)


def season_forward_predict(frame: pd.DataFrame, fit: Callable[[pd.DataFrame], Any],
                           predict: Callable[[Any, pd.DataFrame], np.ndarray],
                           seasons: Iterable[int] = DEFAULT_SEASONS, season_col: str = "season") -> pd.Series:
    """Out-of-season predictions for the rows of `frame` in `seasons`, indexed like `frame`."""
    parts = []
    for split in season_forward(frame, seasons, season_col):
        parts.append(pd.Series(np.asarray(predict(fit(split.train), split.test), float), index=split.test.index))
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype="float64")


def visible_results(warehouse: Warehouse, cutoff) -> pd.DataFrame:
    """Completed games whose results are visible at `cutoff` (kickoff + RESULT_LAG <= cutoff)."""
    out = warehouse.outcomes()
    return out[out["start_time_utc"] + RESULT_LAG <= as_utc(cutoff)].reset_index(drop=True)


def _reasons_column(values, n: int) -> list[str]:
    if values is None:
        return ["[]"] * n
    return [v if isinstance(v, str) else reasons_json(v) for v in values]


def walk_forward(warehouse: Warehouse, fit: FitFn, predict: PredictFn, model_id: str,
                 seasons: Iterable[int] = DEFAULT_SEASONS, games: pd.DataFrame | None = None,
                 refit: str = "season", created_utc: pd.Timestamp | None = None,
                 progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    """Predict every game of `seasons` (or of `games`) cutoff by cutoff.

    Returns predictions in the cfb.core.schemas contract plus `season`, `season_type`, `week`.
    """
    if refit not in REFIT:
        raise ValueError(f"refit must be one of {REFIT}")
    if not MODEL_ID.fullmatch(model_id):
        raise SchemaError(f"invalid model_id {model_id!r}")
    seasons = sorted({int(s) for s in seasons})
    sched = warehouse.schedule(seasons) if games is None else games[games["season"].isin(seasons)]
    created = created_utc if created_utc is not None else pd.Timestamp.now(tz="UTC")
    slates = list(sched.groupby("cutoff_utc", sort=True))
    parts, model, fitted_season = [], None, None
    for i, (cutoff, slate) in enumerate(slates):
        slate = slate.reset_index(drop=True)
        view = warehouse.view(cutoff)
        season = int(slate["season"].iloc[0])
        if refit == "cutoff" or fitted_season != season:
            model, fitted_season = fit(view, visible_results(warehouse, cutoff)), season
        out = predict(model, view, slate)
        if progress:
            progress(i + 1, len(slates))
        if out is None or len(out) == 0:
            continue
        stray = ~out["game_id"].isin(slate["game_id"])
        if stray.any():
            raise SchemaError(f"predict returned {int(stray.sum())} games outside the slate at {cutoff}")
        frame = pd.DataFrame({
            "game_id": out["game_id"].to_numpy(),
            "model_id": model_id,
            "p_home_win": out["p_home_win"].to_numpy(dtype="float64"),
            "pred_margin": out["pred_margin"].to_numpy(dtype="float64") if "pred_margin" in out else np.nan,
            "pred_total": out["pred_total"].to_numpy(dtype="float64") if "pred_total" in out else np.nan,
            "cutoff_utc": out["cutoff_utc"].array if "cutoff_utc" in out else pd.Series([cutoff] * len(out)).array,
            "created_utc": created,
            "reasons": _reasons_column(out["reasons"].tolist() if "reasons" in out else None, len(out)),
        })
        parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=PREDICTION_COLUMNS + ["season", "season_type", "week"])
    preds = validate_predictions(pd.concat(parts, ignore_index=True))
    late = preds.merge(sched[["game_id", "cutoff_utc"]].rename(columns={"cutoff_utc": "_c"}), on="game_id")
    if (late["cutoff_utc"] < late["_c"]).any():
        raise SchemaError("a prediction's cutoff_utc is earlier than its game's cutoff")
    keys = sched[["game_id", "season", "season_type", "week"]]
    return preds.merge(keys, on="game_id", how="left")


def write_weekly(predictions: pd.DataFrame, data_dir: str | Path, replace: bool = False) -> list[Path]:
    """Write predictions as one file per (model, season, season_type, week).

    replace=True first removes the model's existing prediction files, so stale weeks from
    an earlier run do not linger."""
    paths = []
    for model_id, frame in predictions.groupby("model_id", sort=True):
        if replace:
            target = Path(data_dir) / "predictions" / str(model_id)
            if target.exists():
                shutil.rmtree(target)
        for (season, stype, week), d in frame.groupby(["season", "season_type", "week"], sort=True):
            paths.append(write_predictions(d[PREDICTION_COLUMNS], data_dir, int(season), str(stype), int(week)))
    return paths


__all__ = [
    "DEFAULT_SEASONS", "REFIT", "Split", "TEST_SEASON", "season_forward", "season_forward_predict",
    "visible_results", "walk_forward", "write_weekly",
]

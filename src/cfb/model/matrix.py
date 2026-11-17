"""Matrix builder: one row per game from the feature groups.

Team-level groups (keys game_id, team_id) become four columns per feature:
`h_<col>` (home team's value), `a_<col>` (away team's), `d_<col>` (home minus away) and
`s_<col>` (home plus away, for the game total).
Game-level groups keep their column names. Base columns from the schedule are added:
`site` (1 at a true home game, 0 at a neutral site), `home_fbs`, `away_fbs`, `fcs_game`.

Two ways to get the feature values, which give the same numbers:
- `stored_features(data_dir, groups)`: the parquet files written by
  `python -m cfb.features.build`, where each game's values were built at its own cutoff.
  Used for training and the season-forward evaluation.
- `live_features(view, games, groups)`: calls each group's `build(view, games)` for games
  at one cutoff, including on-demand matchups that are in no table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..features.base import FeatureGroup, discover_groups, validate_output

SCHEDULE_COLUMNS = ["game_id", "season", "season_type", "week", "start_time_utc", "cutoff_utc",
                    "neutral", "conference_game", "home_id", "away_id", "home_fbs", "away_fbs"]


def groups_by_name(names: Iterable[str] | None = None) -> list[FeatureGroup]:
    groups = discover_groups()
    if names is None:
        return groups
    names = list(names)
    known = {g.name: g for g in groups}
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise KeyError(f"unknown feature groups: {unknown}")
    return [known[n] for n in names]


def stored_features(data_dir: str | Path, groups: Iterable[FeatureGroup]) -> dict[str, pd.DataFrame]:
    """Feature frames as written by the feature build, one per group."""
    root = Path(data_dir) / "features"
    out = {}
    for g in groups:
        path = root / f"{g.name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run python -m cfb.features.build --groups {g.name}")
        out[g.name] = pd.read_parquet(path)
    return out


def live_features(view, games: pd.DataFrame, groups: Iterable[FeatureGroup]) -> dict[str, pd.DataFrame]:
    """Build each group for `games` (all at view.cutoff) through the view."""
    games = games.reset_index(drop=True)
    return {g.name: validate_output(g, g.build(view, games), games) for g in groups}


def _team_wide(frame: pd.DataFrame, games: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    f = frame[["game_id", "team_id"] + cols]
    g = games[["game_id", "home_id", "away_id"]]
    h = g.merge(f, left_on=["game_id", "home_id"], right_on=["game_id", "team_id"], how="left")
    a = g.merge(f, left_on=["game_id", "away_id"], right_on=["game_id", "team_id"], how="left")
    hv = h[cols].to_numpy(dtype="float64")
    av = a[cols].to_numpy(dtype="float64")
    parts = {}
    for j, c in enumerate(cols):
        parts[f"h_{c}"] = hv[:, j]
        parts[f"a_{c}"] = av[:, j]
        parts[f"d_{c}"] = hv[:, j] - av[:, j]
        parts[f"s_{c}"] = hv[:, j] + av[:, j]
    return pd.DataFrame(parts, index=games.index)


def game_matrix(games: pd.DataFrame, features: dict[str, pd.DataFrame],
                groups: Iterable[FeatureGroup]) -> pd.DataFrame:
    """One row per row of `games` (same order): schedule columns, base columns, features."""
    games = games.reset_index(drop=True)
    base = games[[c for c in SCHEDULE_COLUMNS if c in games.columns]].copy()
    base["site"] = 1.0 - games["neutral"].astype(bool).astype("float64")
    base["home_fbs"] = games["home_fbs"].astype(bool).astype("float64")
    base["away_fbs"] = games["away_fbs"].astype(bool).astype("float64")
    base["fcs_game"] = 1.0 - base["home_fbs"] * base["away_fbs"]
    blocks = [base]
    for g in groups:
        frame = features[g.name]
        cols = g.column_names
        if g.level == "team":
            blocks.append(_team_wide(frame, games, cols))
        else:
            m = games[["game_id"]].merge(frame[["game_id"] + cols], on="game_id", how="left")
            blocks.append(m[cols].astype("float64").set_axis(games.index))
    return pd.concat(blocks, axis=1)


def feature_columns(groups: Iterable[FeatureGroup], forms: tuple[str, ...] = ("d",)) -> list[str]:
    """Matrix column names of the groups: team columns in the requested forms (h, a, d, s),
    game columns as they are."""
    out = []
    for g in groups:
        for c in g.column_names:
            if g.level == "team":
                out += [f"{p}_{c}" for p in forms]
            else:
                out.append(c)
    return out


__all__ = ["SCHEDULE_COLUMNS", "feature_columns", "game_matrix", "groups_by_name", "live_features",
           "stored_features"]

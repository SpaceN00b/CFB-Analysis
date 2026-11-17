"""Helpers shared by the strength groups."""
from __future__ import annotations

import numpy as np
import pandas as pd


def team_keys(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, side): game_id, team_id, opp_id, is_home, neutral, season, start_time_utc."""
    cols = ["game_id", "season", "start_time_utc", "neutral"]
    home = games[cols].assign(team_id=games["home_id"].to_numpy(), opp_id=games["away_id"].to_numpy(), is_home=True)
    away = games[cols].assign(team_id=games["away_id"].to_numpy(), opp_id=games["home_id"].to_numpy(), is_home=False)
    out = pd.concat([home, away], ignore_index=True)
    out["team_id"] = out["team_id"].astype("int64")
    out["opp_id"] = out["opp_id"].astype("int64")
    out["neutral"] = out["neutral"].fillna(False).astype(bool)
    return out


def season_teams(view, seasons) -> pd.DataFrame:
    """season, team_id, team, conference, fbs for the given seasons."""
    return view.table("teams", seasons=seasons)


def by_name(frame: pd.DataFrame, names: pd.DataFrame, name_col: str = "team", season_col: str = "season") -> pd.DataFrame:
    """Attach team_id to a frame keyed by team name and season (rows without a match are dropped)."""
    left = frame.rename(columns={name_col: "_name", season_col: "_season"})
    right = names[["season", "team_id", "team"]].rename(columns={"team": "_name", "season": "_season"})
    out = left.merge(right, on=["_season", "_name"], how="inner")
    return out.rename(columns={"_season": season_col}).drop(columns="_name")


def zscore(values: pd.Series, mask: np.ndarray | None = None) -> pd.Series:
    """Standardize against the rows in `mask` (default all); NaN stays NaN."""
    ref = values[mask] if mask is not None else values
    sd = ref.std()
    if not np.isfinite(sd) or sd == 0:
        return values * np.nan
    return (values - ref.mean()) / sd

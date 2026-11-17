"""Opponent-adjusted points ratings (the margin-based rating).

Same ridge model as the unit ratings, on points per game: each team-game is one
observation (team's points against the opponent's defense, weight 1). Values are in
points per game against an average FBS opponent on a neutral field.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import ridge
from .units import DAY_NS, POOL, TeamIndex, _decay

TG_COLUMNS = ["game_id", "team_id", "opp_id", "is_home", "neutral", "season", "start_time_utc", "points"]


@dataclass(frozen=True)
class PointsParams:
    lam: float = 4.0          # prior strength in games
    half_life: float = np.inf  # weeks
    rho: float = 0.6          # share of last season's final deviation kept
    lam_mu: float = 40.0
    lam_hfa: float = 400.0


DEFAULT = PointsParams()


@dataclass
class PointsFit:
    mu: float
    hfa: float
    off: np.ndarray
    deff: np.ndarray
    games: np.ndarray  # games played (unweighted) per team index


def observations(team_game: pd.DataFrame) -> pd.DataFrame:
    tg = team_game[team_game["points"].notna()]
    neutral = tg["neutral"].fillna(False).astype(bool).to_numpy()
    home = tg["is_home"].astype(bool).to_numpy()
    return pd.DataFrame({
        "game_id": tg["game_id"].to_numpy(), "offense_id": tg["team_id"].to_numpy(dtype=np.int64),
        "defense_id": tg["opp_id"].to_numpy(dtype=np.int64), "season": tg["season"].to_numpy(),
        "start_ns": tg["start_time_utc"].astype("int64").to_numpy(),
        "h": np.where(neutral, 0.0, np.where(home, 1.0, -1.0)),
        "y": tg["points"].to_numpy(dtype=np.float64),
    })


def fit(obs: pd.DataFrame, idx: TeamIndex, now_ns: int, p: PointsParams,
        off0: np.ndarray | None = None, def0: np.ndarray | None = None,
        mu0: float | None = None, hfa0: float = 0.0) -> PointsFit:
    o = idx.index(obs["offense_id"].to_numpy())
    d = idx.index(obs["defense_id"].to_numpy())
    w = _decay(obs["start_ns"].to_numpy(dtype=np.int64), now_ns, p.half_life)
    f = ridge.fit(idx.n, o, d, obs["h"].to_numpy(), obs["y"].to_numpy(), w, p.lam, off0, def0,
                  mu0=mu0, lam_mu=p.lam_mu if mu0 is not None else 0.0, hfa0=hfa0, lam_hfa=p.lam_hfa)
    games = np.bincount(o, minlength=idx.n).astype(np.float64)
    return PointsFit(f.mu, f.hfa, f.off, f.deff, games)


def final_fit(obs_prev: pd.DataFrame, idx_prev: TeamIndex, p: PointsParams) -> PointsFit:
    """Last season's final ratings, without a prior (weak ridge only)."""
    now = int(obs_prev["start_ns"].max()) + DAY_NS
    return fit(obs_prev, idx_prev, now, PointsParams(lam=1.0, half_life=p.half_life, lam_hfa=p.lam_hfa))


def carry(prev: PointsFit, idx_prev: TeamIndex, idx: TeamIndex) -> tuple[np.ndarray, np.ndarray]:
    """Last season's off and def deviations on this season's index (lower division -> pooled)."""
    ids = np.append(idx.fbs_ids, POOL)
    src = idx_prev.index(ids)
    src[-1] = idx_prev.n - 1
    return prev.off[src], prev.deff[src]

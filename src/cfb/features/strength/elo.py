"""In-house Elo, a cross-check on the CFBD pregame Elo and the ridge ratings.

Standard Elo with a margin-of-victory multiplier (the FiveThirtyEight form), home field
in Elo points, and regression toward a division mean between seasons. Teams below FBS
start lower. Ratings are run over every visible completed game since the data begins,
in kickoff order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloParams:
    # Chosen on 2015-2025 (season-forward log loss of home + Elo gap, 0.554; CFBD 0.549).
    k: float = 45.0
    hfa: float = 45.0          # Elo points for the home team (none at neutral sites)
    regress: float = 0.15      # share pulled back toward the division mean each new season
    fbs_start: float = 1500.0
    fcs_start: float = 1200.0


DEFAULT = EloParams()
GAME_COLUMNS = ["game_id", "season", "start_time_utc", "neutral", "home_id", "away_id",
                "home_fbs", "away_fbs", "home_points", "away_points"]


def expected(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def run(games: pd.DataFrame, p: EloParams = DEFAULT, record: bool = False):
    """Run Elo over `games` (completed, any order; sorted here by kickoff).

    Returns (ratings: dict team_id -> Elo, fbs: dict team_id -> bool, season of the last
    game, pre: DataFrame of pregame ratings per game when record=True).
    """
    g = games.sort_values(["start_time_utc", "game_id"])
    ratings: dict[int, float] = {}
    fbs_of: dict[int, bool] = {}
    season_of: dict[int, int] = {}
    pre = []
    cur_season = None
    cols = [g[c].to_numpy() for c in ("game_id", "season", "neutral", "home_id", "away_id", "home_fbs",
                                       "away_fbs", "home_points", "away_points")]
    for gid, season, neutral, h, a, hf, af, hp, ap in zip(*cols):
        season = int(season)
        if season != cur_season:
            cur_season = season
        for t, f in ((h, hf), (a, af)):
            t = int(t)
            if t not in ratings:
                ratings[t] = p.fbs_start if f else p.fcs_start
            elif season_of[t] != season:
                mean = p.fbs_start if f else p.fcs_start
                ratings[t] = ratings[t] + p.regress * (mean - ratings[t])
            season_of[t] = season
            fbs_of[t] = bool(f)
        h, a = int(h), int(a)
        rh, ra = ratings[h], ratings[a]
        adv = 0.0 if neutral else p.hfa
        if record:
            pre.append((gid, rh, ra, adv))
        margin = float(hp) - float(ap)
        e = expected(rh + adv - ra)
        s = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        win_diff = (rh + adv - ra) if margin >= 0 else (ra - rh - adv)
        mult = math.log(abs(margin) + 1.0) * 2.2 / (win_diff * 0.001 + 2.2)
        delta = p.k * mult * (s - e)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    pre_df = pd.DataFrame(pre, columns=["game_id", "home_elo", "away_elo", "adv"]) if record else None
    return ratings, fbs_of, season_of, pre_df


def ratings_at(view, season: int, teams: pd.DataFrame, p: EloParams = DEFAULT) -> pd.Series:
    """Elo of each team of `season` at the view's cutoff, with the new-season regression
    applied to teams that have not played yet this season."""
    gi = view.table("game_index", columns=GAME_COLUMNS + ["completed"])
    gi = gi[gi["completed"].fillna(False).astype(bool) & gi["home_points"].notna() & gi["away_points"].notna()]
    ratings, fbs_of, season_of, _ = run(gi, p)
    ids = teams["team_id"].drop_duplicates().to_numpy(dtype=np.int64)
    fbs_now = dict(zip(teams["team_id"].astype(int), teams["fbs"].fillna(False).astype(bool)))
    out = np.empty(len(ids))
    for i, t in enumerate(ids):
        t = int(t)
        f = fbs_now.get(t, False)
        mean = p.fbs_start if f else p.fcs_start
        if t not in ratings:
            out[i] = mean
        elif season_of[t] != season:
            out[i] = ratings[t] + p.regress * (mean - ratings[t])
        else:
            out[i] = ratings[t]
    return pd.Series(out, index=ids)

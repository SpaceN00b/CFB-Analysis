"""Home and road performance against expectation.

Expectation: a margin model fitted at each cutoff on the visible FBS vs FBS games of the
current and previous three seasons, home margin = a + hfa * on_campus + b * (Elo gap),
with CFBD pregame Elo. A team's home (road) residual is its mean margin above that
expectation in its on-campus home (road) games of the same window, shrunk toward zero
(the league value) as sum / (n + SHRINK_GAMES).

SHRINK_GAMES comes from seasons 2014-2025 (scripts/context_check.py): the between-team
variance of 4-season home residual means is about 1.9 points^2 against a per-game
residual variance of 16.7^2, so k = 16.7^2 / 1.9 = 146. Split-half reliability of the
team values is about zero (docs/context.md); W8 may drop them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup

WINDOW_SEASONS = 4
SHRINK_GAMES = 146.0
MIN_GAMES_FOR_FIT = 200
SOURCES = ("team_game",)


def residual_games(view) -> tuple[pd.DataFrame, dict]:
    """Visible FBS vs FBS games in the window with their residual (home perspective) and the fit."""
    seasons = list(range(view.season - WINDOW_SEASONS + 1, view.season + 1))
    tg = view.table("team_game", seasons=seasons,
                    columns=["game_id", "team_id", "opp_id", "is_home", "neutral", "team_fbs", "opp_fbs",
                             "margin", "pregame_elo"])
    tg = tg[tg["team_fbs"].fillna(False).astype(bool) & tg["opp_fbs"].fillna(False).astype(bool)]
    home = tg[tg["is_home"].fillna(False).astype(bool)]
    away = tg[~tg["is_home"].fillna(False).astype(bool)][["game_id", "pregame_elo"]].rename(
        columns={"pregame_elo": "opp_elo"})
    g = home.merge(away, on="game_id", how="inner")
    g = g[g["margin"].notna() & g["pregame_elo"].notna() & g["opp_elo"].notna()].reset_index(drop=True)
    fit = {"hfa": np.nan, "n": len(g)}
    if len(g) < MIN_GAMES_FOR_FIT:
        g["resid"] = np.nan
        return g, fit
    on_campus = (~g["neutral"].fillna(False).astype(bool)).astype("float64").to_numpy()
    X = np.column_stack([np.ones(len(g)), on_campus, (g["pregame_elo"] - g["opp_elo"]).to_numpy(dtype="float64") / 100])
    y = g["margin"].to_numpy(dtype="float64")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    g["resid"] = y - X @ beta
    fit["hfa"] = float(beta[1])
    return g, fit


class HomeRoad(FeatureGroup):
    name = "ctx_home_road"
    prefix = "ctx_"
    level = "team"
    columns = [
        FeatureColumn("ctx_home_resid", "Team's mean home margin above expectation (Elo + league home edge), "
                      f"last {WINDOW_SEASONS} seasons, shrunk toward 0 (points)", SOURCES, 2014),
        FeatureColumn("ctx_road_resid", "Team's mean road margin above expectation, same window and shrinkage "
                      "(points)", SOURCES, 2014),
        FeatureColumn("ctx_home_games_n", "On-campus FBS home games behind ctx_home_resid", SOURCES, 2014),
        FeatureColumn("ctx_road_games_n", "On-campus FBS road games behind ctx_road_resid", SOURCES, 2014),
    ]

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        g, _ = residual_games(view)
        g = g[~g["neutral"].fillna(False).astype(bool) & g["resid"].notna()]
        home = g.groupby("team_id")["resid"].agg(["sum", "count"])
        road = (-g["resid"]).groupby(g["opp_id"]).agg(["sum", "count"])
        ids = pd.Series(games["home_id"].tolist() + games["away_id"].tolist())
        gid = games["game_id"].tolist() * 2

        def shrunk(table):
            s = ids.map(table["sum"]).fillna(0.0)
            n = ids.map(table["count"]).fillna(0.0)
            return (s / (n + SHRINK_GAMES)).to_numpy(), n.to_numpy(dtype="float64")

        h, hn = shrunk(home)
        r, rn = shrunk(road)
        out = pd.DataFrame({"game_id": gid, "team_id": ids.to_numpy(), "ctx_home_resid": h, "ctx_road_resid": r,
                            "ctx_home_games_n": hn, "ctx_road_games_n": rn})
        if g.empty:  # nothing to fit on yet: unknown, not zero
            out[["ctx_home_resid", "ctx_road_resid"]] = np.nan
        return out.drop_duplicates(["game_id", "team_id"])


class LeagueHomeField(FeatureGroup):
    name = "ctx_home_field"
    prefix = "ctx_"
    level = "game"
    columns = [
        FeatureColumn("ctx_league_hfa", f"League home edge (points) fitted at the cutoff on FBS vs FBS games of the "
                      f"last {WINDOW_SEASONS} seasons, controlling for the Elo gap", SOURCES, 2014),
    ]

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        _, fit = residual_games(view)
        return pd.DataFrame({"game_id": games["game_id"].to_numpy(), "ctx_league_hfa": fit["hfa"]}).drop_duplicates(
            "game_id")

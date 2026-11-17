"""Crowd proxy: a team's home attendance in its last normal season and its stadium size.

A game's own attendance is a result, so it is never used. 2020 (limited or no crowds) is
skipped as the reference season: 2021 games look back to 2019.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup
from ._util import team_homes, venue_table

SKIP_SEASONS = (2020,)
SOURCES = ("games", "venues", "teams_fbs")


def reference_season(season: int) -> int:
    ref = season - 1
    while ref in SKIP_SEASONS:
        ref -= 1
    return ref


class Crowd(FeatureGroup):
    name = "ctx_crowd"
    prefix = "ctx_"
    level = "team"
    columns = [
        FeatureColumn("ctx_crowd_prior_att", "Mean attendance (thousands) at the team's on-campus home games in "
                      "the previous season (2019 for 2021)", SOURCES, 2015),
        FeatureColumn("ctx_crowd_prior_fill", "Same games: attendance over venue capacity", SOURCES, 2015),
        FeatureColumn("ctx_crowd_season_att", "Mean attendance (thousands) at the team's on-campus home games "
                      "already played this season", SOURCES, 2014),
        FeatureColumn("ctx_crowd_capacity", "Capacity of the team's home stadium (thousands)", SOURCES, 2014),
    ]

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        venues = venue_table(view)
        homes = team_homes(view, venues)
        ref = reference_season(view.season)
        g = view.table("games", columns=["id", "season", "neutralSite", "homeId", "venueId", "attendance"],
                       seasons=[ref, view.season])
        g = g[~g["neutralSite"].fillna(False).astype(bool)].copy()
        g["att"] = pd.to_numeric(g["attendance"], errors="coerce").astype("float64")
        g.loc[g["att"] <= 0, "att"] = np.nan
        g = g.dropna(subset=["att"])
        cap = venues.set_index("venue_id")["capacity"]
        g["cap"] = pd.to_numeric(g["venueId"], errors="coerce").astype("float64").map(cap)
        g["fill"] = g["att"] / g["cap"]
        prior = g[g["season"] == ref].groupby("homeId")[["att", "fill"]].mean()
        now = g[g["season"] == view.season].groupby("homeId")["att"].mean()
        ids = pd.Series(games["home_id"].tolist() + games["away_id"].tolist())
        capacity = homes.set_index("team_id")["capacity"]
        out = pd.DataFrame({
            "game_id": games["game_id"].tolist() * 2,
            "team_id": ids.to_numpy(),
            "ctx_crowd_prior_att": (ids.map(prior["att"]) / 1000).to_numpy(dtype="float64"),
            "ctx_crowd_prior_fill": ids.map(prior["fill"]).to_numpy(dtype="float64"),
            "ctx_crowd_season_att": (ids.map(now) / 1000).to_numpy(dtype="float64"),
            "ctx_crowd_capacity": (ids.astype("float64").map(capacity) / 1000).to_numpy(dtype="float64"),
        })
        return out.drop_duplicates(["game_id", "team_id"])

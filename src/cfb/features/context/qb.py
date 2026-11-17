"""Quarterback continuity from player box scores (passing C/ATT), the v1 substitute for
injury data. The passer with the most attempts in a game "led" it; team rows ("Team",
negative athlete ids) are dropped. Counts cover the team's games of this season whose
box score is visible at the cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup

MIN_ATTEMPTS = 10
SOURCES = ("games_players", "schedule")


def passing_by_game(view) -> pd.DataFrame:
    """game_id, team_id, athlete, attempts, start_time_utc for the cutoff season."""
    gp = view.table("games_players", seasons=view.season,
                    columns=["gameId", "homeAway", "category", "statType", "athleteId", "stat"])
    gp = gp[(gp["category"] == "passing") & (gp["statType"] == "C/ATT")]
    gp = gp[gp["athleteId"].notna() & ~gp["athleteId"].astype(str).str.startswith("-")]
    att = pd.to_numeric(gp["stat"].astype(str).str.split("/").str[1], errors="coerce")
    sched = view.schedule()
    sched = sched[sched["season"] == view.season][["game_id", "home_id", "away_id", "start_time_utc"]]
    out = pd.DataFrame({"game_id": gp["gameId"].to_numpy(), "side": gp["homeAway"].to_numpy(),
                        "athlete": gp["athleteId"].astype(str).to_numpy(), "att": att.to_numpy(dtype="float64")})
    out = out.dropna(subset=["att"]).merge(sched, on="game_id", how="inner")
    out["team_id"] = np.where(out["side"] == "home", out["home_id"], out["away_id"])
    out = out.groupby(["game_id", "team_id", "athlete"], as_index=False).agg(
        att=("att", "sum"), start_time_utc=("start_time_utc", "first"))
    return out


class QBContinuity(FeatureGroup):
    name = "ctx_qb"
    prefix = "ctx_"
    level = "team"
    columns = [
        FeatureColumn("ctx_qb_leader_share", "Share of the team's season pass attempts thrown by the passer who "
                      "led its last game", SOURCES, 2014),
        FeatureColumn("ctx_qb_changed", "1 if the leading passer of the last game differs from the one of the "
                      "game before", SOURCES, 2014),
        FeatureColumn("ctx_qb_passers_10", f"Passers with {MIN_ATTEMPTS} or more attempts this season",
                      SOURCES, 2014),
        FeatureColumn("ctx_qb_leader_games", "Games this season led in attempts by the passer who led the last game",
                      SOURCES, 2014),
        FeatureColumn("ctx_qb_games_seen", "Team games this season with a visible passing box score", SOURCES, 2014),
    ]

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        p = passing_by_game(view)
        ids = pd.Series(games["home_id"].tolist() + games["away_id"].tolist())
        gid = games["game_id"].tolist() * 2
        start = pd.Series(games["start_time_utc"].tolist() * 2)
        out = pd.DataFrame({"game_id": gid, "team_id": ids.to_numpy()})
        for c in self.column_names:
            out[c] = np.nan
        if p.empty:
            return out.drop_duplicates(["game_id", "team_id"])
        p = p.sort_values(["team_id", "start_time_utc", "game_id", "att", "athlete"],
                          ascending=[True, True, True, False, True])
        leaders = p.drop_duplicates(["game_id", "team_id"])[["game_id", "team_id", "athlete", "start_time_utc"]]
        stats = dict(tuple(leaders.groupby("team_id", sort=False)))
        for i, team in enumerate(ids):
            lead = stats.get(team)
            if lead is None:
                continue
            lead = lead[lead["start_time_utc"] < start.iat[i]]
            if lead.empty:
                continue
            # Box scores of later games are not visible at the cutoff, so every visible
            # game is earlier than the slate. The filter only matters for a team with
            # two games on one slate (week 0 inside week 1).
            last = lead["athlete"].iat[-1]
            sub = p[(p["team_id"] == team) & p["game_id"].isin(lead["game_id"])]
            per = sub.groupby("athlete")["att"].sum()
            out.iat[i, 2] = float(per.get(last, 0.0) / per.sum()) if per.sum() > 0 else np.nan
            out.iat[i, 3] = float(lead["athlete"].iat[-1] != lead["athlete"].iat[-2]) if len(lead) >= 2 else np.nan
            out.iat[i, 4] = float((per >= MIN_ATTEMPTS).sum())
            out.iat[i, 5] = float((lead["athlete"] == last).sum())
            out.iat[i, 6] = float(len(lead))
        return out.drop_duplicates(["game_id", "team_id"])

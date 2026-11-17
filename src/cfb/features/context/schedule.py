"""Rest, schedule density and season phase. Inputs: the schedule only (no results)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup
from ._util import _at, DAY, history_with_targets, site

BYE_DAYS = 13.0  # rest of 13 days or more: the team had a week off (effects.py used the same rule)
SHORT_WEEK_DAYS = 5.5


class ScheduleContext(FeatureGroup):
    name = "ctx_schedule"
    prefix = "ctx_"
    level = "team"
    columns = [
        FeatureColumn("ctx_site", "+1 at home, -1 on the road, 0 at a neutral site", ("schedule",), 2014),
        FeatureColumn("ctx_rest_days", "Days since the team's previous game this season (empty for its first game)",
                      ("schedule",), 2014),
        FeatureColumn("ctx_off_bye", f"1 if rest is {BYE_DAYS:.0f} days or more (coming off a bye)",
                      ("schedule",), 2014),
        FeatureColumn("ctx_short_week", f"1 if rest is {SHORT_WEEK_DAYS} days or less", ("schedule",), 2014),
        FeatureColumn("ctx_games_last_14d", "Games the team played in the 14 days before kickoff", ("schedule",), 2014),
        FeatureColumn("ctx_game_number", "Game number in the team's season, 1 for its first game", ("schedule",), 2014),
        FeatureColumn("ctx_days_since_regular", "Postseason only: days since the team's last regular-season game",
                      ("schedule",), 2014),
    ]

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        rows = history_with_targets(view, games, seasons=games["season"].unique())
        tgt = rows[rows["_target"]][["game_id", "team_id", "season", "season_type", "start_time_utc", "neutral", "is_home"]]
        tgt = tgt.reset_index(drop=True)
        tgt["_k"] = np.arange(len(tgt))
        pair = tgt[["_k", "team_id", "season", "start_time_utc"]].merge(
            rows[["team_id", "season", "season_type", "start_time_utc"]].rename(
                columns={"start_time_utc": "t", "season_type": "st"}),
            on=["team_id", "season"], how="left")
        before = pair[pair["t"] < pair["start_time_utc"]]
        prev = before.groupby("_k")["t"].max()
        n_before = before.groupby("_k").size()
        recent = before[before["t"] >= before["start_time_utc"] - 14 * DAY].groupby("_k").size()
        last_reg = before[before["st"] == "regular"].groupby("_k")["t"].max()

        k = tgt["_k"]
        rest = (tgt["start_time_utc"] - _at(prev, k)).dt.total_seconds() / 86400.0
        post = tgt["season_type"] != "regular"
        since_reg = (tgt["start_time_utc"] - _at(last_reg, k)).dt.total_seconds() / 86400.0
        out = pd.DataFrame({
            "game_id": tgt["game_id"],
            "team_id": tgt["team_id"],
            "ctx_site": site(tgt),
            "ctx_rest_days": rest.astype("float64"),
            "ctx_off_bye": (rest >= BYE_DAYS).astype("float64").where(rest.notna()),
            "ctx_short_week": (rest <= SHORT_WEEK_DAYS).astype("float64").where(rest.notna()),
            "ctx_games_last_14d": _at(recent, k).fillna(0).astype("float64"),
            "ctx_game_number": (_at(n_before, k).fillna(0) + 1).astype("float64"),
            "ctx_days_since_regular": since_reg.where(post).astype("float64"),
        })
        return out.drop_duplicates(["game_id", "team_id"])


class SeasonPhase(FeatureGroup):
    name = "ctx_season_phase"
    prefix = "ctx_"
    level = "game"
    columns = [
        FeatureColumn("ctx_postseason", "1 for bowls and playoff games", ("schedule",), 2014),
        FeatureColumn("ctx_neutral", "1 at a neutral site", ("schedule",), 2014),
        FeatureColumn("ctx_conference_game", "1 for a conference game", ("schedule",), 2014),
        FeatureColumn("ctx_start_time_tbd", "1 if the kickoff time was still to be decided", ("schedule",), 2014),
        FeatureColumn("ctx_season_week", "Week of the season counted from the first calendar week, by kickoff date "
                      "(continues through the postseason; week 0 games count as week 1)", ("weeks", "schedule"), 2014),
        FeatureColumn("ctx_season_2020", "1 in the 2020 season (limited crowds, irregular schedules)", ("schedule",), 2014),
    ]

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        weeks = view.table("weeks")
        start = weeks[weeks["season_type"] == "regular"].groupby("season")["start_utc"].min()
        first = games["season"].map(start)
        week = np.floor((games["start_time_utc"] - first).dt.total_seconds() / (7 * 86400.0)) + 1
        week = week.clip(lower=1)
        flag = lambda s: pd.Series(s).fillna(False).astype(bool).astype("float64").to_numpy()
        return pd.DataFrame({
            "game_id": games["game_id"].to_numpy(),
            "ctx_postseason": (games["season_type"] != "regular").astype("float64").to_numpy(),
            "ctx_neutral": flag(games["neutral"]),
            "ctx_conference_game": flag(games["conference_game"]) if "conference_game" in games else np.nan,
            "ctx_start_time_tbd": flag(games["start_time_tbd"]) if "start_time_tbd" in games else np.nan,
            "ctx_season_week": week.astype("float64").to_numpy(),
            "ctx_season_2020": (games["season"] == 2020).astype("float64").to_numpy(),
        }).drop_duplicates("game_id")

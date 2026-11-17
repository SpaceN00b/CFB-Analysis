"""Which games a weekly run is about: the target week, the week before it, and the moment
("as of") the report describes.

Regular season: the CFBD calendar week. Its cutoff is the start of the week, the report is
as of that cutoff, and "last week" is the latest earlier week with games.

Postseason: CFBD files every bowl and playoff round under postseason week 1, each game with
its own cutoff (kickoff minus 24 hours). A run with a known `now` takes the postseason games
that kick off in the next `window` days and calls the games of the previous `window` days
"last week"; without `now` (a historical report) it takes the whole postseason.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.timeutil import RESULT_LAG, as_utc

WINDOW = pd.Timedelta(days=8)


@dataclass(frozen=True)
class Slate:
    season: int
    season_type: str
    week: int
    asof: pd.Timestamp            # results visible at this moment are "known" in the report
    games: pd.DataFrame           # schedule rows of the target games, kickoff order
    previous: pd.DataFrame        # schedule rows of last week's games
    previous_label: str

    @property
    def label(self) -> str:
        if self.season_type == "postseason":
            return f"{self.season} postseason"
        return f"{self.season} week {self.week}"

    @property
    def key(self) -> str:
        return f"{self.season}_{self.season_type}_{self.week:02d}"

    @property
    def last_cutoff(self) -> pd.Timestamp:
        return self.games["cutoff_utc"].max() if len(self.games) else self.asof


def current_week(weeks: pd.DataFrame, now: pd.Timestamp) -> tuple[int, str, int]:
    """The calendar week containing `now`; between seasons the next week to start, or else
    the last week on the calendar."""
    w = weeks.sort_values("start_utc")
    inside = w[(w["start_utc"] <= now) & (w["end_utc"] >= now)]
    if len(inside):
        row = inside.iloc[-1]
    elif (w["start_utc"] > now).any():
        row = w[w["start_utc"] > now].iloc[0]
    else:
        row = w.iloc[-1]
    return int(row["season"]), str(row["season_type"]), int(row["week"])


def resolve_slate(warehouse, season: int | None = None, season_type: str | None = None,
                  week: int | None = None, now=None, window: pd.Timedelta = WINDOW) -> Slate:
    """Target games for a run. Missing season/week come from the calendar at `now`
    (default: the current time)."""
    sched = warehouse.schedule()
    now_ts = as_utc(now) if now is not None else None
    if season is None or week is None:
        at = now_ts if now_ts is not None else pd.Timestamp.now(tz="UTC")
        s, t, w = current_week(warehouse.meta["weeks"], at)
        season = s if season is None else season
        season_type = season_type or t
        week = w if week is None else week
        if now_ts is None:
            now_ts = at
    season_type = season_type or "regular"
    season_games = sched[sched["season"] == season]
    week_games = season_games[(season_games["season_type"] == season_type) & (season_games["week"] == week)]

    if season_type == "postseason" and now_ts is not None:
        games = week_games[(week_games["start_time_utc"] >= now_ts) & (week_games["start_time_utc"] < now_ts + window)]
        prev = season_games[(season_games["start_time_utc"] < now_ts)
                            & (season_games["start_time_utc"] >= now_ts - window)]
        asof = now_ts
        prev_label = f"games of the last {window.days} days"
        if prev.empty:
            prev, prev_label = _last_regular(season_games)
    else:
        games = week_games
        if season_type == "postseason":
            asof = games["cutoff_utc"].min() if len(games) else now_ts
            prev, prev_label = _last_regular(season_games)
        else:
            asof = games["cutoff_utc"].min() if len(games) else _week_start(warehouse, season, season_type, week)
            earlier = season_games[(season_games["season_type"] == "regular") & (season_games["week"] < week)]
            if len(earlier):
                pw = int(earlier["week"].max())
                prev, prev_label = earlier[earlier["week"] == pw], f"{season} week {pw}"
            else:
                prev, prev_label = earlier, "none (first week of the season)"
    if asof is None or pd.isna(asof):
        raise ValueError(f"no games and no calendar week for {season} {season_type} week {week}")
    return Slate(int(season), str(season_type), int(week), as_utc(asof),
                 games.sort_values(["start_time_utc", "game_id"]).reset_index(drop=True),
                 prev.sort_values(["start_time_utc", "game_id"]).reset_index(drop=True), prev_label)


def _last_regular(season_games: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    reg = season_games[season_games["season_type"] == "regular"]
    if reg.empty:
        return reg, "none"
    pw = int(reg["week"].max())
    return reg[reg["week"] == pw], f"{int(reg['season'].iloc[0])} week {pw}"


def _week_start(warehouse, season, season_type, week):
    w = warehouse.meta["weeks"]
    hit = w[(w["season"] == season) & (w["season_type"] == season_type) & (w["week"] == week)]
    return hit["start_utc"].iloc[0] if len(hit) else None


def known_results(outcomes: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Completed games whose results were visible at `asof` (kickoff + RESULT_LAG)."""
    return outcomes[outcomes["start_time_utc"] + RESULT_LAG <= asof]


__all__ = ["Slate", "WINDOW", "current_week", "known_results", "resolve_slate"]

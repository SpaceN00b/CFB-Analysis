"""game_index and its companions (weeks, teams, game_pregame), built from the raw store.

game_index is one row per game with the cutoff that governs what a prediction for
that game may know:

    regular season   start of the game's calendar week (week N sees data through N-1)
    postseason       24 hours before kickoff (CFBD files every bowl and playoff round
                     under postseason week 1, so a weekly cutoff would hide the semifinal
                     from the final)
"""
from __future__ import annotations

import pandas as pd

from ..store import Store
from .timeutil import POSTSEASON_LEAD, to_utc_series

REGULAR, POSTSEASON = "regular", "postseason"
SEASON_TYPES = (REGULAR, POSTSEASON)

# The contract columns, in order. Extras follow them in the stored table.
GAME_INDEX_COLUMNS = [
    "game_id", "season", "season_type", "week", "start_time_utc", "cutoff_utc",
    "neutral", "conference_game", "venue_id", "home_id", "away_id",
    "home_fbs", "away_fbs", "completed", "home_points", "away_points",
]
GAME_INDEX_EXTRA = ["home_team", "away_team", "start_time_tbd"]
# Columns that hold a game's outcome. Never shown for games a view cannot see yet.
RESULT_COLUMNS = ["completed", "home_points", "away_points"]
SCHEDULE_COLUMNS = [c for c in GAME_INDEX_COLUMNS + GAME_INDEX_EXTRA if c not in RESULT_COLUMNS]

WEEKS_COLUMNS = ["season", "season_type", "week", "start_utc", "end_utc"]
TEAMS_COLUMNS = ["season", "team_id", "team", "conference", "fbs"]
GAME_PREGAME_COLUMNS = ["game_id", "home_pregame_elo", "away_pregame_elo"]

_INDEPENDENT = {"FBS Independents", "FCS Independents"}


def _raw_games(store: Store) -> pd.DataFrame:
    games = store.load("games")
    if games.empty:
        raise ValueError("raw 'games' dataset is empty")
    games = games.drop_duplicates("id", keep="last").copy()
    games["start_time_utc"] = to_utc_series(games["startDate"])
    if "seasonType" not in games.columns:
        games["seasonType"] = games["season_type"]
    return games


def build_weeks(store: Store, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calendar windows per (season, season_type, week).

    Taken from CFBD /calendar. A week that has games but no calendar row gets a
    window derived from its games: 24 hours before the first kickoff to 12 hours
    after the last.
    """
    games = _raw_games(store) if games is None else games
    try:
        cal = store.load("calendar")
    except FileNotFoundError:
        cal = pd.DataFrame()
    if not cal.empty:
        cal = cal[cal["seasonType"].isin(SEASON_TYPES)]
        cal = cal.reset_index(drop=True)
        weeks = pd.DataFrame({
            "season": cal["season"].astype("int64"),
            "season_type": cal["seasonType"].astype("str"),
            "week": cal["week"].astype("int64"),
            "start_utc": to_utc_series(cal["startDate"]),
            "end_utc": to_utc_series(cal["endDate"]),
        })
    else:
        weeks = pd.DataFrame({c: [] for c in WEEKS_COLUMNS})

    seen = games.groupby(["season", "seasonType", "week"])["start_time_utc"].agg(["min", "max"]).reset_index()
    seen = seen.rename(columns={"seasonType": "season_type"})
    merged = seen.merge(weeks, on=["season", "season_type", "week"], how="left")
    missing = merged[merged["start_utc"].isna()]
    if len(missing):
        fallback = pd.DataFrame({
            "season": missing["season"].astype("int64"),
            "season_type": missing["season_type"].astype("str"),
            "week": missing["week"].astype("int64"),
            "start_utc": missing["min"] - pd.Timedelta(hours=24),
            "end_utc": missing["max"] + pd.Timedelta(hours=12),
        })
        weeks = pd.concat([weeks, fallback], ignore_index=True) if len(weeks) else fallback
    weeks = weeks.drop_duplicates(["season", "season_type", "week"], keep="first")
    weeks["_type_order"] = weeks["season_type"].map({REGULAR: 0, POSTSEASON: 1})
    weeks = weeks.sort_values(["season", "_type_order", "week"]).drop(columns="_type_order")
    weeks["start_utc"] = weeks["start_utc"].dt.as_unit("ns")
    weeks["end_utc"] = weeks["end_utc"].dt.as_unit("ns")
    return weeks[WEEKS_COLUMNS].reset_index(drop=True)


def compute_cutoff(start: pd.Series, season_type: pd.Series, week_start: pd.Series) -> pd.Series:
    """Prediction cutoff per game. See the module docstring for the rule.

    A regular-season game that kicks off before its calendar week starts (none in
    2014 to 2026) falls back to kickoff minus 24 hours so the cutoff never trails kickoff.
    """
    lead = start - POSTSEASON_LEAD
    weekly_ok = (season_type == REGULAR) & week_start.notna() & (week_start <= start)
    return lead.where(~weekly_ok, week_start).dt.as_unit("ns")


def build_game_index(store: Store, weeks: pd.DataFrame | None = None) -> pd.DataFrame:
    games = _raw_games(store)
    weeks = build_weeks(store, games) if weeks is None else weeks
    g = games.merge(
        weeks.rename(columns={"season_type": "seasonType", "start_utc": "_week_start"})[
            ["season", "seasonType", "week", "_week_start"]
        ],
        on=["season", "seasonType", "week"],
        how="left",
    )
    home_conf, away_conf = g.get("homeConference"), g.get("awayConference")
    same_conf = pd.Series(False, index=g.index)
    if home_conf is not None and away_conf is not None:
        same_conf = (home_conf == away_conf) & home_conf.notna() & ~home_conf.isin(_INDEPENDENT)
    out = pd.DataFrame({
        "game_id": g["id"].astype("int64"),
        "season": g["season"].astype("int64"),
        "season_type": g["seasonType"].astype("str"),
        "week": g["week"].astype("int64"),
        "start_time_utc": g["start_time_utc"],
        "cutoff_utc": compute_cutoff(g["start_time_utc"], g["seasonType"], g["_week_start"]),
        "neutral": g["neutralSite"].fillna(False).astype(bool),
        # The raw flag misses some games (12 in 2025, title games among them), so a
        # shared conference also counts.
        "conference_game": g["conferenceGame"].fillna(False).astype(bool) | same_conf.fillna(False).astype(bool),
        "venue_id": pd.to_numeric(g["venueId"], errors="coerce").astype("Int64"),
        "home_id": g["homeId"].astype("int64"),
        "away_id": g["awayId"].astype("int64"),
        "home_fbs": (g["homeClassification"] == "fbs").fillna(False).astype(bool),
        "away_fbs": (g["awayClassification"] == "fbs").fillna(False).astype(bool),
        "completed": g["completed"].fillna(False).astype(bool),
        "home_points": pd.to_numeric(g["homePoints"], errors="coerce").astype("float64"),
        "away_points": pd.to_numeric(g["awayPoints"], errors="coerce").astype("float64"),
        "home_team": g["homeTeam"].astype("str"),
        "away_team": g["awayTeam"].astype("str"),
        "start_time_tbd": g["startTimeTBD"].fillna(False).astype(bool) if "startTimeTBD" in g else False,
    })
    out = out.sort_values(["start_time_utc", "game_id"]).reset_index(drop=True)
    if out["start_time_utc"].isna().any():
        raise ValueError("games without a start time cannot be indexed")
    if (out["cutoff_utc"] > out["start_time_utc"]).any():
        raise AssertionError("cutoff after kickoff")
    return out[GAME_INDEX_COLUMNS + GAME_INDEX_EXTRA]


def build_teams(store: Store) -> pd.DataFrame:
    """One row per (season, team) that appears in a game: name, conference, FBS flag."""
    g = _raw_games(store)
    sides = []
    for side in ("home", "away"):
        sides.append(pd.DataFrame({
            "season": g["season"].astype("int64"),
            "team_id": g[f"{side}Id"].astype("int64"),
            "team": g[f"{side}Team"].astype("str"),
            "conference": g[f"{side}Conference"] if f"{side}Conference" in g else None,
            "fbs": (g[f"{side}Classification"] == "fbs").fillna(False).astype(bool),
        }))
    teams = pd.concat(sides, ignore_index=True)
    teams = teams.sort_values(["season", "team_id", "conference"], na_position="last")
    teams = teams.drop_duplicates(["season", "team_id"], keep="first")
    return teams[TEAMS_COLUMNS].reset_index(drop=True)


def build_game_pregame(store: Store) -> pd.DataFrame:
    """Pregame facts CFBD attaches to the game row: point-in-time Elo for both teams."""
    g = _raw_games(store)
    out = pd.DataFrame({
        "game_id": g["id"].astype("int64"),
        "home_pregame_elo": pd.to_numeric(g.get("homePregameElo"), errors="coerce").astype("float64"),
        "away_pregame_elo": pd.to_numeric(g.get("awayPregameElo"), errors="coerce").astype("float64"),
    })
    return out.sort_values("game_id").reset_index(drop=True)


def season_bounds(weeks: pd.DataFrame) -> pd.DataFrame:
    """Per season: first week start and last week end, indexed by season."""
    b = weeks.groupby("season").agg(start_utc=("start_utc", "min"), end_utc=("end_utc", "max"))
    return b.sort_index()


def schedule_view(game_index: pd.DataFrame) -> pd.DataFrame:
    """game_index without its result columns."""
    return game_index[[c for c in game_index.columns if c not in RESULT_COLUMNS]]


__all__ = [
    "GAME_INDEX_COLUMNS", "GAME_INDEX_EXTRA", "RESULT_COLUMNS", "SCHEDULE_COLUMNS",
    "WEEKS_COLUMNS", "TEAMS_COLUMNS", "GAME_PREGAME_COLUMNS", "REGULAR", "POSTSEASON",
    "build_weeks", "build_game_index", "build_teams", "build_game_pregame",
    "compute_cutoff", "season_bounds", "schedule_view",
]

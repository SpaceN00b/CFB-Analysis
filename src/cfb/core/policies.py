"""Access policy per table: when does a row become known?

AsOfView refuses any table without a policy here. To add a table, add a policy;
feature threads ask the W1 owner instead of registering their own.

Kinds:
    GAME          a game's results. Visible once the game kicked off at least
                  RESULT_LAG (5 hours) before the cutoff.
    PREGAME       facts fixed before kickoff (lines, media, weather, pregame Elo).
                  Visible for every game whose own cutoff is at or before the view's:
                  past games and the slate being predicted.
    PRESEASON     one row per season, known before that season's first game.
                  Visible for seasons up to the cutoff season.
    SEASON_FINAL  year-level values as of the pull date (SP+, FPI, SRS, records,
                  season stats). Same-season values contain later results, so
                  view.table() raises; view.prior_seasons() returns seasons that
                  ended before the cutoff.
    DATED         rows carry their own timestamp. Visible when it is <= cutoff.
    WEEKLY        rows keyed by season, season type and week. "pre" rows are known
                  when that calendar week starts, "post" rows when it ends.
    STATIC        reference data with no time dimension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

GAME, PREGAME, PRESEASON, SEASON_FINAL, DATED, WEEKLY, STATIC = (
    "game", "pregame", "preseason", "season_final", "dated", "weekly", "static",
)
KINDS = (GAME, PREGAME, PRESEASON, SEASON_FINAL, DATED, WEEKLY, STATIC)

# Columns the leak scrambler leaves alone when they are not listed explicitly:
# identifiers and schedule facts, never outcomes.
DEFAULT_KEYS = frozenset({
    "id", "gameId", "game_id", "team", "teamId", "team_id", "opp_id", "opponent", "is_home",
    "season", "year", "season_type", "seasonType", "week", "home_id", "away_id",
    "homeId", "awayId", "homeTeam", "awayTeam", "offense", "defense", "home", "away",
    "offense_id", "defense_id", "start_time_utc", "cutoff_utc", "unit",
})


# Player identity columns in the player-level tables (never outcomes).
PLAYER_KEYS = ("playerId", "athleteId", "collegeAthleteId", "player", "name", "firstName", "lastName",
               "position", "conference", "collegeTeam", "collegeId", "committedTo", "recruitIds")


@dataclass(frozen=True)
class TablePolicy:
    kind: str
    game_col: str | None = None  # GAME, PREGAME
    season_col: str | None = None  # PRESEASON, SEASON_FINAL, WEEKLY; also used by seasons=
    date_col: str | None = None  # DATED
    week_col: str = "week"  # WEEKLY
    season_type_col: str | None = None  # WEEKLY; None means every row is regular season
    weekly: dict = field(default_factory=lambda: {"regular": "pre", "postseason": "pre"})
    hide: tuple[str, ...] = ()  # columns removed entirely ("prefix*" allowed)
    keys: tuple[str, ...] = ()  # extra columns the leak scrambler must not touch
    transform: Callable | None = None  # (frame, view) -> frame, applied to every read
    note: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown policy kind {self.kind!r}")
        if self.kind in (GAME, PREGAME) and not self.game_col:
            raise ValueError(f"{self.kind} policy needs game_col")
        if self.kind in (PRESEASON, SEASON_FINAL, WEEKLY) and not self.season_col:
            raise ValueError(f"{self.kind} policy needs season_col")
        if self.kind == DATED and not self.date_col:
            raise ValueError("dated policy needs date_col")

    def hidden(self, columns) -> list[str]:
        out = []
        for col in columns:
            for pat in self.hide:
                if col == pat or (pat.endswith("*") and col.startswith(pat[:-1])):
                    out.append(col)
                    break
        return out


def _mask_portal_destination(frame: pd.DataFrame, view) -> pd.DataFrame:
    """A portal entry's destination is filled in later. Show it only once the
    portal season has started; before that the player had not enrolled anywhere."""
    if "destination" not in frame.columns or "season" not in frame.columns or frame.empty:
        return frame
    out = frame.copy()
    out["destination"] = out["destination"].where(out["season"] <= view.season)
    return out


def _coach_tenures(frame: pd.DataFrame, view) -> pd.DataFrame:
    """Show a tenure once it has begun: hired on or before the cutoff (for a start year
    no later than next season), or, without a hire date, a start year up to the cutoff
    season. An end that had not happened yet is blanked."""
    if frame.empty:
        return frame
    started = pd.Series(True, index=frame.index)
    if "startYear" in frame.columns:
        start_year = pd.to_numeric(frame["startYear"], errors="coerce")
        started = start_year <= view.season
        if "hireDate" in frame.columns:
            hired = pd.to_datetime(frame["hireDate"], utc=True, errors="coerce", format="ISO8601")
            started = started.where(hired.isna(), (hired <= view.cutoff) & (start_year <= view.season + 1))
    out = frame[started.fillna(False).to_numpy()].copy()
    if "endYear" in out.columns:
        end = pd.to_numeric(out["endYear"], errors="coerce")
        out["endYear"] = end.where(end < view.season)
    if "effectiveEnd" in out.columns:
        eff = pd.to_datetime(out["effectiveEnd"], utc=True, errors="coerce", format="ISO8601")
        out["effectiveEnd"] = out["effectiveEnd"].where(eff <= view.cutoff)
    return out


POLICIES: dict[str, TablePolicy] = {
    # ---- core tables (data/core) -------------------------------------------------
    "game_index": TablePolicy(GAME, game_col="game_id", season_col="season", keys=(
        "neutral", "conference_game", "venue_id", "home_fbs", "away_fbs",
        "home_team", "away_team", "start_time_tbd",
    )),
    "team_game": TablePolicy(GAME, game_col="game_id", season_col="season",
                             keys=("neutral", "team_fbs", "opp_fbs")),
    "pbp": TablePolicy(GAME, game_col="game_id", season_col="season",
                       keys=("play_id", "drive_id", "offense_is_home")),
    "play_detail": TablePolicy(GAME, game_col="game_id", season_col="season",
                               keys=("play_id", "text_format"),
                               note="fields parsed from play text (cfb.core.playtext)"),
    "play_players": TablePolicy(GAME, game_col="game_id", season_col="season",
                                keys=("play_id", "role", "name", "jersey", "side", "athlete_id", "match"),
                                note="named players per play and role (cfb.core.playtext)"),
    "play_penalties": TablePolicy(GAME, game_col="game_id", season_col="season",
                                  keys=("play_id", "n", "team_text", "side", "name", "jersey", "athlete_id", "match"),
                                  note="penalties per play (cfb.core.playtext)"),
    "game_pregame": TablePolicy(PREGAME, game_col="game_id",
                                note="pregame Elo from the game row; point-in-time"),
    "teams": TablePolicy(PRESEASON, season_col="season", keys=("conference", "fbs")),
    "weeks": TablePolicy(STATIC, season_col="season"),
    # ---- derived tables (data/derived) -------------------------------------------
    "unit_ratings": TablePolicy(DATED, date_col="cutoff_utc"),
    # ---- raw: game results ---------------------------------------------------------
    "games": TablePolicy(GAME, game_col="id", season_col="season", keys=(
        "startDate", "startTimeTBD", "neutralSite", "conferenceGame", "venueId", "venue",
        "homeClassification", "homeConference", "awayClassification", "awayConference",
    )),
    "drives": TablePolicy(GAME, game_col="gameId", season_col="season"),
    "plays": TablePolicy(GAME, game_col="gameId", season_col="season"),
    "games_teams": TablePolicy(GAME, game_col="id", season_col="season"),
    "stats_game_advanced": TablePolicy(GAME, game_col="gameId", season_col="season"),
    "ppa_games": TablePolicy(GAME, game_col="gameId", season_col="season"),
    "games_players": TablePolicy(GAME, game_col="gameId", season_col="season",
                                 keys=("category", "statType", "athleteId")),
    "stats_game_havoc": TablePolicy(GAME, game_col="gameId", season_col="season"),
    "punting_games": TablePolicy(GAME, game_col="gameId", season_col="season"),
    # ---- raw: player level, per game (P1) ---------------------------------------------
    "player_success_games": TablePolicy(GAME, game_col="gameId", season_col="season",
                                        keys=PLAYER_KEYS),
    "passing_players_games": TablePolicy(GAME, game_col="gameId", season_col="season",
                                         keys=PLAYER_KEYS, note="FBS passers; data from 2025 only"),
    "ppa_players_games": TablePolicy(WEEKLY, season_col="season", season_type_col="season_type",
                                     weekly={"regular": "post", "postseason": "post"}, keys=PLAYER_KEYS,
                                     note="no gameId in the API: rows are visible when their calendar "
                                          "week ends (the postseason is one week, so bowls are visible "
                                          "after the last bowl); join to games through "
                                          "player_success_games or team + opponent"),
    # ---- raw: pregame --------------------------------------------------------------
    "lines": TablePolicy(PREGAME, game_col="id", season_col="season", hide=("homeScore", "awayScore"),
                         note="includes closing lines, which are later than a week-start cutoff; "
                              "market features only"),
    "media": TablePolicy(PREGAME, game_col="id", season_col="season"),
    "weather": TablePolicy(PREGAME, game_col="id", season_col="season",
                           note="observed weather for played games, forecast for upcoming ones"),
    # ---- raw: preseason ------------------------------------------------------------
    "teams_fbs": TablePolicy(PRESEASON, season_col="season"),
    "talent": TablePolicy(PRESEASON, season_col="season"),
    "returning_production": TablePolicy(PRESEASON, season_col="season"),
    "recruiting_teams": TablePolicy(PRESEASON, season_col="season"),
    "ratings_elo_preseason": TablePolicy(PRESEASON, season_col="season"),
    "roster": TablePolicy(PRESEASON, season_col="season", keys=PLAYER_KEYS,
                          note="as of the pull: the season's final roster, used as if known before "
                               "the season (late additions and position changes are not dated)"),
    "recruiting_players": TablePolicy(PRESEASON, season_col="season", keys=PLAYER_KEYS,
                                      note="season = recruiting class year (signing day, before the season)"),
    "recruiting_players_juco": TablePolicy(PRESEASON, season_col="season", keys=PLAYER_KEYS,
                                           note="junior-college recruits; season = class year"),
    "draft_picks": TablePolicy(PRESEASON, season_col="season", keys=PLAYER_KEYS,
                               note="season = draft year; the draft is held in April, before that season"),
    "coach_tenures": TablePolicy(STATIC, hide=("record.*", "seasons", "active", "attributionComplete"),
                                 transform=_coach_tenures,
                                 note="a tenure appears once it has begun; later end dates are blanked"),
    # ---- raw: season-final, as of the pull date (prior seasons only) -------------
    "ratings_sp": TablePolicy(SEASON_FINAL, season_col="season"),
    "ratings_fpi": TablePolicy(SEASON_FINAL, season_col="season"),
    "ratings_srs": TablePolicy(SEASON_FINAL, season_col="season"),
    "ratings_elo": TablePolicy(SEASON_FINAL, season_col="season"),
    "ratings_core": TablePolicy(SEASON_FINAL, season_col="season"),
    "records": TablePolicy(SEASON_FINAL, season_col="season"),
    "stats_season_advanced": TablePolicy(SEASON_FINAL, season_col="season"),
    "coaches": TablePolicy(SEASON_FINAL, season_col="season"),
    "player_usage": TablePolicy(SEASON_FINAL, season_col="season", keys=PLAYER_KEYS),
    "ppa_players_season": TablePolicy(SEASON_FINAL, season_col="season", keys=PLAYER_KEYS),
    # ---- raw: dated ----------------------------------------------------------------
    "transfer_portal": TablePolicy(DATED, date_col="transferDate", season_col="season",
                                   transform=_mask_portal_destination),
    "ratings_sp_asof": TablePolicy(DATED, date_col="asof_utc", season_col="season"),
    "ratings_fpi_asof": TablePolicy(DATED, date_col="asof_utc", season_col="season"),
    "ratings_srs_asof": TablePolicy(DATED, date_col="asof_utc", season_col="season"),
    "ratings_core_asof": TablePolicy(DATED, date_col="asof_utc", season_col="season"),
    # ---- raw: weekly ---------------------------------------------------------------
    "ratings_elo_weekly": TablePolicy(WEEKLY, season_col="season", season_type_col="season_type",
                                      weekly={"regular": "post", "postseason": "post"},
                                      note="week N is the rating after week N's games"),
    "rankings": TablePolicy(WEEKLY, season_col="season", season_type_col="seasonType",
                            weekly={"regular": "pre", "postseason": "post"},
                            note="regular-season polls are known before that week's games; the "
                                 "postseason poll is the final one. The playoff committee ranking is "
                                 "released Tuesday evening ET, about 1.5 days after the week-start cutoff"),
    # ---- raw: static ---------------------------------------------------------------
    "venues": TablePolicy(STATIC),
    "calendar": TablePolicy(STATIC, season_col="season"),
}


def policy_for(name: str, policies: dict[str, TablePolicy] | None = None) -> TablePolicy | None:
    return (policies if policies is not None else POLICIES).get(name)


def describe(policies: dict[str, TablePolicy] | None = None) -> pd.DataFrame:
    """One line per table: kind and note. Used by the README table and --tables."""
    rows = [{"table": n, "kind": p.kind, "note": p.note} for n, p in (policies or POLICIES).items()]
    return pd.DataFrame(rows)

"""team_game: two rows per game (one per team) with everything recorded about that game.

Column families:
    keys         game_id, team_id, opp_id, is_home
    context      season, season_type, week, start_time_utc, neutral, team_fbs, opp_fbs
    result       points, opp_points, margin, win, postgame_win_prob, pregame_elo, postgame_elo
    box_*        team box score from /games/teams ("5-13" style values are split in two)
    adv_off_* / adv_def_*    CFBD advanced stats from /stats/game/advanced (rates per game)
    ppa_off_* / ppa_def_*    CFBD average PPA from /ppa/games
    pbp_off_* / pbp_def_*    additive totals computed from pbp for the units all, rush, pass:
                             plays, epa_sum, successes, expl_sum (EPA on successful plays)

Use the pbp_ totals for anything aggregated over several games; the adv_ and ppa_
columns are per-game rates and cannot be summed. Every non-key column describes the
game itself, so the whole row is a result and is hidden until the game is visible.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ..store import Store
from .pbp import PBP_STATS, UNITS, unit_sums

KEY_COLUMNS = ["game_id", "team_id", "opp_id", "is_home"]
CONTEXT_COLUMNS = ["season", "season_type", "week", "start_time_utc", "neutral", "team_fbs", "opp_fbs"]
RESULT_COLUMNS = ["points", "opp_points", "margin", "win", "postgame_win_prob", "pregame_elo", "postgame_elo"]
PBP_COLUMNS = [f"pbp_{side}_{u}_{s}" for side in ("off", "def") for u in UNITS for s in PBP_STATS]

# box score categories whose value is "made-attempted"
_SPLIT = {
    "thirdDownEff": ("third_down_conv", "third_down_att"),
    "fourthDownEff": ("fourth_down_conv", "fourth_down_att"),
    "completionAttempts": ("completions", "pass_attempts"),
    "totalPenaltiesYards": ("penalties", "penalty_yards"),
}


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _pair(value) -> tuple[float, float]:
    """'5-13' -> (5, 13). Handles a negative second part such as '3--5'."""
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)\s*", str(value))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def _clock(value) -> float:
    parts = str(value).split(":")
    try:
        return float(parts[0]) * 60 + float(parts[1])
    except (IndexError, ValueError):
        return np.nan


def parse_box(games_teams: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team_id) with box_ columns."""
    rows = []
    for gid, teams in zip(games_teams["id"], games_teams["teams"]):
        for team in json.loads(teams) if isinstance(teams, str) else (teams or []):
            row = {"game_id": int(gid), "team_id": int(team["teamId"])}
            for stat in team.get("stats") or []:
                cat, val = stat.get("category"), stat.get("stat")
                if cat in _SPLIT:
                    a, b = _SPLIT[cat]
                    row[f"box_{a}"], row[f"box_{b}"] = _pair(val)
                elif cat == "possessionTime":
                    row["box_possession_seconds"] = _clock(val)
                elif cat:
                    row[f"box_{snake(cat)}"] = _num(val)
            rows.append(row)
    box = pd.DataFrame(rows)
    if box.empty:
        return pd.DataFrame({"game_id": pd.Series(dtype="int64"), "team_id": pd.Series(dtype="int64")})
    return box.drop_duplicates(["game_id", "team_id"], keep="last")


def _flat_name(col: str, prefix: str) -> str | None:
    """'offense.rushingPlays.successRate' -> '<prefix>_off_rushing_plays_success_rate'."""
    parts = col.split(".")
    side = {"offense": "off", "defense": "def"}.get(parts[0])
    if side is None or len(parts) < 2:
        return None
    return "_".join([prefix, side] + [snake(p) for p in parts[1:]])


def _by_team_name(frame: pd.DataFrame, game_index: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Re-key a (gameId, team name) table to (game_id, team_id) and rename its stat columns."""
    names = game_index[["game_id", "home_id", "away_id", "home_team", "away_team"]]
    m = frame.merge(names, left_on="gameId", right_on="game_id", how="inner")
    is_home, is_away = m["team"] == m["home_team"], m["team"] == m["away_team"]
    m = m[is_home | is_away]
    out = pd.DataFrame({
        "game_id": m["game_id"].astype("int64"),
        "team_id": np.where(m["team"] == m["home_team"], m["home_id"], m["away_id"]).astype("int64"),
    })
    for col in frame.columns:
        new = _flat_name(col, prefix)
        if new:
            out[new] = pd.to_numeric(m[col], errors="coerce").astype("float64").to_numpy()
    return out.drop_duplicates(["game_id", "team_id"], keep="last")


def _load(store: Store, name: str) -> pd.DataFrame:
    try:
        return store.load(name)
    except FileNotFoundError:
        return pd.DataFrame()


def build_team_game(store: Store, game_index: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    raw = store.load("games").drop_duplicates("id", keep="last").set_index("id")
    gi = game_index
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        elo = lambda col: pd.to_numeric(raw[col].reindex(gi["game_id"]), errors="coerce").to_numpy(dtype="float64")
        sides.append(pd.DataFrame({
            "game_id": gi["game_id"].to_numpy(),
            "team_id": gi[f"{side}_id"].to_numpy(),
            "opp_id": gi[f"{other}_id"].to_numpy(),
            "is_home": side == "home",
            "season": gi["season"].to_numpy(),
            "season_type": gi["season_type"].to_numpy(),
            "week": gi["week"].to_numpy(),
            "start_time_utc": gi["start_time_utc"],
            "neutral": gi["neutral"].to_numpy(),
            "team_fbs": gi[f"{side}_fbs"].to_numpy(),
            "opp_fbs": gi[f"{other}_fbs"].to_numpy(),
            "points": gi[f"{side}_points"].to_numpy(),
            "opp_points": gi[f"{other}_points"].to_numpy(),
            "postgame_win_prob": elo(f"{side}PostgameWinProbability"),
            "pregame_elo": elo(f"{side}PregameElo"),
            "postgame_elo": elo(f"{side}PostgameElo"),
        }))
    tg = pd.concat(sides, ignore_index=True)
    tg["margin"] = tg["points"] - tg["opp_points"]
    tg["win"] = np.where(tg["margin"].isna(), np.nan, (tg["margin"] > 0).astype("float64"))
    tg = tg[KEY_COLUMNS + CONTEXT_COLUMNS + RESULT_COLUMNS]

    games_teams = _load(store, "games_teams")
    if len(games_teams) and "teams" in games_teams.columns:
        tg = tg.merge(parse_box(games_teams), on=["game_id", "team_id"], how="left")
    for name, prefix in (("stats_game_advanced", "adv"), ("ppa_games", "ppa")):
        frame = _load(store, name)
        if len(frame) and "gameId" in frame.columns:
            tg = tg.merge(_by_team_name(frame, gi, prefix), on=["game_id", "team_id"], how="left")

    sums = unit_sums(pbp)
    stat_cols = [f"{u}_{s}" for u in UNITS for s in PBP_STATS]
    off = sums.rename(columns={"offense_id": "team_id", **{c: f"pbp_off_{c}" for c in stat_cols}}).drop(columns="defense_id")
    dfn = sums.rename(columns={"defense_id": "team_id", **{c: f"pbp_def_{c}" for c in stat_cols}}).drop(columns="offense_id")
    tg = tg.merge(off, on=["game_id", "team_id"], how="left").merge(dfn, on=["game_id", "team_id"], how="left")
    for col in PBP_COLUMNS:
        if col not in tg.columns:
            tg[col] = np.nan
    # A game with plays where one side never ran a scrimmage play still has zero, not unknown.
    has_plays = tg["game_id"].isin(pd.unique(sums["game_id"]))
    tg.loc[has_plays, PBP_COLUMNS] = tg.loc[has_plays, PBP_COLUMNS].fillna(0.0)

    tg = tg.sort_values(["start_time_utc", "game_id", "is_home"], ascending=[True, True, False])
    return tg.reset_index(drop=True)

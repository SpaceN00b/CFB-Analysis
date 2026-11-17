"""pbp: the raw plays with team ids and the shared play classification.

Every thread that reads plays should use this table so rush/pass, success,
passing downs and garbage time mean the same thing everywhere.

Definitions:
    rush / pass      by play type. Fumble plays are assigned from the play text
                     ("pass", "sacked" -> pass; otherwise rush). CFBD's own split differs
                     in two ways: it counts every fumble play as a rush and leaves the
                     "Interception" play type out of passing. With CFBD's conventions the
                     counts here match /stats/game/advanced exactly (rush 98.8%, pass 100%
                     of team-games, 2019 on), so the remaining gap is those two choices.
    scrimmage        rush or pass, and CFBD gave the play a PPA (EPA) value. CFBD's
                     `offense.plays` counts every play with a PPA value, which adds some
                     field goals and period markers (about 0.7 plays per team-game).
    success          touchdown by the offense, or at least 50% of the distance on
                     first down, 70% on second, 100% on third and fourth. Turnovers fail.
    passing_down     second down with 8+ to go, or third/fourth down with 5+ to go.
    garbage_time     score margin above 38 in the 2nd quarter, 28 in the 3rd, 22 in the 4th.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..store import Store

RUSH_TYPES = frozenset({"Rush", "Rushing Touchdown"})
PASS_TYPES = frozenset({
    "Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack",
    "Pass Interception Return", "Interception", "Interception Return Touchdown",
    "Pass Completion", "Pass",
})
FUMBLE_TYPES = frozenset({
    "Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble Return Touchdown", "Fumble",
})
TURNOVER_TYPES = frozenset({
    "Pass Interception Return", "Interception", "Interception Return Touchdown",
    "Fumble Recovery (Opponent)", "Fumble Return Touchdown",
})
OFFENSE_TD_TYPES = frozenset({"Rushing Touchdown", "Passing Touchdown"})
PUNT_TYPES = frozenset({
    "Punt", "Blocked Punt", "Punt Return", "Punt Return Touchdown", "Blocked Punt Touchdown",
})
FIELD_GOAL_TYPES = frozenset({
    "Field Goal Good", "Field Goal Missed", "Blocked Field Goal", "Missed Field Goal Return",
    "Missed Field Goal Return Touchdown", "Blocked Field Goal Touchdown",
})
KICKOFF_TYPES = frozenset({"Kickoff", "Kickoff Return (Offense)", "Kickoff Return Touchdown"})

GARBAGE_MARGIN = {2: 38, 3: 28, 4: 22}

PBP_COLUMNS = [
    "game_id", "play_id", "drive_id", "drive_number", "play_number",
    "season", "season_type", "week",
    "offense_id", "defense_id", "offense_is_home",
    "period", "clock_seconds", "down", "distance", "yards_to_goal", "yards_gained",
    "offense_score", "defense_score", "score_diff",
    "play_type", "play_class", "is_rush", "is_pass", "is_scrimmage",
    "ppa", "success", "passing_down", "turnover", "scoring", "garbage_time",
]


def load_raw_plays(store: Store) -> pd.DataFrame:
    """All raw play partitions. Weeks cached as empty are skipped."""
    files = sorted((store.root / "tables" / "plays").glob("*.parquet"))
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        if len(frame) and "playType" in frame.columns:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def classify(play_type: pd.Series, play_text: pd.Series | None = None) -> pd.DataFrame:
    """play_class, is_rush, is_pass for each play."""
    pt = play_type.astype("object")
    is_rush = pt.isin(RUSH_TYPES).to_numpy().copy()
    is_pass = pt.isin(PASS_TYPES).to_numpy().copy()
    fumble = pt.isin(FUMBLE_TYPES).to_numpy()
    if fumble.any():
        if play_text is not None:
            text = play_text[fumble].astype("object").fillna("").str.lower()
            passy = text.str.contains(r"\bpass|sacked", regex=True).to_numpy()
        else:
            passy = np.zeros(int(fumble.sum()), dtype=bool)
        idx = np.flatnonzero(fumble)
        is_pass[idx[passy]] = True
        is_rush[idx[~passy]] = True
    cls = np.full(len(pt), "other", dtype=object)
    cls[pt.isin(PUNT_TYPES).to_numpy()] = "punt"
    cls[pt.isin(FIELD_GOAL_TYPES).to_numpy()] = "field_goal"
    cls[pt.isin(KICKOFF_TYPES).to_numpy()] = "kickoff"
    cls[(pt == "Penalty").to_numpy()] = "penalty"
    cls[is_rush] = "rush"
    cls[is_pass] = "pass"
    return pd.DataFrame({"play_class": cls, "is_rush": is_rush, "is_pass": is_pass}, index=play_type.index)


def build_pbp(plays: pd.DataFrame, game_index: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw plays. Plays of games missing from game_index are dropped."""
    if plays.empty:
        return pd.DataFrame({c: [] for c in PBP_COLUMNS})
    gi = game_index[["game_id", "home_id", "away_id", "home_team", "away_team", "start_time_utc"]]
    p = plays.merge(gi, left_on="gameId", right_on="game_id", how="inner")
    off_home = (p["offense"] == p["home_team"]).to_numpy()
    off_away = (p["offense"] == p["away_team"]).to_numpy()
    known = off_home | off_away
    p = p[known].reset_index(drop=True)
    off_home = off_home[known]

    cls = classify(p["playType"], p.get("playText"))
    ppa = pd.to_numeric(p["ppa"], errors="coerce").astype("float64")
    scrimmage = (cls["is_rush"] | cls["is_pass"]).to_numpy() & ppa.notna().to_numpy()

    down = p["down"].astype("int64").to_numpy()
    distance = p["distance"].astype("float64").to_numpy()
    gained = p["yardsGained"].astype("float64").to_numpy()
    need = np.select([down == 1, down == 2, np.isin(down, (3, 4))], [0.5, 0.7, 1.0], default=np.nan)
    turnover = p["playType"].isin(TURNOVER_TYPES).to_numpy()
    off_td = p["playType"].isin(OFFENSE_TD_TYPES).to_numpy()
    success = ((gained >= need * distance) & ~np.isnan(need)) | off_td
    success &= ~turnover
    success_col = pd.array(np.where(scrimmage, success, False), dtype="boolean")
    success_col[~scrimmage] = pd.NA

    period = p["period"].astype("int64").to_numpy()
    off_score = p["offenseScore"].astype("int64").to_numpy()
    def_score = p["defenseScore"].astype("int64").to_numpy()
    margin = np.abs(off_score - def_score)
    garbage = np.zeros(len(p), dtype=bool)
    for q, limit in GARBAGE_MARGIN.items():
        garbage |= (period == q) & (margin > limit)

    out = pd.DataFrame({
        "game_id": p["game_id"].astype("int64"),
        "play_id": p["id"].astype("str"),
        "drive_id": p["driveId"].astype("str"),
        "drive_number": p["driveNumber"].astype("int64"),
        "play_number": p["playNumber"].astype("int64"),
        "season": p["season"].astype("int64"),
        "season_type": p["season_type"].astype("str"),
        "week": p["week"].astype("int64"),
        "offense_id": np.where(off_home, p["home_id"], p["away_id"]).astype("int64"),
        "defense_id": np.where(off_home, p["away_id"], p["home_id"]).astype("int64"),
        "offense_is_home": off_home,
        "period": period,
        "clock_seconds": (p["clock.minutes"].fillna(0) * 60 + p["clock.seconds"].fillna(0)).astype("int64"),
        "down": down,
        "distance": p["distance"].astype("int64"),
        "yards_to_goal": p["yardsToGoal"].astype("int64"),
        "yards_gained": p["yardsGained"].astype("int64"),
        "offense_score": off_score,
        "defense_score": def_score,
        "score_diff": off_score - def_score,
        "play_type": p["playType"].astype("str"),
        "play_class": cls["play_class"].astype("str"),
        "is_rush": cls["is_rush"].to_numpy(),
        "is_pass": cls["is_pass"].to_numpy(),
        "is_scrimmage": scrimmage,
        "ppa": ppa,
        "success": success_col,
        "passing_down": ((down == 2) & (distance >= 8)) | (np.isin(down, (3, 4)) & (distance >= 5)),
        "turnover": turnover,
        "scoring": p["scoring"].fillna(False).astype(bool),
        "garbage_time": garbage,
    })
    order = np.lexsort((
        out["play_number"].to_numpy(), out["drive_number"].to_numpy(),
        out["game_id"].to_numpy(), p["start_time_utc"].astype("int64").to_numpy(),
    ))
    return out.iloc[order][PBP_COLUMNS].reset_index(drop=True)


UNITS = ("all", "rush", "pass")
PBP_STATS = ("plays", "epa_sum", "successes", "expl_sum")


def unit_sums(pbp: pd.DataFrame) -> pd.DataFrame:
    """Additive scrimmage totals per (game_id, offense_id, defense_id).

    Columns `<unit>_plays`, `<unit>_epa_sum`, `<unit>_successes`, `<unit>_expl_sum`
    for unit in all, rush, pass. `expl_sum` is the EPA total on successful plays, so
    explosiveness = expl_sum / successes and EPA per play = epa_sum / plays.
    """
    cols = [f"{u}_{s}" for u in UNITS for s in PBP_STATS]
    keys = ["game_id", "offense_id", "defense_id"]
    s = pbp[pbp["is_scrimmage"]]
    if s.empty:
        return pd.DataFrame({c: [] for c in keys + cols})
    ok = s["success"].fillna(False).astype(bool).to_numpy()
    ppa = s["ppa"].to_numpy()
    work = s[keys].copy()
    for unit, mask in (("all", np.ones(len(s), dtype=bool)), ("rush", s["is_rush"].to_numpy()), ("pass", s["is_pass"].to_numpy())):
        work[f"{unit}_plays"] = mask.astype("float64")
        work[f"{unit}_epa_sum"] = np.where(mask, ppa, 0.0)
        work[f"{unit}_successes"] = (mask & ok).astype("float64")
        work[f"{unit}_expl_sum"] = np.where(mask & ok, ppa, 0.0)
    return work.groupby(keys, as_index=False, sort=False)[cols].sum()

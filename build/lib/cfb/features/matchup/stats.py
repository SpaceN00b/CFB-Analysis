"""Per-game and per-team style statistics shared by the matchup groups.

Everything here reads through an AsOfView and returns plain frames; nothing is kept
between calls.

Two layers:
    game_stats(view, seasons)   one row per visible team-game with additive totals
                                (numerators and denominators), offense as off_*, the
                                opponent's offense in the same game as def_*
    team_profile(...)           per-team rates over a season, optionally blended with
                                the previous season (see PRIOR_GAMES)

Definitions (all on scrimmage plays from the shared pbp flags, garbage time removed,
unless stated):
    neutral          first or second down, score within 14
    buckets          1st down; 2nd and 1-3, 4-7, 8+; 3rd/4th and 1-3, 4+; score within 14
    explosive play   rush of 12+ yards or pass of 16+ yards
    explosive share  positive EPA on explosive plays over all positive EPA
    pace             scrimmage plays per game including garbage time
    seconds/play     drive elapsed time over drive plays (drives table)
    havoc            stats_game_havoc events over plays, play-weighted
    line yards, stuff rate   adv_* per-game rates weighted by the game's rush plays
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Games of prior-season data that the previous season's rates count for when blended
# into the current season's. Chosen on seasons through 2025 (docs/matchup.md).
PRIOR_GAMES = 4.0

EXPLOSIVE_RUSH_YARDS = 12
EXPLOSIVE_PASS_YARDS = 16
NEUTRAL_SCORE = 14

BUCKETS = ("1st", "2nd_short", "2nd_med", "2nd_long", "3rd_short", "3rd_long")

_PBP_COLUMNS = ["game_id", "season", "offense_id", "down", "distance", "score_diff", "yards_gained",
                "is_rush", "is_pass", "is_scrimmage", "ppa", "success", "passing_down", "garbage_time"]

# Additive per-game totals computed from pbp for the offense.
_PBP_SUMS = (
    ["all_plays", "scrim", "rush", "epa", "rush_n", "rush_epa", "pass_n", "pass_epa", "pos_epa", "expl_pos_epa",
     "expl_n", "neu_n", "neu_rush", "sd_n", "sd_succ", "pd_n", "pd_succ", "pbp_game"]
    + [f"b_{b}_{x}" for b in BUCKETS for x in ("n", "rush")]
)
# Additive totals from other tables, already per side.
_OTHER_SUMS = ["drive_secs", "drive_plays", "havoc_ev", "havoc_plays", "ly_num", "ly_den", "st_num", "st_den"]
SUMS = _PBP_SUMS + _OTHER_SUMS


def _pbp_sums(view, seasons: list[int]) -> pd.DataFrame:
    p = view.table("pbp", columns=_PBP_COLUMNS)
    p = p[p["season"].isin(seasons).to_numpy()]
    scrim = p["is_scrimmage"].to_numpy(dtype=bool)
    p = p[scrim]
    garbage = p["garbage_time"].fillna(False).to_numpy(dtype=bool)
    rush = p["is_rush"].to_numpy(dtype=bool)
    pas = p["is_pass"].to_numpy(dtype=bool)
    down = p["down"].to_numpy()
    dist = p["distance"].to_numpy()
    yards = p["yards_gained"].to_numpy()
    ppa = p["ppa"].to_numpy(dtype=float)
    ppa0 = np.nan_to_num(ppa)
    succ = p["success"].fillna(False).to_numpy(dtype=bool)
    pdown = p["passing_down"].fillna(False).to_numpy(dtype=bool)
    close = np.abs(p["score_diff"].to_numpy()) <= NEUTRAL_SCORE
    live = ~garbage
    expl = (rush & (yards >= EXPLOSIVE_RUSH_YARDS)) | (pas & (yards >= EXPLOSIVE_PASS_YARDS))
    pos = np.clip(ppa0, 0, None)
    lc = live & close
    cols = {
        "game_id": p["game_id"].to_numpy(), "team_id": p["offense_id"].to_numpy(),
        "all_plays": np.ones(len(p)),
        "scrim": live * 1.0, "rush": (live & rush) * 1.0, "epa": live * ppa0,
        "rush_n": (live & rush) * 1.0, "rush_epa": (live & rush) * ppa0,
        "pass_n": (live & pas) * 1.0, "pass_epa": (live & pas) * ppa0,
        "pos_epa": live * pos, "expl_pos_epa": (live & expl) * pos, "expl_n": (live & expl) * 1.0,
        "neu_n": (lc & (down <= 2)) * 1.0, "neu_rush": (lc & (down <= 2) & rush) * 1.0,
        "sd_n": (live & ~pdown) * 1.0, "sd_succ": (live & ~pdown & succ) * 1.0,
        "pd_n": (live & pdown) * 1.0, "pd_succ": (live & pdown & succ) * 1.0,
    }
    masks = {
        "1st": down == 1,
        "2nd_short": (down == 2) & (dist <= 3), "2nd_med": (down == 2) & (dist >= 4) & (dist <= 7),
        "2nd_long": (down == 2) & (dist >= 8),
        "3rd_short": (down >= 3) & (dist <= 3), "3rd_long": (down >= 3) & (dist >= 4),
    }
    for b, m in masks.items():
        cols[f"b_{b}_n"] = (lc & m) * 1.0
        cols[f"b_{b}_rush"] = (lc & m & rush) * 1.0
    frame = pd.DataFrame(cols)
    out = frame.groupby(["game_id", "team_id"], sort=False).sum().reset_index()
    out["pbp_game"] = 1.0
    return out


def _team_ids_by_side(view, seasons: list[int]) -> pd.DataFrame:
    sched = view.schedule()
    sched = sched[sched["season"].isin(seasons)]
    return sched[["game_id", "home_id", "away_id", "home_team", "away_team"]]


def _drive_sums(view, seasons: list[int], sides: pd.DataFrame) -> pd.DataFrame | None:
    if not view.has("drives"):
        return None
    d = view.table("drives", columns=["gameId", "isHomeOffense", "plays", "elapsed.minutes", "elapsed.seconds",
                                      "season"])
    d = d[d["season"].isin(seasons).to_numpy()]
    secs = 60.0 * d["elapsed.minutes"].to_numpy(dtype=float) + d["elapsed.seconds"].to_numpy(dtype=float)
    plays = d["plays"].to_numpy(dtype=float)
    ok = (plays > 0) & (secs >= 0) & (secs <= 900)
    d = pd.DataFrame({"game_id": d["gameId"].to_numpy()[ok], "home": d["isHomeOffense"].to_numpy(dtype=bool)[ok],
                      "drive_secs": secs[ok], "drive_plays": plays[ok]})
    d = d.merge(sides[["game_id", "home_id", "away_id"]], on="game_id", how="inner")
    d["team_id"] = np.where(d["home"], d["home_id"], d["away_id"])
    return d.groupby(["game_id", "team_id"], sort=False)[["drive_secs", "drive_plays"]].sum().reset_index()


def _havoc_sums(view, seasons: list[int], sides: pd.DataFrame) -> pd.DataFrame | None:
    """Havoc the team's offense allowed in each game (its defense's is the opponent's row)."""
    if not view.has("stats_game_havoc"):
        return None
    h = view.table("stats_game_havoc", columns=["gameId", "season", "team", "offense.totalHavocEvents",
                                                "offense.totalPlays"])
    h = h[h["season"].isin(seasons).to_numpy()]
    h = h.rename(columns={"gameId": "game_id"}).merge(sides, on="game_id", how="inner")
    team_id = np.where(h["team"] == h["home_team"], h["home_id"],
                       np.where(h["team"] == h["away_team"], h["away_id"], -1))
    ev = pd.to_numeric(h["offense.totalHavocEvents"], errors="coerce").to_numpy(dtype=float)
    pl = pd.to_numeric(h["offense.totalPlays"], errors="coerce").to_numpy(dtype=float)
    ok = (team_id >= 0) & np.isfinite(ev) & np.isfinite(pl) & (pl > 0)
    out = pd.DataFrame({"game_id": h["game_id"].to_numpy()[ok], "team_id": team_id[ok],
                        "havoc_ev": ev[ok], "havoc_plays": pl[ok]})
    return out.groupby(["game_id", "team_id"], sort=False).sum().reset_index()


def _weighted(rate: pd.Series | None, weight: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if rate is None:
        n = len(weight)
        return np.zeros(n), np.zeros(n)
    r = pd.to_numeric(rate, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weight, errors="coerce").fillna(0).to_numpy(dtype=float)
    ok = np.isfinite(r) & (w > 0)
    return np.where(ok, r * w, 0.0), np.where(ok, w, 0.0)


def game_stats(view, seasons) -> pd.DataFrame:
    """One row per visible team-game of `seasons` with off_* and def_* additive totals."""
    seasons = [int(s) for s in (np.atleast_1d(seasons))]
    tg = view.table("team_game", seasons=seasons)
    base = tg[["game_id", "team_id", "opp_id", "season", "season_type", "start_time_utc", "team_fbs",
               "opp_fbs"]].reset_index(drop=True)
    if base.empty:
        return pd.DataFrame(columns=list(base.columns) + [f"{s}_{c}" for s in ("off", "def") for c in SUMS])
    side = base[["game_id", "team_id"]].copy()
    pbp = _pbp_sums(view, seasons)
    side = side.merge(pbp, on=["game_id", "team_id"], how="left")
    sides = _team_ids_by_side(view, seasons)
    for extra in (_drive_sums(view, seasons, sides), _havoc_sums(view, seasons, sides)):
        if extra is not None:
            side = side.merge(extra, on=["game_id", "team_id"], how="left")
    get = lambda c: tg[c].reset_index(drop=True) if c in tg.columns else None
    side["ly_num"], side["ly_den"] = _weighted(get("adv_off_line_yards"), tg["pbp_off_rush_plays"].reset_index(drop=True))
    side["st_num"], side["st_den"] = _weighted(get("adv_off_stuff_rate"), tg["pbp_off_rush_plays"].reset_index(drop=True))
    for c in SUMS:
        if c not in side.columns:
            side[c] = 0.0
    side[SUMS] = side[SUMS].astype(float).fillna(0.0)
    off = side.rename(columns={c: f"off_{c}" for c in SUMS})
    dfn = side.rename(columns={"team_id": "opp_id", **{c: f"def_{c}" for c in SUMS}})
    out = base.merge(off, on=["game_id", "team_id"], how="left").merge(dfn, on=["game_id", "opp_id"], how="left")
    cols = [f"{s}_{c}" for s in ("off", "def") for c in SUMS]
    out[cols] = out[cols].fillna(0.0)
    return out


# ---- per-team rates -------------------------------------------------------------------

# name -> (numerator, denominator) over additive game totals. Output columns get "mat_".
RATES: dict[str, tuple[str, str]] = {
    "off_run_rate": ("off_rush", "off_scrim"),
    "off_run_rate_neutral": ("off_neu_rush", "off_neu_n"),
    **{f"off_run_rate_{b}": (f"off_b_{b}_rush", f"off_b_{b}_n") for b in BUCKETS},
    "off_plays": ("off_all_plays", "off_pbp_game"),
    "def_plays": ("def_all_plays", "def_pbp_game"),
    "off_sec_per_play": ("off_drive_secs", "off_drive_plays"),
    "off_expl_share": ("off_expl_pos_epa", "off_pos_epa"),
    "off_expl_rate": ("off_expl_n", "off_scrim"),
    "def_expl_share": ("def_expl_pos_epa", "def_pos_epa"),
    "def_expl_rate": ("def_expl_n", "def_scrim"),
    "off_sd_success": ("off_sd_succ", "off_sd_n"),
    "off_pd_success": ("off_pd_succ", "off_pd_n"),
    "off_pd_rate": ("off_pd_n", "off_scrim"),
    "def_sd_success": ("def_sd_succ", "def_sd_n"),
    "def_pd_success": ("def_pd_succ", "def_pd_n"),
    "off_havoc_allowed": ("off_havoc_ev", "off_havoc_plays"),
    "def_havoc": ("def_havoc_ev", "def_havoc_plays"),
    "off_line_yards": ("off_ly_num", "off_ly_den"),
    "off_stuff_rate": ("off_st_num", "off_st_den"),
    "def_line_yards": ("def_ly_num", "def_ly_den"),
    "def_stuff_rate": ("def_st_num", "def_st_den"),
    # internal: raw efficiency, used by the style history (not output; ratings are W3's)
    "off_epa": ("off_epa", "off_scrim"),
    "def_epa": ("def_epa", "def_scrim"),
    "off_rush_epa": ("off_rush_epa", "off_rush_n"),
    "off_pass_epa": ("off_pass_epa", "off_pass_n"),
    "def_rush_epa": ("def_rush_epa", "def_rush_n"),
    "def_pass_epa": ("def_pass_epa", "def_pass_n"),
}
_NEEDED = sorted({c for pair in RATES.values() for c in pair})


def _totals(stats: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = stats.groupby(by, sort=True)
    out = g[_NEEDED].sum()
    out["games"] = g.size().astype(float)
    return out


def _rates(totals: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=totals.index)
    for name, (num, den) in RATES.items():
        d = totals[den].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = np.where(d > 0, totals[num].to_numpy(dtype=float) / d, np.nan)
    out["games"] = totals["games"]
    return out


def season_profiles(stats: pd.DataFrame) -> pd.DataFrame:
    """Rates per (season, team_id) over every game in `stats`, no blending."""
    if stats.empty:
        return pd.DataFrame(columns=list(RATES) + ["games"])
    return _rates(_totals(stats, ["season", "team_id"]))


def blended_profiles(stats: pd.DataFrame, season: int, prior_games: float = PRIOR_GAMES) -> pd.DataFrame:
    """Rates per team for `season`, with the previous season counted as `prior_games` games.

    Numerator and denominator of every rate get the prior season's per-game totals times
    `prior_games`, so a team with no games yet gets its prior-season rate and the prior
    fades as games are played. Teams without either season have no row. `games` is the
    number of current-season games.
    """
    cur = stats[stats["season"] == season]
    prev = stats[stats["season"] == season - 1]
    tc = _totals(cur, ["team_id"]) if not cur.empty else pd.DataFrame(columns=_NEEDED + ["games"], dtype=float)
    tp = _totals(prev, ["team_id"]) if not prev.empty else pd.DataFrame(columns=_NEEDED + ["games"], dtype=float)
    teams = tc.index.union(tp.index)
    tc = tc.reindex(teams).fillna(0.0)
    tp = tp.reindex(teams).fillna(0.0)
    scale = np.where(tp["games"].to_numpy() > 0, prior_games / tp["games"].to_numpy().clip(min=1), 0.0)
    blend = tc[_NEEDED] + tp[_NEEDED].mul(scale, axis=0)
    blend["games"] = tc["games"]
    out = _rates(blend)
    out.index.name = "team_id"
    return out


def fbs_teams(view, season: int) -> np.ndarray:
    teams = view.table("teams", seasons=season)
    return teams.loc[teams["fbs"].fillna(False).astype(bool), "team_id"].unique()

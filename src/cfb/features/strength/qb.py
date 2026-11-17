"""Quarterback quality (P1, group `str_qb`).

Player-game rows come from `player_success_games` (passing and rushing plays credited to
the player, with gameId) joined to `ppa_players_games` (average PPA as passer and rusher).
A player's value per game is his PPA over his passing plus rushing plays, adjusted for the
opponent: the defense's pass (rush) EPA allowed at that game's cutoff, relative to the
mean FBS defense, is subtracted per pass (rush) play.

A player's quality at a cutoff is a shrunk, recency-weighted mean of that value over every
game visible at the cutoff, at any school (transfers keep their history):

    q = (sum_w adj_ppa + K * prior) / (sum_w plays + K),    w = DECAY ** (seasons ago)

The prior is a fitted function of the recruiting rating (high-school or junior-college
class) and the roster class year (`scripts/qb_fit.py`, seasons through 2025).

Expected starter of a team at a cutoff:
  * after a game this season: the player with the most passing plays in its last visible game;
  * before its first game: of the season roster's quarterbacks, the one with the most
    passing plays last season (any school), then career, then the best recruiting rating.

Mix: the play-weighted mean of today's quality of the players who took the team's QB plays
in the games behind its rating (this season's games; before the first game, last season's
games for that team). `str_qb_diff` = starter minus mix: a starter change the team's
rating has not seen yet.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup
from .common import team_keys

# Fitted by scripts/qb_fit.py on seasons 2014-2025 (34,842 player-games; docs/qb.md).
K = 150.0  # shrinkage, in plays
DECAY = 0.4  # weight per season of age
PRIOR = {"const": 0.2163, "rating": 0.8401, "rating_center": 0.85, "no_rating": -0.0400, "juco": 0.0309,
         "class_year": 0.0055}
MIN_PASS_PLAYS = 1

SOURCES = ("player_success_games", "ppa_players_games", "unit_ratings", "roster", "recruiting_players",
           "recruiting_players_juco")

SUCC_COLS = ["season", "seasonType", "week", "gameId", "id", "team", "opponent", "position",
             "passing.plays", "passing.successes", "rushing.plays", "rushing.successes"]
PPA_COLS = ["season", "seasonType", "week", "id", "team", "opponent", "averagePPA.pass", "averagePPA.rush"]
KEYS = ["season", "seasonType", "week", "id", "team", "opponent"]


def _num(s) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)


def player_games(view) -> pd.DataFrame:
    """One row per player and game visible at the cutoff, players with a passing play.

    Columns: game_id, season, start_time_utc, cutoff_utc, team_id, opp_id, athlete,
    pass_plays, rush_plays, plays, succ, ppa, adj (opponent-adjusted PPA total).
    """
    if not (view.has("player_success_games") and view.has("ppa_players_games")):
        return pd.DataFrame()
    succ = view.table("player_success_games").reindex(columns=SUCC_COLS)
    succ = succ[_num(succ["passing.plays"]) >= MIN_PASS_PLAYS]
    if succ.empty:
        return pd.DataFrame()
    ppa = view.table("ppa_players_games").reindex(columns=PPA_COLS).drop_duplicates(KEYS)
    for f in (succ, ppa):
        f["id"] = f["id"].astype(str)
    df = succ.merge(ppa, on=KEYS, how="left")
    gi = view.table("game_index", columns=["game_id", "season", "start_time_utc", "cutoff_utc", "home_id",
                                           "away_id", "home_team", "away_team"])
    df = df.merge(gi.drop(columns="season"), left_on="gameId", right_on="game_id", how="inner")
    home = (df["team"] == df["home_team"]).to_numpy()
    out = pd.DataFrame({
        "game_id": df["game_id"].to_numpy(), "season": df["season"].to_numpy(),
        "start_time_utc": df["start_time_utc"].to_numpy(), "cutoff_utc": df["cutoff_utc"].to_numpy(),
        "team_id": np.where(home, df["home_id"], df["away_id"]),
        "opp_id": np.where(home, df["away_id"], df["home_id"]),
        "athlete": df["id"].to_numpy(),
    })
    pp, rp = np.nan_to_num(_num(df["passing.plays"])), np.nan_to_num(_num(df["rushing.plays"]))
    pa, ra = _num(df["averagePPA.pass"]), _num(df["averagePPA.rush"])
    # A missing average with plays > 0 (no PPA row) leaves those plays out of the mean.
    pp_ok, rp_ok = np.where(np.isfinite(pa), pp, 0.0), np.where(np.isfinite(ra), rp, 0.0)
    out["pass_plays"], out["rush_plays"] = pp, rp
    out["plays"] = pp_ok + rp_ok
    out["succ"] = np.nan_to_num(_num(df["passing.successes"])) + np.nan_to_num(_num(df["rushing.successes"]))
    out["ppa"] = np.nan_to_num(pa) * pp_ok + np.nan_to_num(ra) * rp_ok
    out["_pp"], out["_rp"] = pp_ok, rp_ok
    out = _adjust(view, out)
    return out.drop(columns=["_pp", "_rp"])


def _adjust(view, pg: pd.DataFrame) -> pd.DataFrame:
    """Subtract the opponent defense's pass/rush EPA allowed (relative to the mean FBS defense at
    that game's cutoff) per play. Unknown opponent ratings count as average."""
    pg = pg.copy()
    pg["adj"] = pg["ppa"]
    if not view.has("unit_ratings") or pg.empty:
        return pg
    ur = view.table("unit_ratings", columns=["team_id", "cutoff_utc", "unit", "epa_adj", "n_plays"])
    ur = ur[ur["unit"].isin(["def_pass", "def_rush"])]
    if ur.empty:
        return pg
    fbs = set(view.table("teams", columns=["team_id", "fbs"]).query("fbs")["team_id"]) if view.has("teams") else set()
    centre = ur[ur["team_id"].isin(fbs)] if fbs else ur
    mean = centre.groupby(["cutoff_utc", "unit"])["epa_adj"].mean().rename("mu").reset_index()
    ur = ur.merge(mean, on=["cutoff_utc", "unit"], how="left")
    ur = ur.assign(rel=ur["epa_adj"] - ur["mu"])
    wide = ur.pivot_table(index=["cutoff_utc", "team_id"], columns="unit", values="rel").reset_index()
    wide = wide.rename(columns={"team_id": "opp_id"})
    m = pg[["cutoff_utc", "opp_id"]].merge(wide, on=["cutoff_utc", "opp_id"], how="left")
    dp = np.nan_to_num(m.get("def_pass", pd.Series(np.nan, index=m.index)).to_numpy(dtype=np.float64))
    dr = np.nan_to_num(m.get("def_rush", pd.Series(np.nan, index=m.index)).to_numpy(dtype=np.float64))
    pg["adj"] = pg["ppa"].to_numpy() - pg["_pp"].to_numpy() * dp - pg["_rp"].to_numpy() * dr
    return pg


def recruit_info(view) -> pd.DataFrame:
    """athlete -> rating (high school or JUCO, the latest class), juco flag. Recruits are
    linked by athleteId and through the roster's recruitIds."""
    frames = []
    for name, juco in (("recruiting_players", 0.0), ("recruiting_players_juco", 1.0)):
        if view.has(name):
            r = view.table(name).reindex(columns=["id", "athleteId", "rating", "season"])
            frames.append(r.assign(juco=juco))
    if not frames:
        return pd.DataFrame(columns=["athlete", "rating", "juco"])
    rec = pd.concat(frames, ignore_index=True)
    rec["id"] = rec["id"].astype(str)
    links = [rec.loc[rec["athleteId"].notna()].assign(athlete=lambda f: f["athleteId"].astype(str))]
    if view.has("roster"):
        ros = view.table("roster").reindex(columns=["id", "recruitIds"])
        ros = ros[ros["recruitIds"].notna()]
        ids = ros["recruitIds"].map(lambda v: json.loads(v) if isinstance(v, str) else list(v or []))
        ex = pd.DataFrame({"athlete": ros["id"].astype(str).to_numpy(), "rid": ids.to_numpy()}).explode("rid")
        ex = ex.dropna().assign(rid=lambda f: f["rid"].astype(str)).drop_duplicates()
        links.append(ex.merge(rec, left_on="rid", right_on="id", how="inner"))
    lk = pd.concat(links, ignore_index=True)
    lk = lk[lk["rating"].notna()].sort_values(["athlete", "season"])
    return lk.drop_duplicates("athlete", keep="last")[["athlete", "rating", "juco"]]


def prior_mean(rating: np.ndarray, juco: np.ndarray, class_year: np.ndarray) -> np.ndarray:
    p = PRIOR
    has = np.isfinite(rating)
    val = p["const"] + np.where(has, p["rating"] * (np.nan_to_num(rating) - p["rating_center"]), p["no_rating"])
    val = val + p["juco"] * np.nan_to_num(juco)
    cy = np.clip(np.nan_to_num(class_year, nan=2.0), 1, 5)
    return val + p["class_year"] * (cy - 2.0)


def roster_frame(view, season: int) -> pd.DataFrame:
    if not view.has("roster"):
        return pd.DataFrame(columns=["athlete", "team", "position", "class_year"])
    r = view.table("roster", seasons=[season]).reindex(columns=["id", "team", "position", "year"])
    return pd.DataFrame({"athlete": r["id"].astype(str), "team": r["team"], "position": r["position"],
                         "class_year": _num(r["year"])})


def quality(pg: pd.DataFrame, season: int, priors: pd.DataFrame, k: float = K, decay: float = DECAY) -> pd.DataFrame:
    """athlete -> q, plays (unweighted career), plays_w, prior. `priors` has athlete, prior."""
    if pg.empty:
        return pd.DataFrame(columns=["athlete", "q", "plays", "plays_w", "prior", "last_pass"])
    w = decay ** (season - pg["season"].to_numpy(dtype=np.float64))
    g = pd.DataFrame({"athlete": pg["athlete"].to_numpy(), "aw": pg["adj"].to_numpy() * w,
                      "pw": pg["plays"].to_numpy() * w, "plays": pg["plays"].to_numpy()})
    agg = g.groupby("athlete", as_index=False).agg(adj_w=("aw", "sum"), plays_w=("pw", "sum"), plays=("plays", "sum"))
    last = pg[pg["season"] == season - 1].groupby("athlete")["pass_plays"].sum()
    agg["last_pass"] = agg["athlete"].map(last).fillna(0.0)
    agg = agg.merge(priors[["athlete", "prior"]], on="athlete", how="left")
    agg["prior"] = agg["prior"].fillna(prior_mean(np.array([np.nan]), np.array([0.0]), np.array([np.nan]))[0])
    agg["q"] = (agg["adj_w"] + k * agg["prior"]) / (agg["plays_w"] + k)
    return agg


def player_priors(view, season: int, athletes) -> pd.DataFrame:
    rec = recruit_info(view)
    ros = roster_frame(view, season).drop_duplicates("athlete")
    df = pd.DataFrame({"athlete": pd.Index(pd.unique(pd.Series(list(athletes), dtype=object)))})
    df = df.merge(rec, on="athlete", how="left").merge(ros[["athlete", "class_year"]], on="athlete", how="left")
    df["prior"] = prior_mean(df["rating"].to_numpy(dtype=np.float64), df["juco"].to_numpy(dtype=np.float64),
                             df["class_year"].to_numpy(dtype=np.float64))
    return df


def team_names(view) -> dict:
    gi = view.table("game_index", columns=["home_id", "away_id", "home_team", "away_team"])
    names = pd.concat([gi[["home_id", "home_team"]].set_axis(["id", "name"], axis=1),
                       gi[["away_id", "away_team"]].set_axis(["id", "name"], axis=1)])
    return dict(zip(names["name"], names["id"]))


def qb_state(view, starters: dict | None = None, k: float = K, decay: float = DECAY):
    """Per team: expected starter and the values behind the str_qb columns.

    `starters` (team_id -> athlete) overrides the expected starter (oracle backtest,
    manual "QB out" override). Returns (team table indexed by team_id, quality table).
    """
    season = view.season
    pg = player_games(view)
    ros = roster_frame(view, season)
    ros_qb = ros[ros["position"] == "QB"].copy()
    names = team_names(view)
    ros_qb["team_id"] = ros_qb["team"].map(names)
    athletes = set(pg["athlete"]) if len(pg) else set()
    athletes |= set(ros_qb["athlete"]) | set((starters or {}).values())
    pri = player_priors(view, season, athletes)
    q = quality(pg, season, pri, k, decay) if len(pg) else quality(pd.DataFrame(), season, pri)
    qi = q.set_index("athlete")
    pri_i = pri.set_index("athlete")

    def q_of(a):
        if a in qi.index:
            return float(qi.at[a, "q"]), float(qi.at[a, "plays"])
        if a in pri_i.index:
            return float(pri_i.at[a, "prior"]), 0.0
        return np.nan, np.nan

    rows = {}
    cur = pg[pg["season"] == season] if len(pg) else pg
    prev = pg[pg["season"] == season - 1] if len(pg) else pg
    by_team_cur = dict(tuple(cur.groupby("team_id"))) if len(cur) else {}
    by_team_prev = dict(tuple(prev.groupby("team_id"))) if len(prev) else {}
    teams = set(by_team_cur) | set(by_team_prev) | set(ros_qb["team_id"].dropna().astype(int)) | set((starters or {}))
    for t in teams:
        tc = by_team_cur.get(t)
        window = tc if tc is not None and len(tc) else by_team_prev.get(t)
        starter, rule = None, np.nan
        if tc is not None and len(tc):
            last_game = tc.sort_values(["start_time_utc", "game_id"])["game_id"].iat[-1]
            lg = tc[tc["game_id"] == last_game].sort_values(["pass_plays", "athlete"], ascending=[False, True])
            starter, rule = lg["athlete"].iat[0], 0.0
        else:
            cand = ros_qb[ros_qb["team_id"] == t]["athlete"].drop_duplicates()
            if len(cand):
                c = pd.DataFrame({"athlete": cand.to_numpy()})
                c = c.merge(q[["athlete", "last_pass", "plays"]], on="athlete", how="left") \
                     .merge(pri[["athlete", "rating"]], on="athlete", how="left").fillna(
                         {"last_pass": 0.0, "plays": 0.0, "rating": 0.0})
                c = c.sort_values(["last_pass", "plays", "rating", "athlete"], ascending=[False, False, False, True])
                starter, rule = c["athlete"].iat[0], 1.0
        if starters and t in starters:
            starter, rule = starters[t], 2.0
        sq, splays = q_of(starter) if starter is not None else (np.nan, np.nan)
        mix = share = np.nan
        if window is not None and len(window):
            per = window.groupby("athlete")["plays"].sum()
            per = per[per > 0]
            if per.sum() > 0:
                vals = np.array([q_of(a)[0] for a in per.index])
                ok = np.isfinite(vals)
                mix = float(np.sum(vals[ok] * per.to_numpy()[ok]) / per.to_numpy()[ok].sum()) if ok.any() else np.nan
                share = float(per.get(starter, 0.0) / per.sum()) if starter is not None else np.nan
        prior = float(pri_i.at[starter, "prior"]) if starter is not None and starter in pri_i.index else np.nan
        rows[t] = {"athlete": starter, "rule": rule, "q": sq, "plays": splays, "mix": mix, "share": share,
                   "prior": prior}
    cols = ["athlete", "rule", "q", "plays", "mix", "share", "prior"]
    return pd.DataFrame(list(rows.values()), index=list(rows.keys()), columns=cols), q


class QBQuality(FeatureGroup):
    name = "str_qb"
    prefix = "str_"
    level = "team"
    columns = [
        FeatureColumn("str_qb_q", "Expected starter's quality: shrunk, opponent-adjusted PPA per QB play "
                      "(pass + rush) over every visible game at any school", SOURCES, 2014),
        FeatureColumn("str_qb_mix_q", "Play-weighted quality of the QBs behind the team's rating (this "
                      "season's games; before the first game, last season's)", SOURCES, 2014),
        FeatureColumn("str_qb_diff", "str_qb_q minus str_qb_mix_q: starter change not yet in the rating",
                      SOURCES, 2014),
        FeatureColumn("str_qb_plays", "Expected starter's career QB plays visible at the cutoff, log(1 + n)",
                      SOURCES, 2014),
        FeatureColumn("str_qb_share", "Expected starter's share of the team's QB plays behind the rating",
                      SOURCES, 2014),
        FeatureColumn("str_qb_prior", "Expected starter's prior from recruiting rating and class year",
                      SOURCES, 2014),
        FeatureColumn("str_qb_from_roster", "1 if the starter was picked from the roster (no game this "
                      "season yet), 0 if from the last game", SOURCES, 2014),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        st, _ = qb_state(view)
        st = st.reindex(keys["team_id"].to_numpy())
        out = keys[["game_id", "team_id"]].reset_index(drop=True).copy()
        q, mix = st["q"].to_numpy(dtype=np.float64), st["mix"].to_numpy(dtype=np.float64)
        out["str_qb_q"] = q
        out["str_qb_mix_q"] = mix
        out["str_qb_diff"] = q - mix
        out["str_qb_plays"] = np.log1p(st["plays"].to_numpy(dtype=np.float64))
        out["str_qb_share"] = st["share"].to_numpy(dtype=np.float64)
        out["str_qb_prior"] = st["prior"].to_numpy(dtype=np.float64)
        out["str_qb_from_roster"] = st["rule"].to_numpy(dtype=np.float64)
        return out

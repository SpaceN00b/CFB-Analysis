"""Rough team playbooks from parsed play text (P4 + Notre Dame test bench, or one conference).

Experimental: no feature group or model reads this. It works on the detailed stat-crew text
that `cfb.core.playtext` parses (every game from 2025 week 9 on), so the display season is 2026.

What the text has: the formation words "Shotgun" and "No Huddle" (no word means under center),
run direction (left, middle, right), pass depth and direction, air yards, the ball carrier or
intended receiver (roster position from `cfb.lab.player_maps.player_info`), tacklers, sacks and
hurries. What it does not have: personnel groupings, formation names (trips, empty, pistol, I),
motion, play action, RPOs, designed QB runs versus scrambles, or run scheme. Sportradar's
screen, play-action and RPO flags are false on every 2026 play. So a "play" here is
formation x tempo x play type (run direction by ball-carrier position, or pass depth and
direction by target position), not a named concept.

Each team's offense gets:

- formation: shotgun / under center / no huddle shares and the run rate in each formation.
  No huddle depends on the stat crew (venue explains 44% of the game-to-game variance alone),
  so a crew-adjusted share comes from a two-way fit (offense + venue) over every FBS game.
- pass rate over expected (PROE): actual pass share minus an FBS logistic model on down,
  distance, field position, score, quarter and clock.
- mix: play-type shares, success and EPA per play, against the FBS shares.
- situations: down-and-distance and red-zone rows with run rate, shotgun rate, the most
  common play type and success.
- top plays: the most-called formation x play type x direction x position combinations, with
  the player who gets the ball most on each.
- players: ball carriers and targets with the play-style label from the player maps.

Each team's defense gets success allowed by play type (shrunk toward the FBS rate), sack and
hurry rates, and its most active defenders with the player-map role model's role.

Kneels, spikes, no-plays and garbage time are left out of everything.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import player_maps as pm

POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "X": "X", "Z": "Z", "Slot": "Slot"}
WR_GROUPS = ("WR", "X", "Z", "Slot")
# Team depth chart alignment (cfb.depth_charts via player_maps.player_info) replaces the CFBD
# roster position where the player is charted: receivers become X, Z or Slot.
CHART_LABEL = {"X": "X", "Z": "Z", "SLOT": "Slot", "TE": "TE", "RB": "RB", "QB": "QB"}
SITUATIONS = (
    ("1st & 10", lambda p: (p["down"] == 1) & (p["distance"] == 10) & (p["yards_to_goal"] > 10)),
    ("1st, other", lambda p: (p["down"] == 1) & ~((p["distance"] == 10) & (p["yards_to_goal"] > 10))),
    ("2nd & 1-3", lambda p: (p["down"] == 2) & (p["distance"] <= 3)),
    ("2nd & 4-7", lambda p: (p["down"] == 2) & p["distance"].between(4, 7)),
    ("2nd & 8+", lambda p: (p["down"] == 2) & (p["distance"] >= 8)),
    ("3rd/4th & 1-2", lambda p: (p["down"] >= 3) & (p["distance"] <= 2)),
    ("3rd/4th & 3-6", lambda p: (p["down"] >= 3) & p["distance"].between(3, 6)),
    ("3rd/4th & 7+", lambda p: (p["down"] >= 3) & (p["distance"] >= 7)),
    ("Red zone (inside 20)", lambda p: p["yards_to_goal"] <= 20),
    ("Goal to go, inside 5", lambda p: p["yards_to_goal"] <= 5),
    ("Last 2:00 of a half", lambda p: p["period"].isin([2, 4]) & (p["clock_seconds"] <= 120)),
)
MIX_ORDER = ("RB run left", "RB run middle", "RB run right", "QB run", "WR/TE run",
             "Screen / behind LOS", "Short 1-9", "Intermediate 10-19", "Deep 20+",
             "Pass, no target", "Sack")
DEF_GROUPS = ("RB run left", "RB run middle", "RB run right", "QB run", "Screen / behind LOS",
              "Short 1-9", "Intermediate 10-19", "Deep 20+")
# Scopes: "P4" = ACC, Big Ten, Big 12 and SEC plus Notre Dame (the 68 charted teams); any other
# value is read as one conference name.
SCOPES = {"P4": {"conferences": ("ACC", "Big Ten", "Big 12", "SEC"), "teams": ("Notre Dame",),
                 "label": "P4 + Notre Dame"}}
CONF_SHORT = {"FBS Independents": "Independent"}
K_DEF = 20.0           # success allowed shrunk toward the FBS rate with this many plays
MIN_TOP_PLAY = 8       # plays for a top-play row
MIN_PLAYER = 10        # touches for a player row


# ----------------------------------------------------------------------------------------------
# Loading and labels


def scope_ids(ts: pd.DataFrame, scope: str) -> list[int]:
    """Team ids for a scope: "P4" (68 teams) or a single conference name."""
    sc = SCOPES.get(scope.upper())
    if sc is None:
        return sorted(ts.loc[ts["conference"] == scope, "team_id"].astype(int))
    m = ts["conference"].isin(sc["conferences"]) | ts["team"].isin(sc["teams"])
    return sorted(ts.loc[m, "team_id"].astype(int))


def load(data_dir="data", seasons=(2025, 2026)) -> dict:
    b = pm.load(data_dir, seasons)
    extra = pd.read_parquet(Path(data_dir) / "core" / "pbp.parquet", columns=[
        "play_id", "ppa", "score_diff", "clock_seconds", "offense_is_home", "is_rush", "is_pass"])
    b["plays"] = b["plays"].merge(extra, on="play_id", how="left")
    b["info"] = pm.player_info(b)
    return b


def label_plays(b: dict) -> pd.DataFrame:
    """One row per play with the ball carrier / target, position group and play-type labels."""
    p = b["plays"].copy()
    o = b["players"]
    o = o[(o["side"] == "offense") & o["role"].isin(["rusher", "target", "passer"])]
    o = o.drop(columns=["season"], errors="ignore").merge(p[["play_id", "offense_id", "season"]], on="play_id")
    o = o[o["team_id"] == o["offense_id"]]
    info = b["info"][["season", "athlete_id", "player", "position"]
                     + (["chart_align"] if "chart_align" in b["info"] else [])].copy()
    if "chart_align" in info:
        info["position"] = info["chart_align"].map(CHART_LABEL).where(info["chart_align"].notna(), info["position"])
        info = info.drop(columns=["chart_align"])
    o = o.merge(info, on=["season", "athlete_id"], how="left")
    first = lambda role: (o[o["role"] == role].drop_duplicates("play_id")
                          .set_index("play_id")[["athlete_id", "player", "position"]])
    rusher, target, passer = first("rusher"), first("target"), first("passer")
    p = p.join(rusher.add_prefix("rusher_"), on="play_id").join(target.add_prefix("target_"), on="play_id")
    p = p.join(passer[["athlete_id", "player"]].add_prefix("passer_"), on="play_id")
    p["formation"] = np.where(p["shotgun"].fillna(False).astype(bool), "Shotgun", "Under center")
    p["tempo"] = np.where(p["no_huddle"].fillna(False).astype(bool), "No huddle", "Huddle")
    p["shotgun"] = p["shotgun"].fillna(False).astype(bool)
    p["no_huddle"] = p["no_huddle"].fillna(False).astype(bool)
    p["action"] = p["action"].astype(object)
    p["run"] = (p["action"] == "rush").to_numpy(bool)
    rpos = p["rusher_position"].astype(object).map(POS_GROUP).astype(object)
    tpos = p["target_position"].astype(object).map(POS_GROUP).astype(object)
    air = pd.to_numeric(p["air_yards"], errors="coerce").astype(float).to_numpy()
    depth = pd.Series(np.select([air <= 0, air <= 9, air <= 19, air >= 20],
                                ["Screen / behind LOS", "Short 1-9", "Intermediate 10-19", "Deep 20+"],
                                default=""), index=p.index)
    rdir = p["run_dir"].astype(object).fillna("?").astype(str)
    is_pass = (p["action"] == "pass").to_numpy(bool)
    run = p["run"].to_numpy(bool)
    group = np.select(
        [(p["action"] == "sack").to_numpy(bool),
         run & (rpos == "QB").to_numpy(bool),
         run & rpos.isin(list(WR_GROUPS) + ["TE"]).to_numpy(bool),
         run,
         is_pass & (depth == "").to_numpy(bool),
         is_pass],
        ["Sack", "QB run", "WR/TE run", ("RB run " + rdir).to_numpy(object), "Pass, no target",
         depth.to_numpy(object)],
        default="Other")
    p["group"] = group
    p.loc[p["group"].isin(["RB run ?"]), "group"] = "RB run middle"
    p["pos"] = np.where(p["run"], rpos.fillna("?"), np.where(p["action"] == "pass", tpos.fillna("?"), ""))
    pdir = p["pass_dir"].astype(object).fillna("").astype(str)
    p["play"] = np.where(
        p["action"] == "sack", "Sack",
        np.where(p["run"],
                 np.where(rpos == "QB", "QB run " + rdir, rpos.fillna("?") + " run " + rdir),
                 np.where(depth == "", "Pass, no target",
                          depth + " " + pdir + " to " + tpos.fillna("?"))))
    p["ball_player"] = np.where(p["run"], p["rusher_player"], p["target_player"])
    p["ball_id"] = np.where(p["run"], p["rusher_athlete_id"], p["target_athlete_id"])
    p["dropback"] = p["action"].isin(["pass", "sack"])
    return p


# ----------------------------------------------------------------------------------------------
# League-wide fits


def _logit_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 30) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        q = 1 / (1 + np.exp(-z))
        g = X.T @ (q - y) + l2 * w
        H = (X * (q * (1 - q))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _proe_design(p: pd.DataFrame) -> np.ndarray:
    d = p["down"].clip(1, 4).astype(int).to_numpy()
    dist = np.log1p(p["distance"].clip(0, 30).astype(float).to_numpy())
    ytg = p["yards_to_goal"].astype(float).to_numpy() / 100
    sd = p["score_diff"].clip(-28, 28).astype(float).to_numpy() / 14
    per = p["period"].clip(1, 4).astype(int).to_numpy()
    late = ((per == 2) | (per == 4)) * np.clip(1 - p["clock_seconds"].astype(float).to_numpy() / 300, 0, 1)
    q4 = (per == 4).astype(float)
    cols = [np.ones(len(p))]
    for k in (2, 3, 4):
        cols.append((d == k).astype(float))
        cols.append((d == k) * dist)
    cols += [dist, ytg, ytg ** 2, (ytg <= 0.05).astype(float), sd, sd ** 2, sd * q4, late, late * sd,
             (per == 2).astype(float), (per == 3).astype(float), q4]
    return np.column_stack(cols)


def expected_pass(all_plays: pd.DataFrame) -> pd.Series:
    """FBS logistic model of pass (dropback) probability from situation only."""
    p = all_plays
    X = _proe_design(p)
    y = p["dropback"].to_numpy(float)
    w = _logit_fit(X, y)
    return pd.Series(1 / (1 + np.exp(-np.clip(X @ w, -30, 30))), index=p.index)


def crew_adjusted(p: pd.DataFrame, col: str, games: pd.DataFrame) -> pd.Series:
    """Offense effect from a weighted two-way fit share ~ offense + venue over (game, offense) rows,
    returned as the team's share at an average venue."""
    g = p.groupby(["game_id", "offense_id"]).agg(y=(col, "mean"), n=(col, "size")).reset_index()
    g = g.merge(games[["game_id", "venue_id"]], on="game_id", how="left")
    g["venue_id"] = g["venue_id"].fillna(-1).astype(int)
    g = g[g["n"] >= 20].reset_index(drop=True)
    offs = np.sort(g["offense_id"].unique())
    vens = np.sort(g["venue_id"].unique())
    oi = np.searchsorted(offs, g["offense_id"].to_numpy())
    vi = np.searchsorted(vens, g["venue_id"].to_numpy())
    X = np.zeros((len(g), len(offs) + len(vens)))
    X[np.arange(len(g)), oi] = 1
    X[np.arange(len(g)), len(offs) + vi] = 1
    sw = np.sqrt(g["n"].to_numpy(float))
    ridge = 1.0
    A = np.vstack([X * sw[:, None], np.sqrt(ridge) * np.eye(X.shape[1])])
    rhs = np.concatenate([g["y"].to_numpy() * sw, np.zeros(X.shape[1])])
    rhs[len(g) + len(offs):] = 0
    # ridge pulls venue effects to 0; offense effects get a weak pull to 0 too, so add back the mean
    beta, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    off = beta[: len(offs)]
    ven = beta[len(offs):]
    vmean = np.average(ven[vi], weights=g["n"])
    return pd.Series(np.clip(off + vmean, 0, 1), index=offs)


# ----------------------------------------------------------------------------------------------
# Sections


def _share(mask: pd.Series) -> float | None:
    return pm._r(mask.mean()) if len(mask) else None


def _mix_rows(t: pd.DataFrame, fbs: pd.DataFrame) -> list[dict]:
    rows = []
    n, nf = len(t), len(fbs)
    for gname in MIX_ORDER:
        s = t[t["group"] == gname]
        f = fbs[fbs["group"] == gname]
        if not len(s) and not len(f):
            continue
        rows.append({
            "group": gname, "n": int(len(s)), "share": pm._r(len(s) / n) if n else None,
            "fbs_share": pm._r(len(f) / nf) if nf else None,
            "success": pm._r(s["success"].astype(float).mean()) if len(s) else None,
            "fbs_success": pm._r(f["success"].astype(float).mean()) if len(f) else None,
            "epa": pm._r(s["ppa"].mean()) if len(s) else None,
            "fbs_epa": pm._r(f["ppa"].mean()) if len(f) else None,
            "dir": {d: int((s["run_dir"] == d).sum() if gname.startswith(("QB", "WR")) else (s["pass_dir"] == d).sum())
                    for d in pm.DIRS} if gname in ("QB run", "WR/TE run", "Screen / behind LOS", "Short 1-9",
                                                    "Intermediate 10-19", "Deep 20+") else None,
        })
    return rows


def _situation_rows(t: pd.DataFrame, fbs: pd.DataFrame) -> list[dict]:
    rows = []
    for name, fn in SITUATIONS:
        s = t[fn(t)]
        f = fbs[fn(fbs)]
        if not len(s):
            continue
        top = s["group"].value_counts()
        top_play = s["play"].value_counts()
        rows.append({
            "situation": name, "n": int(len(s)),
            "run_rate": _share(s["run"]), "fbs_run_rate": _share(f["run"]),
            "exp_pass": pm._r(s["xpass"].mean()), "pass_rate": _share(s["dropback"]),
            "shotgun": _share(s["shotgun"]), "fbs_shotgun": _share(f["shotgun"]),
            "deep": _share(s["group"] == "Deep 20+"), "screen": _share(s["group"] == "Screen / behind LOS"),
            "top_group": top.index[0], "top_group_share": pm._r(top.iloc[0] / len(s)),
            "top_play": top_play.index[0], "top_play_share": pm._r(top_play.iloc[0] / len(s)),
            "success": pm._r(s["success"].astype(float).mean()), "fbs_success": pm._r(f["success"].astype(float).mean()),
            "epa": pm._r(s["ppa"].mean()),
        })
    return rows


def _top_plays(t: pd.DataFrame, n_top: int = 12) -> list[dict]:
    key = ["formation", "play"]
    g = t[~t["play"].isin(["Sack", "Pass, no target"])].groupby(key)
    sizes = g.size().sort_values(ascending=False)
    rows = []
    for (form, play), n in sizes.items():
        if n < MIN_TOP_PLAY or len(rows) >= n_top:
            break
        s = t[(t["formation"] == form) & (t["play"] == play)]
        who = s["ball_player"].dropna().value_counts()
        downs = s["down"].clip(1, 3).value_counts(normalize=True)
        rows.append({
            "formation": form, "play": play, "n": int(n), "share": pm._r(n / len(t)),
            "no_huddle": _share(s["no_huddle"]),
            "success": pm._r(s["success"].astype(float).mean()), "epa": pm._r(s["ppa"].mean()),
            "yards": pm._r(s["yards_gained"].mean(), 1),
            "top_player": who.index[0] if len(who) else None,
            "top_player_share": pm._r(who.iloc[0] / len(s)) if len(who) else None,
            "downs": {str(int(k)) if k < 3 else "3+": pm._r(v) for k, v in downs.items()},
        })
    return rows


def _injury_index(maps: dict | None) -> dict[int, tuple]:
    """athlete_id -> (status, last update) from the player maps' CFBDepth injury fields."""
    out = {}
    if maps:
        for key in ("offense", "rb", "defense"):
            for r in maps.get(key, {}).get("players", []):
                if r.get("injury_status") and r.get("athlete_id") is not None:
                    out[int(r["athlete_id"])] = (r["injury_status"], r.get("injury_date"))
    return out


def _chart_label(chart: dict, aid) -> str | None:
    c = chart.get(int(aid)) if aid is not None and not pd.isna(aid) else None
    if not c or c[0] is None:
        return None
    return f"{c[0]} {int(c[1])}" if c[1] is not None and not pd.isna(c[1]) else str(c[0])


def _players(t: pd.DataFrame, maps: dict | None, chart: dict | None = None,
             injuries: dict | None = None) -> list[dict]:
    chart, injuries = chart or {}, injuries or {}
    style = {}
    if maps:
        for r in maps.get("offense", {}).get("players", []):
            style[int(r["athlete_id"])] = r.get("style")
        for r in maps.get("rb", {}).get("players", []):
            if r.get("trouble"):
                style.setdefault(int(r["athlete_id"]), None)
    rows = []
    touched = t[t["ball_id"].notna()]
    for aid, s in touched.groupby("ball_id"):
        if len(s) < MIN_PLAYER:
            continue
        runs, tg = s[s["run"]], s[~s["run"]]
        pos = (runs["rusher_position"].dropna().tolist() + tg["target_position"].dropna().tolist() or ["?"])[0]
        top = s["play"].value_counts()
        rows.append({
            "player": s["ball_player"].dropna().iloc[0], "athlete_id": int(aid), "pos": pos,
            "touches": int(len(s)), "rushes": int(len(runs)), "targets": int(len(tg)),
            "share": pm._r(len(s) / len(t)),
            "shotgun": _share(s["shotgun"]),
            "top_play": top.index[0], "top_play_share": pm._r(top.iloc[0] / len(s)),
            "success": pm._r(s["success"].astype(float).mean()),
            "style": style.get(int(aid)),
            "chart": _chart_label(chart, aid),
            "injury": injuries.get(int(aid), (None,))[0],
        })
    rows.sort(key=lambda r: -r["touches"])
    return rows[:10]


def _defenders_from_plays(dp: pd.DataFrame, chart: dict, injuries: dict) -> list[dict]:
    """Most active defenders from play credits when the player maps do not cover the team."""
    if dp is None or not len(dp):
        return []
    rows = []
    for aid, s in dp.groupby("athlete_id"):
        roles = s["role"].value_counts()
        rows.append({"player": s["player"].dropna().iloc[0] if s["player"].notna().any() else s["name"].iloc[0],
                     "pos": s["position"].dropna().iloc[0] if s["position"].notna().any() else None,
                     "chart_pos": _chart_label(chart, aid), "events": int(s["play_id"].nunique()),
                     "pred_role": None, "pred_p": None, "location": None, "alignment": None,
                     "sacks": int(roles.get("sacker", 0)), "hurries": int(roles.get("hurry", 0)),
                     "pbu": int(roles.get("breakup", 0)), "ints": int(roles.get("interceptor", 0)),
                     "injury": injuries.get(int(aid), (None,))[0]})
    rows.sort(key=lambda r: -r["events"])
    return rows[:8]


def _defense(d: pd.DataFrame, fbs: pd.DataFrame, maps: dict | None, team: str,
             dp: pd.DataFrame | None = None, chart: dict | None = None, injuries: dict | None = None) -> dict:
    chart, injuries = chart or {}, injuries or {}
    rows = []
    for gname in DEF_GROUPS:
        s = d[d["group"] == gname]
        f = fbs[fbs["group"] == gname]
        base = float(f["success"].astype(float).mean())
        k = float(s["success"].astype(float).sum())
        shr = (k + K_DEF * base) / (len(s) + K_DEF)
        rows.append({"group": gname, "n": int(len(s)), "share_faced": pm._r(len(s) / len(d)),
                     "fbs_share": pm._r(len(f) / len(fbs)),
                     "success_allowed": pm._r(s["success"].astype(float).mean()) if len(s) else None,
                     "shrunk": pm._r(shr), "fbs_success": pm._r(base),
                     "epa_allowed": pm._r(s["ppa"].mean()) if len(s) else None})
    db = d[d["dropback"]]
    fdb = fbs[fbs["dropback"]]
    defenders = []
    if maps:
        ds = [r for r in maps.get("defense", {}).get("players", []) if r.get("team") == team]
        ds.sort(key=lambda r: -r.get("events", 0))
        defenders = [{k: r.get(k) for k in ("player", "pos", "chart_pos", "events", "pred_role", "pred_p", "location",
                                              "alignment", "sacks", "hurries", "pbu", "ints")} for r in ds[:8]]
        for r, src in zip(defenders, ds[:8]):
            r["injury"] = src.get("injury_status") or injuries.get(int(src.get("athlete_id") or -1), (None,))[0]
            r["chart_pos"] = _chart_label(chart, src.get("athlete_id")) or r.get("chart_pos")
    if not defenders:
        defenders = _defenders_from_plays(dp, chart, injuries)
    return {
        "plays": int(len(d)), "success_allowed": pm._r(d["success"].astype(float).mean()),
        "fbs_success": pm._r(fbs["success"].astype(float).mean()),
        "epa_allowed": pm._r(d["ppa"].mean()), "fbs_epa": pm._r(fbs["ppa"].mean()),
        "sack_rate": _share(db["action"] == "sack"), "fbs_sack_rate": _share(fdb["action"] == "sack"),
        "hurry_rate": _share(db["qb_hurried"].fillna(False).astype(bool)),
        "fbs_hurry_rate": _share(fdb["qb_hurried"].fillna(False).astype(bool)),
        "opp_shotgun": _share(d["shotgun"]), "opp_run_rate": _share(d["run"]),
        "opp_proe": pm._r(d["dropback"].mean() - d["xpass"].mean()),
        "groups": rows, "defenders": defenders,
    }


def _summary(o: dict) -> list[str]:
    """Plain-language lines from the numbers (rules, not a model)."""
    f, lines = o["formation"], []
    uc = f["under_center"]
    if uc >= 0.15:
        lines.append(f"Uses under center a lot for FBS ({uc:.0%} of plays, FBS {f['fbs_under_center']:.0%}).")
    elif uc <= 0.02:
        lines.append(f"Almost always in shotgun ({1 - uc:.0%} of plays).")
    if f["run_rate_uc"] is not None and uc >= 0.03 and f["n_uc"] >= 20:
        lines.append(f"Under center it runs {f['run_rate_uc']:.0%} of the time; from shotgun {f['run_rate_sg']:.0%}.")
    nh = f["no_huddle_adj"]
    if nh is not None:
        if nh >= f["fbs_no_huddle"] + 0.2:
            lines.append(f"Up-tempo: no huddle on about {nh:.0%} of plays after the stat-crew adjustment.")
        elif nh <= f["fbs_no_huddle"] - 0.2:
            lines.append(f"Huddles more than most: no huddle on about {nh:.0%} after the crew adjustment.")
    proe = o["proe"]
    if proe >= 0.04:
        lines.append(f"Passes more than the situation predicts (pass rate over expected +{proe:.1%}).")
    elif proe <= -0.04:
        lines.append(f"Runs more than the situation predicts (pass rate over expected {proe:.1%}).")
    mix = {r["group"]: r for r in o["mix"]}
    for gname, label in (("Screen / behind LOS", "screens and passes behind the line"),
                         ("Deep 20+", "deep shots (20+ air yards)"), ("QB run", "QB runs")):
        r = mix.get(gname)
        if r and r["share"] is not None and r["fbs_share"] and r["share"] >= 1.4 * r["fbs_share"] and r["n"] >= 25:
            lines.append(f"Heavy on {label}: {r['share']:.0%} of plays vs {r['fbs_share']:.0%} FBS "
                         f"({r['success']:.0%} success).")
    rb = [mix[k] for k in ("RB run left", "RB run middle", "RB run right") if k in mix]
    if rb:
        tot = sum(r["n"] for r in rb)
        if tot >= 60:
            top = max(rb, key=lambda r: r["n"])
            if top["n"] / tot >= 0.45:
                lines.append(f"RB runs go {top['group'].split()[-1]} {top['n'] / tot:.0%} of the time.")
    return lines


def team_playbook(t: pd.DataFrame, fbs: pd.DataFrame, d: pd.DataFrame, fbs_def: pd.DataFrame,
                  nh_adj: float | None, maps: dict | None, team: str, conference: str | None = None,
                  dp: pd.DataFrame | None = None, chart: dict | None = None,
                  injuries: dict | None = None) -> dict:
    sg = t[t["shotgun"]]
    uc = t[~t["shotgun"]]
    o = {
        "team": team, "conference": conference,
        "plays": int(len(t)), "games": int(t["game_id"].nunique()),
        "formation": {
            "shotgun": _share(t["shotgun"]), "under_center": _share(~t["shotgun"]),
            "fbs_under_center": _share(~fbs["shotgun"]),
            "no_huddle_raw": _share(t["no_huddle"]), "no_huddle_adj": pm._r(nh_adj),
            "fbs_no_huddle": _share(fbs["no_huddle"]),
            "run_rate_sg": _share(sg["run"]), "run_rate_uc": _share(uc["run"]) if len(uc) else None,
            "n_uc": int(len(uc)),
            "fbs_run_rate_sg": _share(fbs.loc[fbs["shotgun"], "run"]),
            "fbs_run_rate_uc": _share(fbs.loc[~fbs["shotgun"], "run"]),
            "success_sg": pm._r(sg["success"].astype(float).mean()),
            "success_uc": pm._r(uc["success"].astype(float).mean()) if len(uc) else None,
        },
        "run_rate": _share(t["run"]), "fbs_run_rate": _share(fbs["run"]),
        "proe": pm._r(t["dropback"].mean() - t["xpass"].mean()),
        "mix": _mix_rows(t, fbs),
        "situations": _situation_rows(t, fbs),
        "top_plays": _top_plays(t),
        "players": _players(t, maps, chart, injuries),
        "defense": _defense(d, fbs_def, maps, team, dp, chart, injuries),
    }
    o["summary"] = _summary(o)
    return o


def _league_row(t: dict) -> dict:
    mix = {r["group"]: r for r in t["mix"]}
    share = lambda g: mix[g]["share"] if g in mix else 0.0
    f = t["formation"]
    return {
        "team": t["team"], "conference": t.get("conference"), "plays": t["plays"], "under_center": f["under_center"],
        "under_center_adj": f.get("under_center_adj"), "run_rate_uc": f["run_rate_uc"],
        "no_huddle_adj": f["no_huddle_adj"], "run_rate": t["run_rate"], "proe": t["proe"],
        "rb_run": sum(share(g) for g in ("RB run left", "RB run middle", "RB run right")),
        "qb_run": share("QB run"), "screen": share("Screen / behind LOS"), "deep": share("Deep 20+"),
        "success": pm._r(np.average([r["success"] or 0 for r in t["mix"]], weights=[r["n"] for r in t["mix"]])),
        "epa": pm._r(np.average([r["epa"] or 0 for r in t["mix"]], weights=[r["n"] for r in t["mix"]])),
        "def_success": t["defense"]["success_allowed"], "def_epa": t["defense"]["epa_allowed"],
        "def_sack_rate": t["defense"]["sack_rate"],
    }


# ----------------------------------------------------------------------------------------------
# Build


def build(data_dir="data", season: int = 2026, conference: str = "P4", maps_path: str | None = None) -> dict:
    b = load(data_dir, (season,))
    p = label_plays(b)
    p = p[~p["garbage_time"].fillna(False).astype(bool)].reset_index(drop=True)
    p["xpass"] = expected_pass(p)
    teams = b["teams"]
    ts = teams[teams["season"] == season]
    fbs_ids = set(ts.loc[ts["fbs"].astype(bool), "team_id"].astype(int))
    conf_ids = scope_ids(ts, conference)
    name = {int(i): n for i, n in zip(ts["team_id"], ts["team"])}
    conf_of = {int(i): CONF_SHORT.get(c, c) for i, c in zip(ts["team_id"], ts["conference"])}
    fbs = p[p["offense_id"].isin(fbs_ids)]
    fbs_def = p[p["defense_id"].isin(fbs_ids)]
    scope = p[p["offense_id"].isin(fbs_ids) | p["defense_id"].isin(fbs_ids)].assign(uc=lambda x: ~x["shotgun"])
    nh = crew_adjusted(scope, "no_huddle", b["games"])
    uc = crew_adjusted(scope, "uc", b["games"])
    maps = None
    mp = Path(maps_path) if maps_path else Path(data_dir) / "lab" / f"player_maps_{season}_{conference.lower()}.json"
    if mp.exists():
        maps = json.loads(mp.read_text(encoding="utf-8"))
    injuries = _injury_index(maps)
    info = b["info"][b["info"]["season"] == season]
    chart = {}
    if "chart_pos" in info:
        ci = info.dropna(subset=["chart_pos"])
        chart = {int(a): (pos, dep) for a, pos, dep in zip(ci["athlete_id"], ci["chart_pos"], ci["chart_depth"])}
    dpl = b["players"]
    dpl = dpl[(dpl["side"] == "defense") & dpl["play_id"].isin(p["play_id"])].drop(columns=["season"], errors="ignore")
    dpl = dpl.merge(p[["play_id", "defense_id"]], on="play_id")
    dpl = dpl[dpl["team_id"] == dpl["defense_id"]].merge(
        info[["athlete_id", "player", "position"]].drop_duplicates("athlete_id"), on="athlete_id", how="left")
    out = []
    for tid in conf_ids:
        t = p[p["offense_id"] == tid]
        d = p[p["defense_id"] == tid]
        if not len(t):
            continue
        tb = team_playbook(t, fbs, d, fbs_def, float(nh.get(tid, np.nan)), maps, name[tid], conf_of[tid],
                           dpl[dpl["defense_id"] == tid], chart, injuries)
        tb["formation"]["under_center_adj"] = pm._r(uc.get(tid, np.nan))
        out.append(tb)
    weeks = sorted(p["week"].unique().tolist())
    return {
        "meta": {
            "season": season, "conference": conference,
            "scope_label": SCOPES.get(conference.upper(), {}).get("label", conference),
            "conferences": sorted({conf_of[i] for i in conf_ids}),
            "injuries": len(injuries) > 0,
            "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "weeks": [int(weeks[0]), int(weeks[-1])] if weeks else None,
            "plays_fbs": int(len(fbs)), "player_maps": maps is not None,
            "fbs_run_rate": _share(fbs["run"]), "fbs_under_center": _share(~fbs["shotgun"]),
            "fbs_no_huddle": _share(fbs["no_huddle"]),
            "fbs_mix": _mix_rows(fbs, fbs),
        },
        "teams": out,
        "league": [_league_row(t) for t in out],
    }


def write(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = result["meta"]
    stem = f"playbooks_{m['season']}_{m['conference'].lower()}"
    payload = pm._clean(result)
    js = out_dir / f"{stem}.json"
    js.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    template = Path(__file__).with_name("playbooks_viewer.html").read_text(encoding="utf-8")
    html = out_dir / f"{stem}.html"
    html.write_text(template.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"))
                                     .replace("</", "<\\/")), encoding="utf-8")
    return js, html


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--conference", default="P4",
                    help="P4 = ACC, Big Ten, Big 12, SEC + Notre Dame (68 teams), or one conference name")
    ap.add_argument("--player-maps", default=None,
                    help="player maps JSON (default data/lab/player_maps_<season>_<conference lower>.json)")
    ap.add_argument("--out", default=None, help="output folder (default data/lab)")
    a = ap.parse_args(argv)
    res = build(a.data_dir, a.season, a.conference, a.player_maps)
    out = Path(a.out) if a.out else Path(a.data_dir) / "lab"
    js, html = write(res, out)
    for t in res["teams"]:
        f = t["formation"]
        print(f"{t['team']:<18} plays {t['plays']:>4}  UC {f['under_center']:.2f}  NH raw {f['no_huddle_raw']:.2f} "
              f"adj {f['no_huddle_adj']:.2f}  PROE {t['proe']:+.3f}  run {t['run_rate']:.2f}")
    print(js)
    print(html)
    return 0


if __name__ == "__main__":
    sys.exit(main())

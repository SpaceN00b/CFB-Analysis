"""Defensive player impact scores (test bench, 2026-11-20; extends PR 35's depth-chart impact scores).

PR 26 priced a defender by his share of team tackles and PR 34 by CFBDepth snap share; both found
no measurable cost. Tackles mostly measure how often the ball carrier reached him, and snap share
says who plays, not how well. This module prices a defender by the plays he wins, in points:

* Disruption value per player-game (box score, 2016 on, every FBS game): sacks, non-sack tackles
  for loss, interceptions, passes defended (box-score PD excludes INTs) and QB hurries, each at the
  mean EPA swing of that play against an average play of its kind (dropback or rush), from CFBD
  play-by-play and /plays/stats, 2018-2024: sack 1.45, TFL 1.29, INT 4.80, PBU 0.93. Hurries are
  0.5 by assumption: /plays/stats attaches hurries only to incompletions (2025 on), so their EPA
  swing cannot be measured from it.
* Groups and regulars: DL (roster DL/DE/DT/NT/EDGE, chart IDL/EDGE), LB, DB (CB/S/NB). Regulars per
  team-game, pregame: DL 4, LB 3, DB 5 with the most defensive involvement so far this season
  (tackles + PD + hurries + INT, at least 2 appearances), the `dc_total` rule of PR 35: a starter who
  has been out long enough for his backup to pass him drops out.
* Starter rate: (disruption this season + 0.5 x last season + K x league starter mean) /
  (games + 0.5 x last-season games + K), K games from split-half reliability (`reliability_k`).
* Replacement: "fill" = the league mean disruption per game of the player who filled in for an out
  regular at that group (measured in those games, 2018-2024); "own" = the next healthy non-regular
  in the group, his disruption per involvement times the starter's involvement per game, shrunk
  toward the fill mean with K. With a chart (this week's picks), the next man up at the starter's
  chart label, priced by the "own" rule (a starter not on the chart falls back to the usage pick).
* Cost = max(starter rate - replacement rate, 0) x P(out), in points against the team's margin.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cfb.lab import availability as A

DGROUPS = ("DL", "LB", "DB")
N_START = {"DL": 4, "LB": 3, "DB": 5}
CHART_GRP = {"IDL": "DL", "EDGE": "DL", "LB": "LB", "CB": "DB", "S": "DB", "NB": "DB"}
EV_HURRY = 0.5          # assumption (see module doc)
PREV_W = 0.5
MIN_GAMES = 2
SCALE_CAP = 4.0         # "own" replacement: at most 4x his involvement per game
INT_PLAYS = ("Pass Interception Return", "Interception", "Interception Return Touchdown")
STATS = ("tot", "sacks", "tfl", "pd", "hur", "int")


# ---------------------------------------------------------------------------- values and player games

def event_values(data: Path, seasons=range(2018, 2025)) -> dict:
    """Mean EPA swing (points, defense's view) of each event against an average play of its kind."""
    p = pd.read_parquet(data / "core" / "pbp.parquet", columns=[
        "play_id", "season", "play_type", "is_rush", "is_pass", "is_scrimmage", "ppa", "yards_gained", "garbage_time"])
    p = p[p["season"].isin(list(seasons)) & p["is_scrimmage"].fillna(False).astype(bool)
          & ~p["garbage_time"].fillna(False).astype(bool) & p["ppa"].notna()]
    drop = p[p["is_pass"].fillna(False).astype(bool)]
    rush = p[p["is_rush"].fillna(False).astype(bool)]
    st = A._parts(data, "plays_stats", seasons)
    pbu = set(st.loc[st["statType"] == "Pass Breakup", "playId"].astype(str))
    md, mr = drop["ppa"].mean(), rush["ppa"].mean()
    return {"sack": float(md - drop.loc[drop["play_type"] == "Sack", "ppa"].mean()),
            "int": float(md - drop.loc[drop["play_type"].isin(INT_PLAYS), "ppa"].mean()),
            "pbu": float(md - drop.loc[drop["play_id"].astype(str).isin(pbu), "ppa"].mean()),
            "tfl": float(mr - rush.loc[rush["yards_gained"] < 0, "ppa"].mean()),
            "hurry": EV_HURRY}


def disruption(df: pd.DataFrame, ev: dict) -> pd.Series:
    """Points per row from box-score counts (sacks are inside TFL; PD excludes INTs)."""
    return (ev["sack"] * df["sacks"] + ev["tfl"] * (df["tfl"] - df["sacks"]).clip(lower=0)
            + ev["int"] * df["int"] + ev["pbu"] * df["pd"] + ev["hurry"] * df["hur"])


def defender_games(data: Path, u: pd.DataFrame, ev: dict, chart: pd.DataFrame | None = None,
                   teams: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (game_id, team_id, athlete_id) appearance of a DL / LB / DB (presence = any
    box-score or /plays/stats row, PR 34's player games `u`), with box-score defensive counts,
    involvement (`inv`) and disruption points (`dv`). Group: roster position of that season; for
    2026, the player's 2026 defensive chart role where charted."""
    seasons = sorted(u["season"].unique())
    if teams is None:
        teams = pd.read_parquet(data / "core" / "teams.parquet")
    tid = {(s, t): i for s, t, i in zip(teams["season"], teams["team"], teams["team_id"])}
    gp = A._parts(data, "games_players", seasons)
    gp = gp[gp["category"].isin(["defensive", "interceptions"]) & gp["athleteId"].notna()]
    gp = gp[~gp["athleteId"].astype(str).str.startswith("-")]
    gp["athlete_id"] = pd.to_numeric(gp["athleteId"], errors="coerce")
    gp = gp[gp["athlete_id"].notna() & (gp["athlete_id"] > 0)]
    gp["team_id"] = [tid.get((s, t)) for s, t in zip(gp["season"], gp["team"])]
    gp = gp[gp["team_id"].notna()]
    gp["game_id"] = pd.to_numeric(gp["gameId"], errors="coerce")
    key = {("defensive", "TOT"): "tot", ("defensive", "SACKS"): "sacks", ("defensive", "TFL"): "tfl",
           ("defensive", "PD"): "pd", ("defensive", "QB HUR"): "hur", ("interceptions", "INT"): "int"}
    gp["k"] = [key.get((c, s)) for c, s in zip(gp["category"], gp["statType"])]
    gp = gp[gp["k"].notna()]
    gp["v"] = pd.to_numeric(gp["stat"], errors="coerce").fillna(0.0)
    w = gp.pivot_table(index=["game_id", "team_id", "athlete_id"], columns="k", values="v", aggfunc="sum",
                       fill_value=0.0).reset_index()
    for c in STATS:
        if c not in w:
            w[c] = 0.0
    w = w.astype({"game_id": "int64", "team_id": "int64", "athlete_id": "int64"})
    d = u[u["grp"].isin(DGROUPS)][["season", "game_id", "team_id", "athlete_id", "start_time_utc", "grp"]]
    d = d.merge(w[["game_id", "team_id", "athlete_id", *STATS]], on=["game_id", "team_id", "athlete_id"], how="left")
    d[list(STATS)] = d[list(STATS)].fillna(0.0)
    if chart is not None and len(chart):
        c = chart[(chart["unit"] == "defense") & chart["athlete_id"].notna() & chart["role"].isin(list(CHART_GRP))]
        c = c.sort_values("depth").drop_duplicates("athlete_id")
        cg = dict(zip(c["athlete_id"].astype("int64"), c["role"].map(CHART_GRP)))
        m = d["season"] == 2026
        d.loc[m, "grp"] = [cg.get(a, g) for a, g in zip(d.loc[m, "athlete_id"], d.loc[m, "grp"])]
    d["inv"] = d["tot"] + d["pd"] + d["hur"] + d["int"]
    d["dv"] = disruption(d, ev)
    # a player on two lines of a game (two groups) keeps one
    return d.drop_duplicates(["game_id", "team_id", "athlete_id"]).reset_index(drop=True)


def prev_season(dg: pd.DataFrame) -> pd.DataFrame:
    """(season S, athlete_id) -> last season's (S-1) games and disruption, any team."""
    s = dg.groupby(["season", "athlete_id"]).agg(prev_gp=("dv", "size"), prev_dv=("dv", "sum")).reset_index()
    s["season"] = s["season"] + 1
    return s


def season_regulars(dg: pd.DataFrame) -> pd.DataFrame:
    """Season top-N per team and group by total involvement (for league means and reliability)."""
    s = dg.groupby(["season", "team_id", "grp", "athlete_id"]).agg(gp=("dv", "size"), inv=("inv", "sum"),
                                                                    dv=("dv", "sum")).reset_index()
    s = s[s["gp"] >= MIN_GAMES].sort_values("inv", ascending=False)
    s["rank"] = s.groupby(["season", "team_id", "grp"]).cumcount()
    s["regular"] = s["rank"] < s["grp"].map(N_START)
    return s


def reliability_k(dg: pd.DataFrame, seasons=range(2018, 2025), min_games: int = 8) -> dict:
    """K (games) per group from odd / even game split-half correlation of disruption per game among
    season regulars: reliability of an n-game mean = n / (n + K)."""
    x = dg[dg["season"].isin(list(seasons))].sort_values("start_time_utc").copy()
    x["k"] = x.groupby(["season", "team_id", "athlete_id"]).cumcount()
    reg = season_regulars(x)
    reg = reg[reg["regular"] & (reg["gp"] >= min_games)][["season", "team_id", "athlete_id", "grp", "gp"]]
    x = x.merge(reg[["season", "team_id", "athlete_id"]], on=["season", "team_id", "athlete_id"])
    h = x.assign(half=x["k"] % 2).groupby(["season", "team_id", "athlete_id", "grp", "half"])["dv"].mean().unstack()
    out = {}
    for g in DGROUPS:
        hh = h.xs(g, level="grp").dropna()
        r = float(np.corrcoef(hh[0], hh[1])[0, 1])
        n_half = float(reg.loc[reg["grp"] == g, "gp"].median()) / 2.0
        out[g] = n_half * (1 - r) / r if r > 0 else 1e6
    return out


# ---------------------------------------------------------------------------- per-game costs

def _label_of(crep: dict, athlete: int):
    best = None
    for lab, x in crep.items():
        ids = x.get("all", x["ids"])
        if athlete in ids:
            d = ids.index(athlete)
            if best is None or d < best[0]:
                best = (d, lab)
    return best[1] if best else None


def team_costs(dg: pd.DataFrame, sched: pd.DataFrame, mu: dict, K: dict, p_out: dict | None, realized: bool,
               seasons, regular: set | None = None, record_all: bool = False,
               chart_repl: dict | None = None, streak: bool = False,
               prev: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (season, game_id, team_id) with >= 2 earlier team games: {dl,lb,db}_cost (fill
    replacement), {..}_cost_own ("own" replacement, or the chart's next man up), {..}_n_out, plus detail
    rows for every regular with P(out) > 0 (all regulars of the last game with record_all).

    mu: {group: {"start": m, "fill": m}} disruption points per game. K: {group: games}.
    p_out: {(game_id, team_id, athlete_id): probability} (listed mode). realized: a regular with no
    box-score line in a regular-season game (game_id in `regular`) is out; with streak=True only when he
    also has no line in the team's previous or next game (a full-time defender has no box-score line in
    about 4% of the games he plays, CFBDepth 2026 snaps; an isolated blank game is read as played).
    chart_repl (charted teams): {team_id: {label: {"role": role, "ids": [athlete ids by depth]}}} for
    the defense; the replacement is the next man up at the starter's label who is not out and not a
    healthy regular.
    prev: `prev_season(dg)` computed beforehand (the local app passes the exported table, since its
    `dg` holds only this season's games of the week's teams)."""
    prev = (prev_season(dg) if prev is None else prev).set_index(["season", "athlete_id"])
    rows, detail = [], []
    dgs = {k: v for k, v in dg[dg["season"].isin(list(seasons))].groupby(["season", "team_id"])}
    grp_s = {s: g.drop_duplicates("athlete_id").set_index("athlete_id")["grp"] for s, g in dg.groupby("season")}
    for (season, team), g in sched[sched["season"].isin(list(seasons))].groupby(["season", "team_id"], sort=False):
        games = g.sort_values("start_time_utc")["game_id"].to_numpy()
        n = len(games)
        if n < 3:
            continue
        gidx = {gid: i for i, gid in enumerate(games)}
        pu = dgs.get((season, team))
        if pu is None:
            continue
        pu = pu[pu["game_id"].isin(gidx)]
        crep = chart_repl.get(team) if chart_repl else None
        extra = set()
        for x_ in (crep or {}).values():
            extra.update(x_["ids"])
        gmap = grp_s.get(season)
        if crep:
            # charted players who have not played this season take their group from the chart role
            cr = {a: CHART_GRP.get(x_["role"]) for x_ in crep.values() for a in x_["ids"]}
            gmap = pd.concat([gmap, pd.Series({a: g_ for a, g_ in cr.items() if g_ and a not in gmap.index},
                                              dtype=object)])
        ath = np.array(sorted(set(pu["athlete_id"]) | {a for a in extra if a in gmap.index}))
        if not len(ath):
            continue
        aidx = {a: i for i, a in enumerate(ath)}
        m = len(ath)
        P, INV, DV = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))
        ri = pu["athlete_id"].map(aidx).to_numpy()
        ci = pu["game_id"].map(gidx).to_numpy()
        P[ri, ci] = 1.0
        INV[ri, ci] = pu["inv"].to_numpy()
        DV[ri, ci] = pu["dv"].to_numpy()
        cP, cI, cD = (np.cumsum(X, axis=1) - X for X in (P, INV, DV))
        grp = gmap.reindex(ath).to_numpy()
        pv = prev.reindex(pd.MultiIndex.from_arrays([np.full(m, season), ath]))
        pgp = np.nan_to_num(pv["prev_gp"].to_numpy(dtype=float))
        pdv = np.nan_to_num(pv["prev_dv"].to_numpy(dtype=float))
        for k in range(2, n):
            gid = games[k]
            gp = cP[:, k]
            if realized:
                po = (1.0 - P[:, k]) if (regular is None or gid in regular) else np.zeros(m)
                if streak:
                    nb = (1.0 - P[:, k - 1]) if k + 1 >= n else np.maximum(1.0 - P[:, k - 1], 1.0 - P[:, k + 1])
                    po = po * nb
            else:
                po = np.array([p_out.get((gid, team, a), 0.0) for a in ath]) if p_out else np.zeros(m)
            row = {"season": season, "game_id": gid, "team_id": team}
            starters_all, per = set(), {}
            for G in DGROUPS:
                ok = (grp == G) & (gp >= MIN_GAMES) & (cI[:, k] > 0)
                pool = np.where(ok)[0]
                order = pool[np.argsort(-cI[pool, k], kind="stable")]
                per[G] = list(order[:N_START[G]])
                starters_all.update(per[G])
            healthy = {si for si in starters_all if po[si] < 0.5}
            for G in DGROUPS:
                gl = G.lower()
                ms, mf, kk = mu[G]["start"], mu[G]["fill"], K[G]
                starters = per[G]
                rate_s = (cD[:, k] + PREV_W * pdv + kk * ms) / (gp + PREV_W * pgp + kk)
                bench = [i for i in np.argsort(-cI[:, k], kind="stable")
                         if grp[i] == G and gp[i] >= 1 and cI[i, k] > 0 and i not in starters_all and po[i] < 0.5]
                used = set()
                cost = cost_own = n_out = 0.0
                for si in starters:
                    pick, lab = None, None
                    if crep:
                        lab = _label_of(crep, int(ath[si]))
                        for a_ in (crep.get(lab) or {}).get("ids", []):
                            c_ = aidx.get(a_)
                            if c_ is None or c_ == si or c_ in used or c_ in healthy or po[c_] >= 0.5:
                                continue
                            pick = c_
                            break
                    if pick is None and (not crep or lab is None):
                        # not charted (or not on this week's chart): next healthy non-regular by usage
                        pick = next((c for c in bench if c not in used), None)
                    if po[si] >= 0.5 and pick is not None:
                        used.add(pick)
                    if pick is not None and gp[pick] >= 1 and cI[pick, k] > 0:
                        inv_s = cI[si, k] / max(gp[si], 1)
                        inv_r = cI[pick, k] / gp[pick]
                        scale = float(np.clip(inv_s / max(inv_r, 1e-9), 1.0, SCALE_CAP))
                        rr = (cD[pick, k] * scale + kk * mf) / (gp[pick] + kk)
                    else:
                        rr = mf
                    vor_f = max(rate_s[si] - mf, 0.0)
                    vor_o = max(rate_s[si] - rr, 0.0)
                    cost += vor_f * po[si]
                    cost_own += vor_o * po[si]
                    n_out += po[si]
                    if po[si] > 0 or (record_all and k == n - 1):
                        detail.append((season, gid, team, G, int(ath[si]), float(po[si]), float(gp[si]),
                                       float(cI[si, k] / max(gp[si], 1)), float(cD[si, k] / max(gp[si], 1)),
                                       float(rate_s[si]), int(ath[pick]) if pick is not None else -1, float(rr),
                                       float(vor_f), float(vor_o)))
                row.update({f"{gl}_cost": cost, f"{gl}_cost_own": cost_own, f"{gl}_n_out": n_out})
            rows.append(row)
    det = pd.DataFrame(detail, columns=["season", "game_id", "team_id", "grp", "athlete_id", "p_out", "games",
                                        "inv_pg", "dv_pg", "rate", "repl_id", "repl_rate", "vor_fill", "vor_own"])
    return pd.DataFrame(rows), det


def league_means(dg: pd.DataFrame, sched: pd.DataFrame, regular: set, seasons=range(2018, 2025)) -> dict:
    """Per group: "start" = disruption per game of season regulars when present; "fill" = disruption
    in that game of the player who filled in for an out regular (realized absences, regular season);
    "bench" = everyone else's per-game mean."""
    x = dg[dg["season"].isin(list(seasons))]
    reg = season_regulars(x)
    out = {}
    for G in DGROUPS:
        r = reg[reg["grp"] == G]
        out[G] = {"start": float(r.loc[r["regular"], "dv"].sum() / r.loc[r["regular"], "gp"].sum()),
                  "bench": float(r.loc[~r["regular"], "dv"].sum() / max(r.loc[~r["regular"], "gp"].sum(), 1)),
                  "fill": 0.0}
    tmp = {G: {"start": out[G]["start"], "fill": out[G]["bench"]} for G in DGROUPS}
    _, det = team_costs(x, sched, tmp, {G: 10.0 for G in DGROUPS}, None, True, seasons, regular)
    det = det[(det["p_out"] >= 0.5) & (det["repl_id"] >= 0)]
    dvm = x.set_index(["game_id", "team_id", "athlete_id"])["dv"]
    fill = dvm.reindex(pd.MultiIndex.from_frame(det[["game_id", "team_id", "repl_id"]])).to_numpy()
    det = det.assign(fill_dv=fill)
    for G in DGROUPS:
        f = det.loc[det["grp"] == G, "fill_dv"].dropna()
        out[G]["fill"] = float(f.mean()) if len(f) else out[G]["bench"]
        out[G]["n_fill"] = int(len(f))
    return out


def chart_defense(rows: pd.DataFrame, teams: pd.DataFrame) -> dict:
    """{team_id: {label: {"role": role, "ids": [athlete ids by depth], "all": ids + removed}}} for the
    defensive labels of this week's chart (`depth_impact.week_chart`: removed players have no depth).
    "ids" is the next-man-up order; "all" also holds the removed (listed-out) players, so an out
    starter is found at his label."""
    tid = dict(zip(teams.loc[teams["season"] == 2026, "team"], teams.loc[teams["season"] == 2026, "team_id"]))
    r = rows[(rows["unit"] == "defense") & rows["athlete_id"].notna()].copy()
    r["athlete_id"] = r["athlete_id"].astype("int64")
    out = {}
    for team, g in r.groupby("team"):
        t = tid.get(team)
        if t is None:
            continue
        out[t] = {}
        for lab, x in g.groupby("position"):
            on = x[x["depth"].notna()].sort_values("depth")
            ids = list(dict.fromkeys(on["athlete_id"]))
            out[t][lab] = {"role": x["role"].iloc[0], "ids": ids,
                           "all": ids + [a for a in dict.fromkeys(x.loc[x["depth"].isna(), "athlete_id"]) if a not in ids]}
    return out

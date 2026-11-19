"""Depth-chart player impact scores (test bench, 2026-11-19; rebuild of PR 26 on the team charts).

PR 26 priced an absence by the player's usage share alone. This module prices it by what the
team loses at the player's depth chart slot: the full-strength starter's value minus the value of
the player who actually fills the slot.

* Slots: offense QB, RB, WR (X, Z, SLOT, or WIDE / WR when the exact spot is unknown) and TE.
  Source order: the 2026 team charts (`cfb.depth_charts`, 68 P4 + Notre Dame teams; any team, so a
  transfer keeps his 2026 spot), the CFBDepth WR export (Slot Rate >= 50% -> SLOT, else WIDE), the
  CFBD roster position of that season, and >= 10 dropbacks in a game -> QB. A chart spot is used for
  an earlier season ("similar positions compared to this year") only when it uses the same kind of
  touch (dropback / carry / target) the player mostly had that season.
* Full-strength chart (pregame, per team-game): at each group the players with the most touches per
  game when present (QB 1, RB 1, WR 3, TE 1), from the team's earlier games this season (>= 2
  appearances). A player listed out keeps his slot: the chart is built from usage while healthy,
  not from the current chart, so a starter the team chart has moved to the bench is put back.
  A player with a starter's role last season for the same team who has not played this season is
  a candidate too (season-long injury): he takes the slot when his touches per game last season
  were at least 80% of the in-season leader's. Transfers are left out: a transfer who never plays
  was mostly beaten out (2025: Mikey Keene at Michigan, Kadin Semonza at Tulane).
* Value at a slot: shrunk EPA per touch x touches per game (QB rates add last season's dropbacks at
  half weight, which also prices a season-long absentee). Replacement = the healthy non-starter at
  the same slot with the most touches per game (WR: same spot first), else the league backup rate.
  Cost of a starter out = max(starter rate - replacement rate, 0) x starter touches per game, in
  points, times P(out). QB rates use PR 34's shrinkage (120 dropbacks toward league starter /
  backup means); RB / WR / TE use K from year-to-year reliability (250 carries, 100 / 200 targets).
* Baked-in discount (variant): model_v2's ratings already contain the games played without him.
  str_power's prior counts 4 games, so after `m` missed games out of `k` the remaining cost is
  1 - m / (4 + k).

`team_costs` returns one row per (game_id, team_id) with cost columns; `scripts/depth_impact_select.py`
turns home minus away into a model_v2 adjustment. `adjusted_chart` writes the current full-strength
chart for the 68 charted teams with listed-out starters put back in their slot (all units).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.lab import availability as A
from cfb.lab import player_maps as PM

GROUPS = ("QB", "RB", "WR", "TE")
TOUCH = {"QB": "drop", "RB": "carry", "WR": "tgt", "TE": "tgt"}
N_START = {"QB": 1, "RB": 1, "WR": 3, "TE": 1}
K = {"QB": 120.0, "RB": 250.0, "WR": 100.0, "TE": 200.0}
PREV_W = {"QB": 0.5, "RB": 0.0, "WR": 0.0, "TE": 0.0}   # weight of last season's touches in a rate
MIN_GAMES = 2
PRIOR_GAMES = 4.0       # str_power prior strength in games (features/strength/points.py)
PREV_SHARE = {"QB": 0.50, "RB": 0.30, "WR": 0.12, "TE": 0.08}   # last-season share for a season-long candidate
PREV_MIN_GP = 4
PREV_TAKE = 0.8         # season-long candidate starts if his touches/game >= 80% of the in-season leader's
WR_SPOT = {"X": {"X", "WIDE", "WR"}, "Z": {"Z", "WIDE", "WR"}, "SLOT": {"SLOT", "WR"},
           "WIDE": {"X", "Z", "WIDE", "WR"}, "WR": {"X", "Z", "SLOT", "WIDE", "WR"}}
ALIGN_GRP = {"QB": "QB", "RB": "RB", "X": "WR", "Z": "WR", "SLOT": "WR", "WIDE": "WR", "WR": "WR", "TE": "TE"}


# ---------------------------------------------------------------------------- charts and slots

def load_chart(data: Path) -> pd.DataFrame:
    """All rows of the latest download per team, with a role per row: offense QB/RB/X/Z/SLOT/TE/OL,
    defense IDL/EDGE/LB/CB/S/NB (player_maps.chart_role), else the unit name (bench, special_teams)."""
    files = glob.glob(str(data / "tables" / "depth_charts" / "*.parquet"))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d[d["asof_utc"] == d.groupby("team")["asof_utc"].transform("max")].copy()
    d["athlete_id"] = pd.to_numeric(d["athlete_id"], errors="coerce")
    edge_teams = set(d.loc[(d["unit"] == "defense") & (d["pos_group"] == "EDGE") & (d["depth"] == 1), "team"])
    role = []
    for p, g, t, u in zip(d["position"], d["pos_group"], d["team"], d["unit"]):
        if u == "defense":
            role.append(PM.chart_role(p, g, t in edge_teams)[0])
        elif u == "offense":
            role.append("OL" if g == "OL" else PM.CHART_ALIGN.get(str(p).upper(), g))
        else:
            role.append(u)
    d["role"] = role
    return d


def chart_offense(chart: pd.DataFrame) -> pd.DataFrame:
    """Best (lowest depth) offensive skill spot per charted athlete: athlete_id, chart_team, align."""
    c = chart[(chart["unit"] == "offense") & chart["role"].isin(["QB", "RB", "X", "Z", "SLOT", "TE"])
              & chart["athlete_id"].notna()]
    c = c.sort_values("depth").drop_duplicates("athlete_id")
    return pd.DataFrame({"athlete_id": c["athlete_id"].astype("int64").to_numpy(),
                         "chart_team": c["team"].to_numpy(), "align": c["role"].to_numpy(),
                         "chart_depth": c["depth"].to_numpy()})


def wr_export_spots(export_dir: Path, data: Path, teams: pd.DataFrame) -> pd.DataFrame:
    """CFBDepth WR export (2026 season): SLOT when Slot Rate >= 50%, else WIDE."""
    f = export_dir / "cfbdepth-wr-advanced.csv"
    if not f.exists():
        return pd.DataFrame(columns=["athlete_id", "align"])
    info = pd.read_csv(export_dir / "cfbdepth-general-information.csv")
    abbr = dict(zip(info["Abbrev"], info["School Name"]))
    w = pd.read_csv(f)
    r, tid = A._roster_keys(data, 2026, teams)
    w["team_id"] = w["School"].map(abbr).map(tid)
    w = w[w["team_id"].notna()]
    w["athlete_id"] = A._match(w.rename(columns={"Name": "name"}), r)
    w = w[w["athlete_id"].notna()]
    rate = pd.to_numeric(w["Slot Rate"].astype(str).str.rstrip("%"), errors="coerce") / 100.0
    return pd.DataFrame({"athlete_id": w["athlete_id"].astype("int64").to_numpy(),
                         "align": np.where(rate.to_numpy() >= 0.5, "SLOT", "WIDE")}).drop_duplicates("athlete_id")


def roster_groups(data: Path, seasons) -> pd.DataFrame:
    """(season, team_id, athlete_id, rgrp) from the CFBD roster (QB/RB/WR/TE only)."""
    teams = pd.read_parquet(data / "core" / "teams.parquet")
    out = []
    for s in seasons:
        f = data / "tables" / "roster" / f"{s}.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f, columns=["id", "team", "position"])
        tid = dict(zip(teams.loc[teams["season"] == s, "team"], teams.loc[teams["season"] == s, "team_id"]))
        r["athlete_id"] = pd.to_numeric(r["id"], errors="coerce")
        r["team_id"] = r["team"].map(tid)
        r["rgrp"] = r["position"].map({"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"})
        r = r.dropna(subset=["athlete_id", "team_id", "rgrp"])
        r["season"] = s
        out.append(r[["season", "team_id", "athlete_id", "rgrp"]])
    o = pd.concat(out, ignore_index=True)
    o["athlete_id"] = o["athlete_id"].astype("int64")
    o["team_id"] = o["team_id"].astype("int64")
    return o.drop_duplicates(["season", "team_id", "athlete_id"])


def slot_labels(u: pd.DataFrame, rost: pd.DataFrame, chart_off: pd.DataFrame, wr_spots: pd.DataFrame,
                use_chart: bool = True) -> pd.DataFrame:
    """(season, athlete_id) -> align (QB/RB/X/Z/SLOT/WIDE/WR/TE) and grp, by the source order above.
    use_chart=False gives the roster-position version (PR 26 / PR 34 positions) for comparison."""
    tot = u.groupby(["season", "athlete_id"])[["drop", "carry", "tgt"]].sum()
    qb_flag = u[u["drop"] >= 10].groupby(["season", "athlete_id"]).size().rename("qbg")
    base = pd.concat([tot.reset_index()[["season", "athlete_id"]],
                      rost[["season", "athlete_id"]]]).drop_duplicates()
    b = base.merge(tot.reset_index(), on=["season", "athlete_id"], how="left").fillna({"drop": 0, "carry": 0, "tgt": 0})
    b = b.merge(rost.drop_duplicates(["season", "athlete_id"])[["season", "athlete_id", "rgrp"]],
                on=["season", "athlete_id"], how="left")
    b = b.merge(qb_flag.reset_index(), on=["season", "athlete_id"], how="left")
    ug = u.drop_duplicates(["season", "athlete_id"])[["season", "athlete_id", "grp"]]
    b = b.merge(ug, on=["season", "athlete_id"], how="left")
    # usage type of the season
    vol = b[["drop", "carry", "tgt"]].to_numpy()
    dom = np.array(["drop", "carry", "tgt"])[vol.argmax(axis=1)]
    has = vol.max(axis=1) >= 10
    base_grp = b["rgrp"].where(b["rgrp"].notna(), b["grp"].where(b["grp"].isin(GROUPS)))
    align = base_grp.copy()
    if use_chart:
        b = b.merge(chart_off[["athlete_id", "align"]].rename(columns={"align": "c_align"}), on="athlete_id", how="left")
        b = b.merge(wr_spots.rename(columns={"align": "w_align"}), on="athlete_id", how="left")
        c_grp = b["c_align"].map(ALIGN_GRP)
        c_touch = c_grp.map(TOUCH)
        ok = b["c_align"].notna() & (~has | (c_touch.to_numpy() == dom))
        align = align.where(~ok, b["c_align"])
        wok = (~ok) & b["w_align"].notna() & (align.map(ALIGN_GRP) == "WR")
        align = align.where(~wok, b["w_align"])
    align = align.where(b["qbg"].isna(), "QB")
    b["align"] = align
    b["grp"] = b["align"].map(ALIGN_GRP)
    b = b[b["grp"].isin(GROUPS)]
    return b[["season", "athlete_id", "align", "grp"]].drop_duplicates(["season", "athlete_id"])


# ---------------------------------------------------------------------------- league means

def league_means(u: pd.DataFrame, lab: pd.DataFrame, before: int, first: int = 2018) -> dict:
    """EPA per touch of season starters and of other players, per group, seasons [first, before)."""
    x = u[(u["season"] >= first) & (u["season"] < before)].drop(columns=["grp"], errors="ignore").merge(
        lab, on=["season", "athlete_id"], how="inner")
    out = {}
    for g in GROUPS:
        t = TOUCH[g]
        s = x[x["grp"] == g].groupby(["season", "team_id", "athlete_id"])[[t, f"{t}_epa"]].sum().reset_index()
        s = s.sort_values(t, ascending=False)
        top = s.groupby(["season", "team_id"]).head(N_START[g])
        rest = s.drop(top.index)
        out[g] = (float(top[f"{t}_epa"].sum() / max(top[t].sum(), 1)), float(rest[f"{t}_epa"].sum() / max(rest[t].sum(), 1)))
    return out


# ---------------------------------------------------------------------------- per-game costs

def prev_season_stats(u: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Per (season S, athlete): last season's (S-1) team, games, touches, EPA and team totals, with his
    season-S group (team_id is the S-1 team)."""
    s = u.groupby(["season", "team_id", "athlete_id"])[["drop", "carry", "tgt", "drop_epa", "carry_epa", "tgt_epa"]].sum()
    gp = u.groupby(["season", "team_id", "athlete_id"]).size().rename("gp")
    s = s.join(gp).reset_index()
    team = u.groupby(["season", "team_id"])[["drop", "carry", "tgt"]].sum().add_prefix("team_").reset_index()
    s = s.merge(team, on=["season", "team_id"])
    # a player on two teams in a season keeps the team where he had more appearances
    s = s.sort_values("gp", ascending=False).drop_duplicates(["season", "athlete_id"])
    s["season"] = s["season"] + 1
    return s.merge(lab, on=["season", "athlete_id"], how="inner")


def _chart_label(crep: dict, athlete: int, align: str, grp: str):
    """The chart label for a starter: where the chart lists him (best depth), else the team's first
    label for his spot (X, Z, SLOT; WIDE / WR -> X then Z; QB, RB, TE)."""
    best = None
    for lab, x in crep.items():
        if athlete in x["ids"]:
            d = x["ids"].index(athlete)
            if best is None or d < best[0]:
                best = (d, lab)
    if best is not None and ALIGN_GRP.get(crep[best[1]]["role"], crep[best[1]]["role"]) == grp:
        return best[1]
    want = {"WIDE": ["X", "Z"], "WR": ["X", "Z", "SLOT"]}.get(align, [align])
    for w in want + [grp]:
        for lab, x in crep.items():
            if x["role"] == w or (w == grp and ALIGN_GRP.get(x["role"]) == grp):
                return lab
    return None


def team_costs(u: pd.DataFrame, sched: pd.DataFrame, lab: pd.DataFrame, rost: pd.DataFrame, mu: dict,
               p_out: dict | None, realized: bool, seasons, regular: set | None = None,
               season_long: bool = True, record_all: bool = False, starter_by: str = "per_game",
               prev_w: dict | None = None, repl_min_gp: int = 1, groups=GROUPS,
               chart_ref: dict | None = None, chart_repl: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (season, game_id, team_id) for games with >= 2 earlier team games.

    p_out: {(game_id, team_id, athlete_id): probability} from listings (used when realized=False).
    realized: Josiah's 2025 assumption, a player with a role who did not play in a regular-season
    game was out injured (p_out 1). regular: game_ids that count as regular season (realized mode).
    mu: season -> {group: (starter mean, backup mean)}.
    starter_by: "per_game" (touches per game when present; a starter who missed games keeps his
    slot) or "total" (most touches so far this season, PR 34's QB rule). prev_w: weight of last
    season's touches in the rates. repl_min_gp: appearances a replacement needs.
    chart_ref / chart_repl (from `chart_slots`, charted teams only): with chart_ref the starters are
    that chart's first team at each slot (the adjusted, full-strength chart) instead of the usage
    rule, and the replacement is the next player at the slot's label on chart_repl (default
    chart_ref) who is not out (P(out) < 0.5) and not a healthy starter elsewhere; his rate is his
    own if he has played this season, else the league backup rate."""
    prev_w = PREV_W if prev_w is None else prev_w
    prev = prev_season_stats(u, lab)
    prev_by = {s: g for s, g in prev.groupby("season")}
    rows, detail = [], []
    lab_s = {s: g.set_index("athlete_id")["grp"] for s, g in lab.groupby("season")}
    align_s = {s: g.set_index("athlete_id")["align"] for s, g in lab.groupby("season")}
    for (season, team), g in sched[sched["season"].isin(list(seasons))].groupby(["season", "team_id"], sort=False):
        games = g.sort_values("start_time_utc")["game_id"].to_numpy()
        n = len(games)
        if n < 3:
            continue
        gidx = {gid: i for i, gid in enumerate(games)}
        pu = u[(u["team_id"] == team) & u["game_id"].isin(gidx)]
        grp_map = lab_s.get(season)
        al_map = align_s.get(season)
        if grp_map is None:
            continue
        # season-long candidates: on this team's roster, last-season starter-level share, no appearance here
        pv = prev_by.get(season, prev.iloc[:0])
        ros = set(rost.loc[(rost["season"] == season) & (rost["team_id"] == team), "athlete_id"])
        cand = set()
        if season_long:
            for x in pv[pv["athlete_id"].isin(ros) & (pv["team_id"] == team)].itertuples(index=False):
                t = TOUCH[x.grp]
                sh = getattr(x, t) / max(getattr(x, f"team_{t}"), 1)
                if x.gp >= PREV_MIN_GP and sh >= PREV_SHARE[x.grp]:
                    cand.add(x.athlete_id)
        cref = chart_ref.get(team) if chart_ref else None
        crep = chart_repl.get(team) if chart_repl else None
        if cref and crep is None:
            crep = {lab_: {"role": G_, "ids": ids} for G_, slots in cref.items() for lab_, ids in slots}
        extra = set()
        for slots in (cref or {}).values():
            for _lab, ids in slots:
                extra.update(ids)
        for x_ in (crep or {}).values():
            extra.update(x_["ids"])
        extra = {a for a in extra if a in grp_map.index}
        ath = np.array(sorted(set(pu["athlete_id"]) | cand | extra))
        ath = ath[[a in grp_map.index for a in ath]]
        if not len(ath):
            continue
        aidx = {a: i for i, a in enumerate(ath)}
        m = len(ath)
        P = np.zeros((m, n))
        V = {c: np.zeros((m, n)) for c in ("drop", "carry", "tgt", "drop_epa", "carry_epa", "tgt_epa")}
        pu = pu[pu["athlete_id"].isin(list(aidx))]
        ri = pu["athlete_id"].map(aidx).to_numpy()
        ci = pu["game_id"].map(gidx).to_numpy()
        P[ri, ci] = 1.0
        for c in V:
            V[c][ri, ci] = pu[c].to_numpy()
        cP = np.cumsum(P, axis=1) - P
        cV = {c: np.cumsum(V[c], axis=1) - V[c] for c in V}
        grp = grp_map.reindex(ath).to_numpy()
        align = al_map.reindex(ath).to_numpy()
        pvi = pv.set_index("athlete_id").reindex(ath)
        has_prev = np.array([a in cand for a in ath])
        first = np.where(P.any(axis=1), P.argmax(axis=1), n)
        mu_s = mu[season]
        for k in range(2, n):
            gid = games[k]
            gp = cP[:, k]
            if realized:
                po = (1.0 - P[:, k]) if (regular is None or gid in regular) else np.zeros(m)
            else:
                po = np.array([p_out.get((gid, team, a), 0.0) for a in ath]) if p_out else np.zeros(m)
            # missed games already in the ratings: since first appearance, or from week 1 for a
            # player in last season's returning production
            start = np.where(has_prev, 0, np.minimum(first, k))
            missed = np.maximum((k - start) - gp, 0)
            disc = np.clip(1.0 - missed / (PRIOR_GAMES + k), 0.0, 1.0)
            row = {"season": season, "game_id": gid, "team_id": team}
            for G in groups:
                t = TOUCH[G]
                in_g = grp == G
                cur_ok = in_g & (gp >= MIN_GAMES)
                v_cur = np.where(cur_ok, cV[t][:, k] / np.maximum(gp, 1), 0.0)
                prev_n = pvi[t].to_numpy(dtype=float) if t in pvi else np.zeros(m)
                prev_n = np.nan_to_num(prev_n)
                prev_e = np.nan_to_num(pvi[f"{t}_epa"].to_numpy(dtype=float))
                prev_gp = np.nan_to_num(pvi["gp"].to_numpy(dtype=float))
                # a season-long candidate counts only when he is out for this game (listed, or did not play)
                v_prev = np.where(in_g & has_prev & (gp < MIN_GAMES) & (po >= 0.5), prev_n / np.maximum(prev_gp, 1), 0.0)
                lead = v_cur.max() if cur_ok.any() else 0.0
                v_prev = np.where(v_prev >= PREV_TAKE * lead, v_prev, 0.0)
                v = np.maximum(v_cur, v_prev)
                pool = np.where(v > 0)[0]
                if not len(pool):
                    row.update({f"{G.lower()}_cost": 0.0, f"{G.lower()}_cost_d": 0.0, f"{G.lower()}_n_out": 0.0,
                                f"{G.lower()}_vor": 0.0})
                    continue
                rank = v if starter_by == "per_game" else np.where(v_cur > 0, cV[t][:, k], v_prev * PREV_MIN_GP)
                order = pool[np.argsort(-rank[pool], kind="stable")]
                starters = list(order[:N_START[G]])
                slot_lab = {}
                if cref:
                    # chart mode: first team at each slot label; v from usage (0 if no role yet)
                    starters = []
                    for lab_, ids in cref.get(G, []):
                        if ids and ids[0] in aidx and aidx[ids[0]] not in starters:
                            starters.append(aidx[ids[0]])
                            slot_lab[aidx[ids[0]]] = lab_
                    starters = [si for si in starters if v[si] > 0]
                n_tch = cV[t][:, k] + prev_w[G] * prev_n
                e_tch = cV[f"{t}_epa"][:, k] + prev_w[G] * prev_e
                ms, mb = mu_s[G]
                rate_s = (e_tch + K[G] * ms) / (n_tch + K[G])
                rate_b = (e_tch + K[G] * mb) / (n_tch + K[G])
                # replacements: healthy non-starters with any touches this season, most touches/game first
                bench_v = np.where(in_g & (gp >= repl_min_gp), cV[t][:, k] / np.maximum(gp, 1), 0.0)
                if starter_by == "total":
                    bench_v = np.where(bench_v > 0, cV[t][:, k], 0.0)
                cands = [i for i in np.argsort(-bench_v, kind="stable") if bench_v[i] > 0 and i not in starters
                         and po[i] < 0.5]
                used = set()
                cost = cost_d = n_out = vor_sum = 0.0
                healthy_st = {si for si in starters if po[si] < 0.5}
                for si in starters:
                    pick = None
                    if crep:
                        # next man up on the chart at the starter's label
                        lab_ = slot_lab.get(si) or _chart_label(crep, int(ath[si]), str(align[si]), G)
                        for a_ in (crep.get(lab_) or {}).get("ids", []):
                            ci_ = aidx.get(a_)
                            if ci_ is None or ci_ == si or ci_ in used or ci_ in healthy_st or po[ci_] >= 0.5:
                                continue
                            pick = ci_
                            break
                    else:
                        if G == "WR":
                            ok = WR_SPOT.get(str(align[si]), WR_SPOT["WR"])
                            pick = next((c for c in cands if c not in used and str(align[c]) in ok), None)
                        if pick is None:
                            pick = next((c for c in cands if c not in used), None)
                    if po[si] >= 0.5 and pick is not None:
                        used.add(pick)
                    rr = rate_b[pick] if pick is not None else mb
                    vor = max(rate_s[si] - rr, 0.0) * v[si]
                    vor_sum += vor
                    cost += vor * po[si]
                    cost_d += vor * po[si] * disc[si]
                    n_out += po[si]
                    if po[si] > 0 or (record_all and k == n - 1):
                        detail.append((season, gid, team, G, int(ath[si]), str(align[si]), float(po[si]),
                                       float(v[si]), float(rate_s[si]), int(ath[pick]) if pick is not None else -1,
                                       float(rr), float(vor), float(disc[si]), bool(v_prev[si] > v_cur[si])))
                row.update({f"{G.lower()}_cost": cost, f"{G.lower()}_cost_d": cost_d, f"{G.lower()}_n_out": n_out,
                            f"{G.lower()}_vor": vor_sum})
            rows.append(row)
    det = pd.DataFrame(detail, columns=["season", "game_id", "team_id", "grp", "athlete_id", "align", "p_out",
                                        "touches_pg", "rate", "repl_id", "repl_rate", "vor", "disc", "season_long"])
    return pd.DataFrame(rows), det


def chart_slots(rows: pd.DataFrame, teams: pd.DataFrame) -> tuple[dict, dict]:
    """Offensive skill slots from chart rows (team, unit, position, depth, athlete_id, role).

    Returns (ref, repl): ref = {team_id: {group: [(label, [athlete ids by depth]), ...]}} with QB 1,
    RB 1 (RB before FB), WR up to 3 labels and TE 1 label, the order of `N_START`; repl = {team_id:
    {label: {"role": QB/RB/X/Z/SLOT/TE, "ids": [athlete ids by depth]}}} for every offensive skill label."""
    tid = dict(zip(teams.loc[teams["season"] == 2026, "team"], teams.loc[teams["season"] == 2026, "team_id"]))
    r = rows[(rows["unit"] == "offense") & rows["role"].isin(["QB", "RB", "X", "Z", "SLOT", "TE"])].copy()
    r = r[r["athlete_id"].notna() & r["depth"].notna()]
    r["athlete_id"] = r["athlete_id"].astype("int64")
    r["depth"] = r["depth"].astype(float)
    r["grp"] = r["role"].map(ALIGN_GRP)
    ref, repl = {}, {}
    for team, g in r.groupby("team"):
        t = tid.get(team)
        if t is None:
            continue
        labs, roles = {}, {}
        for lab, x in g.sort_values("depth").groupby("position", sort=False):
            labs[lab] = (x["grp"].iloc[0], list(dict.fromkeys(x["athlete_id"])))
            roles[lab] = x["role"].iloc[0]
        repl[t] = {lab: {"role": roles[lab], "ids": ids} for lab, (_g, ids) in labs.items()}
        out = {}
        for G in GROUPS:
            cand = [(lab, ids) for lab, (gg, ids) in labs.items() if gg == G]
            if G == "RB":
                cand = sorted(cand, key=lambda li: li[0] != "RB")
            if G == "TE":
                cand = sorted(cand, key=lambda li: li[0] not in ("TE", "TE-Y"))
            if G == "WR":
                order = {"WR-X": 0, "WR-Z": 1, "WR-SL": 2}
                cand = sorted(cand, key=lambda li: order.get(li[0], 3))
            out[G] = cand[:N_START[G]]
        ref[t] = out
    return ref, repl


def week_chart(chart: pd.DataFrame, out_ids: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """This week's chart: the current team charts (offense and defense) less every player listed out
    (out_ids: team_id, athlete_id, status), with depth renumbered at each label."""
    tid = dict(zip(teams.loc[teams["season"] == 2026, "team"], teams.loc[teams["season"] == 2026, "team_id"]))
    c = chart[chart["unit"].isin(["offense", "defense"])].copy()
    c["team_id"] = c["team"].map(tid)
    st = out_ids.drop_duplicates(["team_id", "athlete_id"]).set_index(["team_id", "athlete_id"])["status"].to_dict()
    c["removed_status"] = [st.get((t, a)) for t, a in zip(c["team_id"], c["athlete_id"])]
    kept = c[c["removed_status"].isna()].sort_values(["team", "position", "depth"]).copy()
    kept["depth"] = kept.groupby(["team", "position"]).cumcount() + 1
    removed = c[c["removed_status"].notna()].assign(depth=np.nan)
    return pd.concat([kept, removed], ignore_index=True).sort_values(["team", "unit", "position", "depth"])



# ---------------------------------------------------------------------------- adjusted chart (all units)

UNIT_ROLE = {"OL": "OL", "IDL": "DL", "EDGE": "DL", "LB": "LB", "CB": "DB", "S": "DB", "NB": "DB"}
EXPORT_UNIT = {"OL": "ol", "DL": "dl", "LB": "lb", "DB": "db"}


def export_roles(export_dir: Path, data: Path, teams: pd.DataFrame) -> pd.DataFrame:
    """Natural spot from the CFBDepth season exports (most snaps): OL LT/LG/C/RG/RT; DL A/B gap ->
    IDL, outside -> EDGE; LB off-ball -> LB, on the line -> EDGE; DB CB / Slot / FS -> CB / NB / S.
    Plus snaps per game and the export's starter share (snaps per game / team top-N mean)."""
    if not (export_dir / "cfbdepth-general-information.csv").exists():
        return pd.DataFrame()
    vals = A.cfbdepth_unit_values(export_dir, data, teams)
    info = pd.read_csv(export_dir / "cfbdepth-general-information.csv")
    abbr = dict(zip(info["Abbrev"], info["School Name"]))
    r, tid = A._roster_keys(data, 2026, teams)
    spots = []
    rules = {
        "ol": {"LT": "Snaps LT", "LG": "Snaps LG", "C": "Snaps C", "RG": "Snaps RG", "RT": "Snaps RT"},
        "dl": {"IDL": ("Snaps A Gap", "Snaps B Gap"), "EDGE": ("Snaps OutT", "Snaps OverT"), "LB": ("Snaps OffBall",)},
        "lb": {"LB": ("Snaps OffBall", "Snaps Box", "Snaps Drop"), "EDGE": ("Snaps DL",), "S": ("Snaps FS",)},
        "db": {"CB": ("Snaps CB",), "NB": ("Snaps Slot",), "S": ("Snaps FS",)},
    }
    for unit, rule in rules.items():
        d = pd.read_csv(export_dir / f"cfbdepth-{unit}-advanced.csv")
        sc = {}
        for spot, cols in rule.items():
            cols = (cols,) if isinstance(cols, str) else cols
            sc[spot] = sum(pd.to_numeric(d[c], errors="coerce").fillna(0.0) for c in cols if c in d)
        s = pd.DataFrame(sc)
        d["spot"] = s.idxmax(axis=1).where(s.max(axis=1) > 0)
        d["team_id"] = d["School"].map(abbr).map(tid)
        d = d[d["team_id"].notna() & d["spot"].notna()]
        d["athlete_id"] = A._match(d.rename(columns={"Name": "name"}), r)
        d = d[d["athlete_id"].notna()]
        spots.append(d[["team_id", "athlete_id", "spot"]])
    sp = pd.concat(spots).astype({"team_id": "int64", "athlete_id": "int64"}).drop_duplicates(["team_id", "athlete_id"])
    return vals.merge(sp, on=["team_id", "athlete_id"], how="left")


def adjusted_chart(chart: pd.DataFrame, listed: pd.DataFrame, exports: pd.DataFrame, off_scores: pd.DataFrame,
                   teams: pd.DataFrame) -> pd.DataFrame:
    """Current full-strength chart for the charted teams.

    listed: (team_id, athlete_id, status, p_out) for the coming game. exports: export_roles().
    off_scores: offensive full-strength starters from team_costs details (team_id, athlete_id, grp, align).
    A listed player with a real role (export starter share >= 0.5, or an offensive full-strength
    starter) who sits on the bench or below his natural depth is moved to his natural spot:
    his chart spot if he is still on offense / defense, else the export spot (OL exact; CB / S / NB /
    IDL / EDGE / LB by role; WR by SLOT / WIDE). Where the team has two labels for the role (LCB/RCB,
    LDE/RDE, X/Z) he takes the one whose first-team player has the lower starter share. Depth: first
    team for an offensive full-strength starter, else behind every player at that label with a larger
    export starter share. Players below him move down one, keeping their order. Players are placed in
    ascending share order, so the stronger of two out players at one label ends up higher."""
    tid = dict(zip(teams.loc[teams["season"] == 2026, "team"], teams.loc[teams["season"] == 2026, "team_id"]))
    c = chart[chart["unit"].isin(["offense", "defense"])].copy()
    c["team_id"] = c["team"].map(tid)
    c["inserted"] = False
    c["note"] = ""
    ex = exports.set_index(["team_id", "athlete_id"]) if len(exports) else None
    share = {}
    if ex is not None:
        share = ex["starter"].to_dict()
    c["starter_share"] = [share.get((t, a)) for t, a in zip(c["team_id"], c["athlete_id"])]
    out = []
    lst = listed.copy()
    for team, g in c.groupby("team", sort=True):
        t_id = tid.get(team)
        g = g.copy()
        L = lst[lst["team_id"] == t_id]
        offs = off_scores[off_scores["team_id"] == t_id] if len(off_scores) else off_scores
        L = L.assign(_s=[share.get((t_id, a_), 0.0) or 0.0 for a_ in L["athlete_id"]]).sort_values("_s")
        for x in L.itertuples(index=False):
            a = x.athlete_id
            sh = share.get((t_id, a))
            off_start = a in set(offs["athlete_id"]) if len(offs) else False
            if not ((sh is not None and sh >= 0.5) or off_start):
                continue
            on = g[(g["athlete_id"] == a)]
            first_team = on["depth"].min() if len(on) else np.nan
            if first_team == 1:
                continue
            # natural role
            role = None
            if len(on):
                role = on.sort_values("depth").iloc[0]["role"]
            elif off_start:
                al = offs.loc[offs["athlete_id"] == a, "align"].iloc[0]
                role = {"WIDE": "XZ", "WR": "XZ"}.get(al, al)
            elif ex is not None and (t_id, a) in ex.index:
                sp = ex.loc[(t_id, a), "spot"]
                role = sp if sp in ("IDL", "EDGE", "LB", "CB", "S", "NB") else ("OL:" + sp if isinstance(sp, str) else None)
            if role is None:
                continue
            if len(on):
                labels = np.array([on.sort_values("depth").iloc[0]["position"]])
            elif role.startswith("OL:"):
                labels = g[(g["role"] == "OL") & (g["position"] == role[3:])]["position"].unique()
                if not len(labels):
                    labels = g[g["role"] == "OL"]["position"].unique()
            elif role == "XZ":
                labels = g[g["role"].isin(["X", "Z"])]["position"].unique()
            else:
                labels = g[g["role"] == role]["position"].unique()
            if not len(labels):
                continue
            # the label whose first-team player has the lowest starter share
            ft = g[g["position"].isin(labels) & (g["depth"] == 1)]
            ft = ft.assign(s=ft["starter_share"].fillna(0.0).astype(float))
            lab = ft.sort_values("s")["position"].iloc[0] if len(ft) else labels[0]
            # depth: first team for an offensive full-strength starter; otherwise behind the players at
            # the label with a larger starter share (a rotation player does not jump a full-time starter)
            here = (g["position"] == lab) & g["depth"].notna()
            at = g[here & (g["athlete_id"] != a)].sort_values("depth")
            if off_start or sh is None:
                dep = 1
            else:
                bigger = (at["starter_share"].fillna(0.0).astype(float) > sh).to_numpy()
                dep = 1 + int(np.argmin(bigger)) if (~bigger).any() else len(at) + 1
            mine = g[here & (g["athlete_id"] == a)]
            if len(mine) and mine["depth"].min() <= dep:
                continue
            row = (mine.iloc[0].copy() if len(mine) else on.iloc[0].copy() if len(on)
                   else g[g["position"] == lab].iloc[0].copy())
            row["position"], row["athlete_id"] = lab, a
            row["player"] = row["player"] if len(on) else str(getattr(x, "name", ""))
            row["role"] = g.loc[g["position"] == lab, "role"].iloc[0] if (g["position"] == lab).any() else role
            row["inserted"] = True
            row["note"] = f"{x.status}; was {'bench' if not len(on) else 'depth ' + str(int(first_team))}"
            row["starter_share"] = sh
            new_at = pd.concat([at.iloc[:dep - 1], row.to_frame().T, at.iloc[dep - 1:]], ignore_index=True)
            new_at["depth"] = np.arange(1, len(new_at) + 1)
            g = pd.concat([g[~here], new_at], ignore_index=True)
        out.append(g)
    o = pd.concat(out, ignore_index=True)
    st = lst.set_index(["team_id", "athlete_id"])["status"].to_dict()
    o["status"] = [st.get((t, a)) for t, a in zip(o["team_id"], o["athlete_id"])]
    return o.sort_values(["team", "unit", "position", "depth"])

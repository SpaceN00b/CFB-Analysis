"""Depth-chart impact scores vs model_v2 and PR 34's injury model (2026-11-19, test bench).

Josiah's ask: rebuild PR 26 with the team depth charts and positions; put players with a real role
who are listed out back in the slot that suits them; test on 2025, treating a player who did not
play in a 2025 regular-season game as out injured, with players in positions like their 2026 ones.

Inputs (build first, same as PR 34):
    python -m cfb.core.build
    python scripts/impact_base.py               # model_v2 season-forward margins
    python scripts/injury_select.py --build     # data/exploration/avail_player_games.parquet, avail_roles.parquet

Versions (all fixed points, nothing fitted; game term = home minus away, through model_v2's slope):
    pr34            PR 34 fixed_listed: QB value over the next QB + PR 26 per-share points (RB/WR/TE)
    dc              depth-chart value over the slot's replacement, QB/RB/WR/TE, chart positions, an out
                    starter keeps his slot (starter = most touches per game when present), season-long
                    absentees in their slot, baked-in discount 1 - missed / (4 + games)
    dc_full         dc without the discount
    dc_total        starter = most touches so far this season (PR 34's QB rule), no discount
    dc_qb, dc_skill dc's QB part, dc's RB/WR/TE part
    pr34qb+dc_skill PR 34's QB term + dc's RB/WR/TE term
    dc_nolong       dc without season-long absentees
    dc_roster       dc with CFBD roster positions instead of chart positions
    dc_share        dc's starters priced with PR 26 per-share points for RB/WR/TE (QB as dc)
Absence information per test:
    2025 realized   did not play in the regular-season game -> out (Josiah's assumption)
    2026 listed     CFBDepth list + per-player checks + conference reports (PR 34 listings)
    2026 realized   same assumption as 2025, for a like-for-like season

    python scripts/depth_impact_select.py [--out DIR] [--week 12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import injury_select as IS  # noqa: E402

from cfb.eval.bootstrap import paired_bootstrap  # noqa: E402
from cfb.lab import availability as A  # noqa: E402
from cfb.lab import depth_impact as D  # noqa: E402
from cfb.model.models import log_loss, sigmoid  # noqa: E402

DATA = Path("data")
EXP = DATA / "exploration"
EXPORTS = Path("/mnt/project-files/injury-reports/cfbdepth_exports_2026-11-18")
FIXED = IS.FIXED
SHARE_PTS = {"RB": 1.94, "WR": 3.24, "TE": 3.77}      # PR 26 points per 100% share (cost, positive)


def boot(d, p, p0):
    b = paired_bootstrap(d["y"].to_numpy(), np.asarray(p), np.asarray(p0),
                         clusters=(d["season"].astype(str) + "-" + d["week"].astype(str)).to_numpy())
    return b["diff"], b["se"]


def game_adj(base: pd.DataFrame, tf: pd.DataFrame, cost_col: str) -> pd.Series:
    """Home minus away injury points (a cost lowers the team's margin)."""
    c = tf.set_index(["game_id", "team_id"])[cost_col]
    h = pd.Series([c.get((g, t), 0.0) for g, t in zip(base["game_id"], base["home_id"])], index=base.index)
    a = pd.Series([c.get((g, t), 0.0) for g, t in zip(base["game_id"], base["away_id"])], index=base.index)
    return -(h.fillna(0.0) - a.fillna(0.0))


def pr34_costs(ro: pd.DataFrame, mu_qb: dict, p_out: pd.Series | None, realized: bool, regular: set) -> pd.DataFrame:
    """PR 34 team columns with the given absence information -> cost per team-game (positive points)."""
    r = ro.copy()
    if realized:
        r["p_out"] = np.where(r["game_id"].isin(regular), r["absent"], 0.0)
        r["src"] = "realized"
    else:
        r["p_out"] = p_out.reindex(pd.MultiIndex.from_frame(r[["game_id", "team_id", "athlete_id"]])).fillna(0.0).to_numpy()
        r["src"] = np.where(r["p_out"] > 0, "listing", "none")
    tf = A.team_features(r, mu_qb)
    tf["cost"] = tf["qb_out"] - sum(tf[c] * v for c, v in FIXED.items())
    return tf


def share_costs(det: pd.DataFrame, u: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """dc_share: QB from dc; RB/WR/TE starters out priced by PR 26 per-share points x share of team
    touches (touches per game / team touches per game before the game); baked-in discount as dc."""
    tot = u.groupby(["game_id", "team_id"])[["drop", "carry", "tgt"]].sum()
    s = sched.sort_values("start_time_utc").copy()
    s = s.merge(tot.reset_index(), on=["game_id", "team_id"], how="left").fillna({"drop": 0, "carry": 0, "tgt": 0})
    for c in ("carry", "tgt"):
        s[f"tpg_{c}"] = s.groupby(["season", "team_id"])[c].transform(lambda x: x.shift().expanding().mean())
    tpg = s.set_index(["game_id", "team_id"])[["tpg_carry", "tpg_tgt"]]
    d = det.copy()
    d = d.join(tpg, on=["game_id", "team_id"])
    d["share"] = np.where(d["grp"] == "RB", d["touches_pg"] / d["tpg_carry"], d["touches_pg"] / d["tpg_tgt"])
    d["share"] = d["share"].clip(upper=1.0).fillna(0.0)
    d["c"] = np.where(d["grp"] == "QB", d["vor"], d["grp"].map(SHARE_PTS).fillna(0.0) * d["share"]) * d["p_out"] * d["disc"]
    return d.groupby(["game_id", "team_id"], as_index=False)["c"].sum().rename(columns={"c": "cost"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=EXP / "depth_impact")
    ap.add_argument("--week", type=int, default=12, help="coming week for the impact-score table")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    u = pd.read_parquet(EXP / "avail_player_games.parquet")
    ro = pd.read_parquet(EXP / "avail_roles.parquet")
    base = pd.read_parquet(EXP / "impact_base.parquet")
    teams = pd.read_parquet(DATA / "core" / "teams.parquet")
    gi = pd.read_parquet(DATA / "core" / "game_index.parquet")
    regular = set(gi.loc[gi["season_type"] == "regular", "game_id"])
    sched = A.team_schedule(DATA, range(2017, 2027))
    sched = sched.merge(gi[["game_id", "season_type"]], on="game_id")

    chart = D.load_chart(DATA)
    coff = D.chart_offense(chart)
    wr = D.wr_export_spots(EXPORTS, DATA, teams)
    rost = D.roster_groups(DATA, range(2017, 2027))
    lab_c = D.slot_labels(u, rost, coff, wr, use_chart=True)
    lab_r = D.slot_labels(u, rost, coff, wr, use_chart=False)
    print(f"chart rows {len(chart)}, charted offensive skill players {len(coff)}, WR export spots {len(wr)}")
    for s in (2025, 2026):
        x = lab_c[lab_c["season"] == s].merge(lab_r[lab_r["season"] == s], on=["season", "athlete_id"],
                                              suffixes=("_c", "_r"))
        print(f"{s}: labelled QB/RB/WR/TE {len(x)}; chart/export spot used {int((x['align_c'] != x['align_r']).sum())}; "
              f"group changed by chart {int((x['grp_c'] != x['grp_r']).sum())}; WR spots "
              f"{x.loc[x['grp_c'] == 'WR', 'align_c'].value_counts().to_dict()}")

    mu = {s: D.league_means(u, lab_c, s) for s in (2025, 2026)}
    mu_r = {s: D.league_means(u, lab_r, s) for s in (2025, 2026)}
    print("league EPA/touch (starter, other):", {s: {g: tuple(round(v, 3) for v in t) for g, t in m.items()} for s, m in mu.items()})
    mu_qb = {s: A.qb_league_means(ro, s) for s in ro["season"].unique() if s > 2014}

    # ---- 2026 listings (PR 34: CFBDepth list + checks + conference reports to fill)
    sched26 = A.team_schedule(DATA, range(2025, 2027))
    hdi = A.hdi_listings(DATA, sched26, teams, IS.HDI)
    checks = IS.load_checks(IS.CHECKS)
    cfd = A.cfbdepth_listings(DATA, sched26, teams, IS.CFBDEPTH, checks, IS.DOWNLOAD)
    r25 = A.status_rates(ro, hdi[hdi["source"] == "hdi2025"])
    rate26 = dict(r25)
    rate26.setdefault("Out", (0.99, 0))
    rate26["Available"] = (0.0, 0)
    l26 = pd.concat([cfd, hdi[hdi["source"] == "hdi2026"]]).drop_duplicates(["game_id", "team_id", "athlete_id"])
    l26 = l26.assign(p=l26["status"].map({k: v[0] for k, v in rate26.items()})).dropna(subset=["p"])
    p26 = l26.set_index(["game_id", "team_id", "athlete_id"])["p"]
    p26 = p26[~p26.index.duplicated()]
    p26d = p26.to_dict()

    # ---- team costs per version
    runs = {}
    s25, s26 = [2025], [2026]
    for name, lab, m, kw in (("pg", lab_c, mu, {}), ("tot", lab_c, mu, {"starter_by": "total"}),
                             ("pg_nolong", lab_c, mu, {"season_long": False}), ("pg_roster", lab_r, mu_r, {})):
        tr, dr = D.team_costs(u, sched, lab, rost, m, None, True, s25 + s26, regular, **kw)
        tl, dl = D.team_costs(u, sched, lab, rost, m, p26d, False, s26, **kw)
        runs[name] = {"real": (tr, dr), "list": (tl, dl)}
        print(f"{name}: team-games {len(tr)} realized / {len(tl)} listed; starters out realized "
              f"{int((dr['p_out'] >= 0.5).sum())} (season-long {int(dr['season_long'].sum())}), listed {len(dl)}")
    runs["pg"]["real"][1].to_parquet(a.out / "dc_details_realized.parquet", index=False)
    runs["pg"]["list"][1].to_parquet(a.out / "dc_details_listed.parquet", index=False)

    def tot(tf, cols):
        t = tf[["game_id", "team_id"]].copy()
        t["cost"] = tf[cols].sum(axis=1)
        return t

    ALLC = ["qb_cost", "rb_cost", "wr_cost", "te_cost"]
    ALLD = [c + "_d" for c in ALLC]
    pr = {"real": pr34_costs(ro[ro["season"].isin(s25 + s26)], mu_qb, None, True, regular),
          "list": pr34_costs(ro[ro["season"] == 2026], mu_qb, p26, False, regular)}
    costs = {"real": {}, "list": {}}
    for mode in ("real", "list"):
        tf, det = runs["pg"][mode]
        costs[mode]["pr34"] = pr[mode][["game_id", "team_id", "cost"]]
        costs[mode]["dc"] = tot(tf, ALLD)
        costs[mode]["dc_full"] = tot(tf, ALLC)
        costs[mode]["dc_total"] = tot(runs["tot"][mode][0], ALLC)
        costs[mode]["dc_qb"] = tot(tf, ["qb_cost_d"])
        costs[mode]["dc_skill"] = tot(tf, ALLD[1:])
        sk = tot(tf, ALLD[1:]).set_index(["game_id", "team_id"])["cost"]
        q34 = pr[mode].set_index(["game_id", "team_id"])["qb_out"]
        hy = pd.concat([sk, q34], axis=1).fillna(0.0).sum(axis=1).rename("cost").reset_index()
        costs[mode]["pr34qb+dc_skill"] = hy
        costs[mode]["dc_nolong"] = tot(runs["pg_nolong"][mode][0], ALLD)
        costs[mode]["dc_roster"] = tot(runs["pg_roster"][mode][0], ALLD)
        costs[mode]["dc_share"] = share_costs(det, u, sched)

    # ---- evaluation
    conf = teams[teams["season"] == 2026].set_index("team_id")["conference"]
    nd = teams.loc[(teams["season"] == 2026) & (teams["team"] == "Notre Dame"), "team_id"]
    p4 = set(conf[conf.isin(IS.P4)].index) | set(nd)
    g = base[base["season"].isin([2025, 2026])].copy()
    g["p4"] = g["home_id"].isin(p4) | g["away_id"].isin(p4)
    g["p0"] = sigmoid(g["lin"])
    tests = {
        "2025 realized, regular season, all FBS": ("real", (g["season"] == 2025) & (g["season_type"] == "regular")),
        "2025 realized, regular season, P4/ND": ("real", (g["season"] == 2025) & (g["season_type"] == "regular") & g["p4"]),
        "2026 listed, all FBS": ("list", g["season"] == 2026),
        "2026 listed, P4/ND": ("list", (g["season"] == 2026) & g["p4"]),
        "2026 realized, all FBS": ("real", g["season"] == 2026),
        "2026 realized, P4/ND": ("real", (g["season"] == 2026) & g["p4"]),
    }
    order = ["pr34", "dc", "dc_full", "dc_total", "dc_qb", "dc_skill", "pr34qb+dc_skill", "dc_nolong", "dc_roster",
             "dc_share"]
    rows, preds = [], {}
    for lab_t, (mode, msk) in tests.items():
        d = g[msk.to_numpy()].copy()
        pp = {}
        for v in order:
            adj = game_adj(d, costs[mode][v], "cost")
            pp[v] = sigmoid(d["lin"] + d["calib_s"] * adj)
            d[f"adj_{v}"] = adj
        for v in order:
            diff, se = boot(d, pp[v], d["p0"])
            dvp, sevp = boot(d, pp[v], pp["pr34"]) if v != "pr34" else (np.nan, np.nan)
            n_adj = int((d[f"adj_{v}"].abs() > 1e-9).sum())
            rows.append({"test": lab_t, "version": v, "n": len(d), "n_adjusted": n_adj,
                         "ll_v2": round(log_loss(d["y"], d["p0"]), 4), "ll": round(log_loss(d["y"], pp[v]), 4),
                         "vs_v2": round(diff, 4), "se": round(se, 4),
                         "vs_pr34": None if v == "pr34" else round(dvp, 4), "se_vs_pr34": None if v == "pr34" else round(sevp, 4),
                         "acc_v2": round(((d["p0"] >= 0.5) == (d["y"] == 1)).mean(), 4),
                         "acc": round(((pp[v] >= 0.5) == (d["y"] == 1)).mean(), 4),
                         "mean_abs_adj": round(d[f"adj_{v}"].abs().mean(), 3)})
        preds[lab_t] = d
    # pooled: 2025 realized + 2026 listed (each season with its own absence information)
    for lab_t, (k25, k26) in {"pooled 2025 realized + 2026 listed, all FBS": ("2025 realized, regular season, all FBS", "2026 listed, all FBS"),
                              "pooled 2025 realized + 2026 listed, P4/ND": ("2025 realized, regular season, P4/ND", "2026 listed, P4/ND")}.items():
        d = pd.concat([preds[k25], preds[k26]], ignore_index=True)
        pp = {v: sigmoid(d["lin"] + d["calib_s"] * d[f"adj_{v}"]) for v in order}
        for v in order:
            diff, se = boot(d, pp[v], d["p0"])
            dvp, sevp = boot(d, pp[v], pp["pr34"]) if v != "pr34" else (np.nan, np.nan)
            rows.append({"test": lab_t, "version": v, "n": len(d), "n_adjusted": int((d[f"adj_{v}"].abs() > 1e-9).sum()),
                         "ll_v2": round(log_loss(d["y"], d["p0"]), 4), "ll": round(log_loss(d["y"], pp[v]), 4),
                         "vs_v2": round(diff, 4), "se": round(se, 4),
                         "vs_pr34": None if v == "pr34" else round(dvp, 4), "se_vs_pr34": None if v == "pr34" else round(sevp, 4),
                         "acc_v2": round(((d["p0"] >= 0.5) == (d["y"] == 1)).mean(), 4),
                         "acc": round(((pp[v] >= 0.5) == (d["y"] == 1)).mean(), 4),
                         "mean_abs_adj": round(d[f"adj_{v}"].abs().mean(), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(a.out / "results.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + res.to_string(index=False))

    # ---- in-sample slopes (diagnostic: points of residual per point of cost, home minus away)
    for lab_t in ("2025 realized, regular season, all FBS", "2026 listed, all FBS"):
        d = preds[lab_t]
        resid = d["margin"] - d["pred"]
        for v in ("pr34", "dc", "dc_full", "dc_total", "dc_qb", "dc_skill"):
            x = d[f"adj_{v}"].to_numpy()
            if (x != 0).sum() > 10:
                b = float((x * resid).sum() / (x * x).sum())
                print(f"in-sample slope {lab_t} | {v}: {b:+.2f} (1.0 = costs taken at face value)")

    # ---- forward: impact scores and the adjusted chart for the coming week (68 charted teams)
    upto = pd.Timestamp("2026-11-25", tz="UTC")
    schedf = A.team_schedule(DATA, [2026], upcoming_until=upto)
    schedf = schedf.merge(gi[["game_id", "season_type", "week"]], on="game_id")
    schedf = schedf[(schedf["week"] <= a.week)]
    cfd_f = A.cfbdepth_listings(DATA, schedf, teams, IS.CFBDEPTH, checks, IS.DOWNLOAD)
    lf = cfd_f.assign(p=cfd_f["status"].map({k: v[0] for k, v in rate26.items()})).dropna(subset=["p"])
    pf = lf.set_index(["game_id", "team_id", "athlete_id"])["p"]
    pf = pf[~pf.index.duplicated()].to_dict()
    tff, detf = D.team_costs(u, schedf, lab_c, rost, mu, pf, False, [2026], record_all=True)
    _, dett = D.team_costs(u, schedf, lab_c, rost, mu, pf, False, [2026], record_all=True, starter_by="total")
    nxt = schedf.merge(gi[["game_id", "week"]].rename(columns={"week": "wk"}), on="game_id")
    nxt = nxt[nxt["wk"] == a.week][["game_id", "team_id"]]
    sc = detf.merge(nxt, on=["game_id", "team_id"])
    tt = dett.merge(nxt, on=["game_id", "team_id"])
    tt = tt.assign(cost_total=tt["vor"] * tt["p_out"])[["game_id", "team_id", "grp", "athlete_id", "cost_total", "repl_id"]]
    sc = sc.merge(tt.rename(columns={"repl_id": "repl_id_total"}), on=["game_id", "team_id", "grp", "athlete_id"], how="left")
    sc["starter_total"] = sc["cost_total"].notna()
    sc["cost_total"] = sc["cost_total"].fillna(0.0)
    ros26 = pd.read_parquet(DATA / "tables" / "roster" / "2026.parquet", columns=["id", "firstName", "lastName"])
    ros26["athlete_id"] = pd.to_numeric(ros26["id"], errors="coerce")
    nm = (ros26["firstName"].fillna("") + " " + ros26["lastName"].fillna("")).str.strip()
    names = dict(zip(ros26["athlete_id"], nm))
    tname = dict(zip(teams.loc[teams["season"] == 2026, "team_id"], teams.loc[teams["season"] == 2026, "team"]))
    stat = lf.drop_duplicates(["game_id", "team_id", "athlete_id"]).set_index(["game_id", "team_id", "athlete_id"])["status"]
    sc["team"] = sc["team_id"].map(tname)
    sc["player"] = sc["athlete_id"].map(names)
    sc["replacement"] = sc["repl_id"].map(names)
    sc["status"] = [stat.get((gg, t, at)) for gg, t, at in zip(sc["game_id"], sc["team_id"], sc["athlete_id"])]
    sc["cost_dc"] = sc["vor"] * sc["p_out"] * sc["disc"]
    sc["charted_team"] = sc["team_id"].isin(set(chart["team"].map({v: k for k, v in tname.items()})))
    keep = ["team", "grp", "align", "player", "athlete_id", "status", "p_out", "touches_pg", "rate", "replacement",
            "repl_rate", "vor", "disc", "cost_dc", "starter_total", "cost_total", "season_long", "charted_team", "game_id"]
    sc = sc[keep].sort_values(["team", "grp", "touches_pg"], ascending=[True, True, False])
    sc.to_csv(a.out / f"impact_scores_week{a.week}_2026.csv", index=False)
    print(f"\nimpact scores week {a.week}: {len(sc)} starters, {int((sc['p_out'] >= 0.5).sum())} listed out; "
          f"cost >= 1 pt: dc {int((sc['cost_dc'] >= 1).sum())}, dc_total {int((sc['cost_total'] >= 1).sum())}")
    show = ["team", "grp", "player", "status", "touches_pg", "rate", "replacement", "repl_rate", "vor", "disc", "cost_dc",
            "starter_total", "cost_total"]
    print(sc[(sc["cost_dc"] >= 1) | (sc["cost_total"] >= 1)].sort_values("cost_total", ascending=False)[show]
          .head(30).to_string(index=False))

    # adjusted chart: listed out (p >= 0.5) for the coming game, all units
    exr = D.export_roles(EXPORTS, DATA, teams)
    lo = lf.merge(nxt, on=["game_id", "team_id"])
    lo = lo[lo["p"] >= 0.5].drop_duplicates(["team_id", "athlete_id"])
    lo = lo.assign(name=lo["athlete_id"].map(names))
    offs = sc[sc["charted_team"]][["athlete_id", "grp", "align"]].assign(
        team_id=sc.loc[sc["charted_team"], "team"].map({v: k for k, v in tname.items()}))
    adj = D.adjusted_chart(chart, lo[["team_id", "athlete_id", "status", "name"]], exr, offs, teams)
    cols = ["team", "unit", "position", "depth", "player", "athlete_id", "role", "status", "inserted", "note", "starter_share"]
    adj[cols].to_csv(a.out / f"adjusted_depth_charts_week{a.week}_2026.csv", index=False)
    ins = adj[adj["inserted"]]
    print(f"adjusted charts: {len(ins)} listed-out players moved back up to their natural spot "
          f"({ins['unit'].value_counts().to_dict()}; at first team {int((ins['depth'] == 1).sum())})")
    print(ins[["team", "unit", "position", "player", "status", "note", "starter_share"]].head(40).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())

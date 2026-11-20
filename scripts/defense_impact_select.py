"""Defensive player impact scores vs model_v2 and PR 35's depth-chart scores (2026-11-20, test bench).

Josiah's ask: use defensive players in the player impact scores. Defenders are priced by the plays
they win (box-score sacks, TFL, INT, PBU, hurries at their EPA swing) over the player who fills in
(`cfb.lab.defense_impact`). Same tests as PR 35, plus 2018-2024 realized for the defensive term.

Inputs (as PR 35):
    python -m cfb.core.build
    python scripts/impact_base.py
    python scripts/injury_select.py --build

Versions (fixed points, nothing fitted; game term = home minus away through model_v2's slope):
    dc_total            PR 35's recommended offense term (QB / RB / WR / TE, starter by season touches)
    def_fill            defenders: value over the league fill-in rate
    def_own             defenders: value over the next healthy non-regular (his rate per involvement)
    def_count           every regular out at the league mean (starter - fill), no player value
    def_fill_streak     realized tests only: def_fill, an isolated blank box-score game read as played
    dc_total+def_*      offense + defense

    python scripts/defense_impact_select.py [--out DIR] [--week 12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_impact_select as DS  # noqa: E402
import injury_select as IS  # noqa: E402

from cfb.lab import availability as A  # noqa: E402
from cfb.lab import defense_impact as F  # noqa: E402
from cfb.lab import depth_impact as D  # noqa: E402
from cfb.model.models import log_loss, sigmoid  # noqa: E402

DATA = Path("data")
EXP = DATA / "exploration"
EXPORTS = DS.EXPORTS
ALLC = ["qb_cost", "rb_cost", "wr_cost", "te_cost"]


def tot(tf, cols):
    t = tf[["game_id", "team_id"]].copy()
    t["cost"] = tf[cols].sum(axis=1) if len(tf) else 0.0
    return t


def add(*cs):
    x = pd.concat([c.set_index(["game_id", "team_id"])["cost"] for c in cs], axis=1).fillna(0.0).sum(axis=1)
    return x.rename("cost").reset_index()


def def_versions(tf: pd.DataFrame, mu: dict, suffix: str = "") -> dict:
    cnt = tf[["game_id", "team_id"]].copy()
    cnt["cost"] = sum(tf[f"{g.lower()}_n_out"] * max(mu[g]["start"] - mu[g]["fill"], 0.0) for g in F.DGROUPS)
    return {f"def_fill{suffix}": tot(tf, [f"{g.lower()}_cost" for g in F.DGROUPS]),
            f"def_own{suffix}": tot(tf, [f"{g.lower()}_cost_own" for g in F.DGROUPS]),
            f"def_count{suffix}": cnt,
            **{f"def_{g.lower()}{suffix}": tot(tf, [f"{g.lower()}_cost"]) for g in F.DGROUPS}}


def score(d: pd.DataFrame, costs: dict, versions, ref: str | None) -> list[dict]:
    pp = {v: sigmoid(d["lin"] + d["calib_s"] * DS.game_adj(d, costs[v], "cost")) for v in versions}
    out = []
    for v in versions:
        adj = DS.game_adj(d, costs[v], "cost")
        diff, se = DS.boot(d, pp[v], d["p0"])
        dr, sr = DS.boot(d, pp[v], pp[ref]) if ref and v != ref and ref in pp else (np.nan, np.nan)
        out.append({"version": v, "n": len(d), "n_adjusted": int((adj.abs() > 1e-9).sum()),
                    "ll_v2": round(log_loss(d["y"], d["p0"]), 4), "ll": round(log_loss(d["y"], pp[v]), 4),
                    "vs_v2": round(diff, 4), "se": round(se, 4),
                    "vs_dc_total": None if np.isnan(dr) else round(dr, 4), "se_vs_dc_total": None if np.isnan(sr) else round(sr, 4),
                    "acc_v2": round(((d["p0"] >= 0.5) == (d["y"] == 1)).mean(), 4),
                    "acc": round(((pp[v] >= 0.5) == (d["y"] == 1)).mean(), 4), "mean_abs_adj": round(adj.abs().mean(), 3)})
    return out


def slope(d: pd.DataFrame, cost: pd.DataFrame) -> tuple[float, float, int]:
    """Points of residual (margin - model_v2 prediction) per point of game term, through the origin."""
    x = DS.game_adj(d, cost, "cost").to_numpy()
    r = (d["margin"] - d["pred"]).to_numpy()
    mk = x != 0
    if mk.sum() < 10:
        return np.nan, np.nan, int(mk.sum())
    b = float((x * r).sum() / (x * x).sum())
    s2 = float(((r - b * x) ** 2).sum() / (len(r) - 1))
    return b, float(np.sqrt(s2 / (x * x).sum())), int(mk.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=EXP / "defense_impact")
    ap.add_argument("--week", type=int, default=12)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    u = pd.read_parquet(EXP / "avail_player_games.parquet")
    base = pd.read_parquet(EXP / "impact_base.parquet")
    teams = pd.read_parquet(DATA / "core" / "teams.parquet")
    gi = pd.read_parquet(DATA / "core" / "game_index.parquet")
    regular = set(gi.loc[gi["season_type"] == "regular", "game_id"])
    sched = A.team_schedule(DATA, range(2016, 2027))
    chart = D.load_chart(DATA)

    # ---- defensive values
    ev = F.event_values(DATA)
    dg = F.defender_games(DATA, u[u["season"] >= 2016], ev, chart, teams)
    K = F.reliability_k(dg)
    mu_d = F.league_means(dg, sched, regular)
    print("event values (points):", {k: round(v, 3) for k, v in ev.items()})
    print("K (games):", {k: round(v, 1) for k, v in K.items()})
    print("league disruption per game:", {g: {k: round(v, 3) for k, v in m.items()} for g, m in mu_d.items()})

    # ---- 2026 listings (PR 34 / PR 35)
    sched26 = A.team_schedule(DATA, range(2025, 2027))
    hdi = A.hdi_listings(DATA, sched26, teams, IS.HDI)
    checks = IS.load_checks(IS.CHECKS)
    cfd = A.cfbdepth_listings(DATA, sched26, teams, IS.CFBDEPTH, checks, IS.DOWNLOAD)
    ro = pd.read_parquet(EXP / "avail_roles.parquet")
    r25 = A.status_rates(ro, hdi[hdi["source"] == "hdi2025"])
    rate26 = dict(r25)
    rate26.setdefault("Out", (0.99, 0))
    rate26["Available"] = (0.0, 0)
    l26 = pd.concat([cfd, hdi[hdi["source"] == "hdi2026"]]).drop_duplicates(["game_id", "team_id", "athlete_id"])
    l26 = l26.assign(p=l26["status"].map({k: v[0] for k, v in rate26.items()})).dropna(subset=["p"])
    p26 = l26.set_index(["game_id", "team_id", "athlete_id"])["p"]
    p26d = p26[~p26.index.duplicated()].to_dict()
    dgl = dg[dg["season"] == 2026].drop_duplicates("athlete_id").set_index("athlete_id")["grp"]
    ld = l26[l26["athlete_id"].isin(dgl.index)]
    print(f"2026 listings: {len(l26)} rows, defenders {len(ld)} ({ld['athlete_id'].map(dgl).value_counts().to_dict()})")

    # ---- PR 35 dc_total (offense)
    coff = D.chart_offense(chart)
    wr = D.wr_export_spots(EXPORTS, DATA, teams)
    rost = D.roster_groups(DATA, range(2017, 2027))
    lab_c = D.slot_labels(u, rost, coff, wr, use_chart=True)
    mu_o = {s: D.league_means(u, lab_c, s) for s in (2025, 2026)}
    s25, s26 = [2025], [2026]
    to_r, _ = D.team_costs(u, sched, lab_c, rost, mu_o, None, True, s25 + s26, regular, starter_by="total")
    to_l, _ = D.team_costs(u, sched, lab_c, rost, mu_o, p26d, False, s26, starter_by="total")
    off = {"real": tot(to_r, ALLC), "list": tot(to_l, ALLC)}

    # ---- defensive team costs
    seasons_r = list(range(2018, 2027))
    tr, dr = F.team_costs(dg, sched, mu_d, K, None, True, seasons_r, regular)
    trs, _ = F.team_costs(dg, sched, mu_d, K, None, True, seasons_r, regular, streak=True)
    tr.to_parquet(a.out / "def_team_costs_realized.parquet", index=False)
    trs.to_parquet(a.out / "def_team_costs_realized_streak.parquet", index=False)
    tl, dl = F.team_costs(dg, sched, mu_d, K, p26d, False, s26)
    tl.to_parquet(a.out / "def_team_costs_listed.parquet", index=False)
    dr.to_parquet(a.out / "def_details_realized.parquet", index=False)
    dl.to_parquet(a.out / "def_details_listed.parquet", index=False)
    print(f"regulars out: realized {int((dr['p_out'] >= 0.5).sum())} (2025 {int(((dr['p_out'] >= 0.5) & (dr['season'] == 2025)).sum())}), "
          f"listed 2026 {int((dl['p_out'] >= 0.5).sum())} (P > 0: {len(dl)})")
    costs = {"real": {**def_versions(tr, mu_d), **def_versions(trs, mu_d, "_streak")},
             "list": def_versions(tl, mu_d)}
    # one fitted number: residual points per point of cost, 2018-2024 realized with the streak rule
    # (cleanest absences), applied to 2025 (streak costs) and 2026 listed (def_fill costs)
    g0 = base[(base["season"].between(2018, 2024)) & (base["season_type"] == "regular")]
    beta, beta_se, _ = slope(g0, costs["real"]["def_fill_streak"])
    print(f"defensive cost scale fitted on 2018-2024 (streak): {beta:.2f} (SE {beta_se:.2f})")
    costs["real"]["def_fill_x"] = costs["real"]["def_fill_streak"].assign(cost=lambda x: x["cost"] * beta)
    costs["list"]["def_fill_x"] = costs["list"]["def_fill"].assign(cost=lambda x: x["cost"] * beta)
    for mode in ("real", "list"):
        c = costs[mode]
        c["dc_total+def_fill_x"] = add(off[mode], c["def_fill_x"])
        c["dc_total"] = off[mode]
        c["dc_total+def_fill"] = add(off[mode], c["def_fill"])
        c["dc_total+def_own"] = add(off[mode], c["def_own"])
        c["dc_total+def_count"] = add(off[mode], c["def_count"])
        if mode == "real":
            c["dc_total+def_fill_streak"] = add(off[mode], c["def_fill_streak"])

    # ---- evaluation
    conf = teams[teams["season"] == 2026].set_index("team_id")["conference"]
    nd = teams.loc[(teams["season"] == 2026) & (teams["team"] == "Notre Dame"), "team_id"]
    p4 = set(conf[conf.isin(IS.P4)].index) | set(nd)
    g = base[base["season"] >= 2018].copy()
    g["p4"] = g["home_id"].isin(p4) | g["away_id"].isin(p4)
    g["p0"] = sigmoid(g["lin"])
    reg_s = g["season_type"] == "regular"
    DEFV = ["def_fill", "def_own", "def_count", "def_dl", "def_lb", "def_db"]
    tests = {
        "2018-2024 realized, regular season, all FBS": ("real", (g["season"] <= 2024) & reg_s,
                                                        DEFV + ["def_fill_streak", "def_own_streak"]),
        "2025 realized, regular season, all FBS": ("real", (g["season"] == 2025) & reg_s, None),
        "2025 realized, regular season, P4/ND": ("real", (g["season"] == 2025) & reg_s & g["p4"], None),
        "2026 listed, all FBS": ("list", g["season"] == 2026, None),
        "2026 listed, P4/ND": ("list", (g["season"] == 2026) & g["p4"], None),
        "2026 realized, all FBS": ("real", g["season"] == 2026, None),
    }
    full_r = ["dc_total", *DEFV, "def_fill_streak", "def_fill_x", "dc_total+def_fill", "dc_total+def_own",
              "dc_total+def_count", "dc_total+def_fill_streak", "dc_total+def_fill_x"]
    full_l = ["dc_total", *DEFV, "def_fill_x", "dc_total+def_fill", "dc_total+def_own", "dc_total+def_count",
              "dc_total+def_fill_x"]
    rows, preds = [], {}
    for lab_t, (mode, msk, vers) in tests.items():
        d = g[msk.to_numpy()].copy()
        vers = vers or (full_r if mode == "real" else full_l)
        for r in score(d, costs[mode], vers, None if lab_t.startswith("2018") else "dc_total"):
            rows.append({"test": lab_t, **r})
        preds[lab_t] = (mode, d)
    for lab_t, (k25, k26) in {"pooled 2025 realized + 2026 listed, all FBS": ("2025 realized, regular season, all FBS", "2026 listed, all FBS"),
                              "pooled 2025 realized + 2026 listed, P4/ND": ("2025 realized, regular season, P4/ND", "2026 listed, P4/ND")}.items():
        d = pd.concat([preds[k25][1], preds[k26][1]], ignore_index=True)
        cc = {v: pd.concat([costs["real"][v][costs["real"][v]["game_id"].isin(set(preds[k25][1]["game_id"]))],
                            costs["list"][v]]) for v in full_l}
        for r in score(d, cc, full_l, "dc_total"):
            rows.append({"test": lab_t, **r})
    res = pd.DataFrame(rows)
    res.to_csv(a.out / "results.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + res.to_string(index=False))

    # ---- in-sample slopes: residual points per point of defensive cost (1.0 = face value)
    srows = []
    for lab_t, (mode, msk) in {"2018-2025 realized, regular season": ("real", (g["season"] <= 2025) & reg_s),
                               "2026 listed": ("list", g["season"] == 2026)}.items():
        d = g[msk.to_numpy()]
        for v in ["def_fill", "def_own", "def_count", "def_dl", "def_lb", "def_db"] + (["def_fill_streak"] if mode == "real" else []):
            b, se, nn = slope(d, costs[mode][v])
            srows.append({"test": lab_t, "version": v, "slope": round(b, 2), "se": round(se, 2), "n_games_adjusted": nn})
    sl = pd.DataFrame(srows)
    sl.to_csv(a.out / "slopes.csv", index=False)
    print("\n" + sl.to_string(index=False))

    # ---- this week's defensive impact scores: regulars vs the next man up on this week's chart
    upto = pd.Timestamp("2027-02-01", tz="UTC")  # every scheduled 2026 game; the week filter picks
    schedf = A.team_schedule(DATA, [2026], upcoming_until=upto).merge(gi[["game_id", "week", "season_type"]], on="game_id")
    schedf = schedf[(schedf["week"] <= a.week) & (schedf["season_type"] == "regular")].drop(columns="season_type")
    cfd_f = A.cfbdepth_listings(DATA, schedf, teams, IS.CFBDEPTH, checks, IS.DOWNLOAD)
    cfd_f = IS.forward_listings(cfd_f, schedf, teams, schedf.loc[schedf["week"] == a.week, "game_id"])
    lf = cfd_f.assign(p=cfd_f["status"].map({k: v[0] for k, v in rate26.items()})).dropna(subset=["p"])
    pf = lf.set_index(["game_id", "team_id", "athlete_id"])["p"]
    pf = pf[~pf.index.duplicated()].to_dict()
    nxt = schedf[schedf["week"] == a.week][["game_id", "team_id"]]
    wk_file = EXP / "depth_impact" / f"depth_charts_week{a.week}_2026.csv"
    if not wk_file.exists():
        print(f"\n{wk_file} missing: run scripts/depth_impact_select.py --week {a.week} first for the week chart")
        return 0
    wkc = pd.read_csv(wk_file)
    repl_wk = F.chart_defense(wkc, teams)
    _, dw = F.team_costs(dg, schedf, mu_d, K, pf, False, [2026], record_all=True, chart_repl=repl_wk)
    ros26 = pd.read_parquet(DATA / "tables" / "roster" / "2026.parquet", columns=["id", "firstName", "lastName"])
    ros26["athlete_id"] = pd.to_numeric(ros26["id"], errors="coerce")
    names = dict(zip(ros26["athlete_id"], (ros26["firstName"].fillna("") + " " + ros26["lastName"].fillna("")).str.strip()))
    tname = dict(zip(teams.loc[teams["season"] == 2026, "team_id"], teams.loc[teams["season"] == 2026, "team"]))
    stat = lf.drop_duplicates(["game_id", "team_id", "athlete_id"]).set_index(["game_id", "team_id", "athlete_id"])["status"]
    sc = dw.merge(nxt, on=["game_id", "team_id"])
    sc["team"] = sc["team_id"].map(tname)
    sc["player"] = sc["athlete_id"].map(names)
    sc["replacement"] = sc["repl_id"].map(names)
    sc["status"] = [stat.get((gg, t, at)) for gg, t, at in zip(sc["game_id"], sc["team_id"], sc["athlete_id"])]
    sc["charted_team"] = sc["team_id"].isin(set(repl_wk))
    sc["cost_fill"] = sc["vor_fill"] * sc["p_out"]
    sc["cost_chart"] = np.where(sc["charted_team"], sc["vor_own"], sc["vor_fill"]) * sc["p_out"]
    keep = ["team", "grp", "player", "athlete_id", "status", "p_out", "games", "inv_pg", "dv_pg", "rate", "replacement",
            "repl_rate", "vor_fill", "vor_own", "cost_fill", "cost_chart", "charted_team", "game_id", "team_id"]
    sc = sc[keep].sort_values(["team", "grp", "rate"], ascending=[True, True, False])
    sc.to_csv(a.out / f"def_impact_scores_week{a.week}_2026.csv", index=False)
    print(f"\nweek {a.week} defensive regulars: {len(sc)}, listed out (P >= 0.5) {int((sc['p_out'] >= 0.5).sum())}, "
          f"cost >= 0.5 pt (fill) {int((sc['cost_fill'] >= 0.5).sum())}")
    show = ["team", "grp", "player", "status", "dv_pg", "rate", "replacement", "repl_rate", "cost_fill", "cost_chart"]
    print(sc[sc["p_out"] > 0].sort_values("cost_fill", ascending=False)[show].head(25).to_string(index=False))
    tc = sc.groupby(["game_id", "team_id", "team"], as_index=False)[["cost_fill", "cost_chart"]].sum()
    tc.to_csv(a.out / f"def_team_costs_week{a.week}_2026.csv", index=False)

    # ---- inputs of the local app's player toggles (scripts/availability_export.py packs them)
    app = a.out / f"app_week{a.week}_2026"
    app.mkdir(parents=True, exist_ok=True)
    wk_teams = set(nxt["team_id"])
    dg[(dg["season"] == 2026) & dg["team_id"].isin(wk_teams)][
        ["season", "game_id", "team_id", "athlete_id", "grp", "inv", "dv"]].to_parquet(app / "dg.parquet", index=False)
    pv = F.prev_season(dg)
    pv[pv["season"] == 2026].to_parquet(app / "dprev.parquet", index=False)
    chart[chart["unit"] == "defense"][["team", "unit", "position", "depth", "player", "athlete_id", "role"]].to_parquet(
        app / "def_chart.parquet", index=False)
    sc.to_parquet(app / "def_scores.parquet", index=False)
    import json as _json
    (app / "def.json").write_text(_json.dumps({"mu": mu_d, "K": K, "ev": ev}))
    print(f"app defense inputs -> {app}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

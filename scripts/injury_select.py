"""Injury-adjusted test model vs model_v2 (2026-11-19, test bench; nothing in a model config changes).

Inputs (build first):
    python scripts/impact_base.py                 # model_v2 season-forward margins (data/exploration/impact_base.parquet)
    python scripts/injury_select.py --build       # player games + roles (data/exploration/avail_*.parquet)

The test model keeps model_v2 and adds an injury term in points:

    margin = model_v2 margin + beta . (home - away injury columns)
    p      = sigmoid(lin_v2 + calib_s * beta . (home - away injury columns))

Injury columns per team (src/cfb/lab/availability.py): QB value over replacement x p_out, QB value
x share of the season already missed ("baked in"), RB carry / WR, TE target share x p_out, defensive
tackle share x p_out, two or more key offensive players out, and (2026 only) the CFBDepth Impact
of listed players (any position, offensive line included).

p_out: availability listing for the game when one exists (2026: CFBDepth list + per-player checks +
conference reports; 2025: conference reports, conference games only), else the box-score streak
(missed the last k games). beta is fitted on earlier seasons (2018..S-1), where only the streak is
known, then applied to season S with its listings. 2026 also gets a version fitted on its own
earlier weeks for the CFBDepth Impact column.

    python scripts/injury_select.py [--build] [--checks DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.eval.bootstrap import paired_bootstrap
from cfb.lab import availability as A
from cfb.model.models import log_loss, sigmoid

DATA = Path("data")
EXP = DATA / "exploration"
# the scheduled injury pulls (scripts/injury_pull.py) point these at the newest pull and list
# the scheduled injury pulls (scripts/injury_pull.py) point these at the newest pull and list
CFBDEPTH_1118 = Path("/mnt/project-files/injury-reports/cfbdepth_2026-11-18/cfbdepth-player-updates.csv")
CFBDEPTH = Path(os.environ.get("CFB_CFBDEPTH_LIST", str(CFBDEPTH_1118)))
HDI = Path(os.environ.get("CFB_HDI_DIR", "/mnt/project-files/injury-reports/hdi_raw_2026-11-18"))
HDI_PULLED = os.environ.get("CFB_HDI_PULLED_UTC")  # UTC time of the HDI pull in HDI (forward use)
CHECKS = Path("/mnt/project-files/injury-reports/checks_2026-11-19")
DOWNLOAD = pd.Timestamp(os.environ.get("CFB_CFBDEPTH_DOWNLOAD", "2026-11-18"))
P4 = {"ACC", "Big Ten", "Big 12", "SEC"}
# PR 26 per-100%-share points (2018-2025 oracle regression, target share for WR/TE)
FIXED = {"rb_out": -1.94, "wr_out": -3.24, "te_out": -3.77}

SETS = {
    "qb": ["qb_out", "qb_baked"],
    "qb+skill": ["qb_out", "qb_baked", "rb_out", "wr_out", "te_out"],
    "qb+skill+def+multi": ["qb_out", "qb_baked", "rb_out", "wr_out", "te_out", "def_out", "key_multi"],
    "all+baked": ["qb_out", "qb_baked", "rb_out", "wr_out", "te_out", "rb_baked", "wr_baked", "te_baked",
                  "def_out", "key_multi"],
}


def build():
    seasons = range(2014, 2027)
    u = A.player_games(DATA, seasons)
    u.to_parquet(EXP / "avail_player_games.parquet", index=False)
    ro = A.roles(u, A.team_schedule(DATA, seasons))
    ro.to_parquet(EXP / "avail_roles.parquet", index=False)
    print("player games", len(u), "role rows", len(ro))


def forward_listings(cfd: pd.DataFrame, sched: pd.DataFrame, teams: pd.DataFrame, week_games) -> pd.DataFrame:
    """The coming week's listings: the CFBDepth list, and, when a scheduled HDI pull is set
    (CFB_HDI_PULLED_UTC), that pull's conference reports for the week's games. The pull wins where
    both list a player if it is from the list's download date or later (its reports are newer;
    'Available' there clears a listing); otherwise the list wins. HDI fills the list's gaps."""
    if not HDI_PULLED:
        return cfd
    h = A.hdi_listings(DATA, sched, teams, HDI)
    h = h[(h["source"] == "hdi2026") & h["game_id"].isin(set(week_games))].assign(source="hdi_pull")
    if h.empty:
        return cfd
    pulled = pd.Timestamp(HDI_PULLED).tz_convert("America/New_York").tz_localize(None).normalize()
    parts = [h, cfd] if pulled >= DOWNLOAD else [cfd, h]
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["game_id", "team_id", "athlete_id"])
    print(f"forward listings: CFBDepth {len(cfd)}, HDI pull {len(h)} (pulled {HDI_PULLED}, "
          f"{'wins' if pulled >= DOWNLOAD else 'fills gaps'}), combined {len(out)}")
    return out


def load_checks(d: Path) -> pd.DataFrame:
    if CFBDEPTH != CFBDEPTH_1118:  # the checks are keyed by row of the 2026-11-18 list
        print(f"per-player checks skipped: they belong to {CFBDEPTH_1118}, not {CFBDEPTH}")
        return pd.DataFrame()
    files = sorted(glob.glob(str(d / "out_batch_*.csv")))
    if not files:
        return pd.DataFrame()
    c = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    c["row_id"] = pd.to_numeric(c["row_id"], errors="coerce").astype("Int64")
    print(f"per-player checks: {len(c)} rows from {len(files)} batches; confidence "
          f"{c['confidence'].value_counts().to_dict()}")
    return c


def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def boot(d, p, p0):
    return paired_bootstrap(d["y"].to_numpy(), np.asarray(p), np.asarray(p0),
                            clusters=(d["season"].astype(str) + "-" + d["week"].astype(str)).to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--checks", type=Path, default=CHECKS)
    ap.add_argument("--out", type=Path, default=EXP / "injury_model")
    ap.add_argument("--lam", type=float, default=20.0)
    ap.add_argument("--exports", type=Path, default=Path("/mnt/project-files/injury-reports/cfbdepth_exports_2026-11-18"),
                    help="CFBDepth position exports (D:\\claude cfbdepth-*-advanced/stats/ratings.csv)")
    a = ap.parse_args()
    if a.build or not (EXP / "avail_roles.parquet").exists():
        build()
    ro = pd.read_parquet(EXP / "avail_roles.parquet")
    base = pd.read_parquet(EXP / "impact_base.parquet")
    teams = pd.read_parquet(DATA / "core" / "teams.parquet")
    sched = A.team_schedule(DATA, range(2025, 2027))

    # ---- listings and status -> P(out)
    hdi = A.hdi_listings(DATA, sched, teams, HDI)
    checks = load_checks(a.checks)
    cfd = A.cfbdepth_listings(DATA, sched, teams, CFBDEPTH, checks, DOWNLOAD)
    print(f"HDI listings {len(hdi)} ({hdi['source'].value_counts().to_dict()}); "
          f"CFBDepth listings {len(cfd)} ({cfd['source'].value_counts().to_dict() if len(cfd) else {}})")
    r25 = A.status_rates(ro, hdi[hdi["source"] == "hdi2025"])
    r26 = A.status_rates(ro, hdi[hdi["source"] == "hdi2026"])
    print("P(absent | status), role players:  2025", {k: (round(v[0], 3), v[1]) for k, v in r25.items()})
    print("                                   2026", {k: (round(v[0], 3), v[1]) for k, v in r26.items()})
    # 2026 uses rates measured on 2025 reports; 2025 uses 2026's (another season's outcomes)
    rate26 = {k: v for k, v in r25.items()}
    rate25 = {k: v for k, v in r26.items()}
    for r in (rate26, rate25):
        r.setdefault("Out", (0.99, 0))
        r["Available"] = (0.0, 0)
    # listings (project rule, larger dataset first): the CFBDepth list, then conference reports for the
    # players and games it does not cover (97% of the 1,800 overlapping rows agree)
    l26 = pd.concat([cfd, hdi[hdi["source"] == "hdi2026"]]).drop_duplicates(["game_id", "team_id", "athlete_id"])
    l25 = hdi[hdi["source"] == "hdi2025"]

    mu = {s: A.qb_league_means(ro, s) if s > 2014 else A.qb_league_means(ro, 2015) for s in ro["season"].unique()}
    po = A.streak_p_out(ro)
    po = A.overlay_listings(po, l25, rate25)
    po = A.overlay_listings(po, l26, rate26)
    print("role rows by p_out source:", po[po["season"] >= 2024].groupby("season")["src"].value_counts()
          .unstack().fillna(0).astype(int).to_dict("index"))

    # CFBDepth Impact (2026): any listed player, offensive line included
    d = pd.read_csv(CFBDEPTH)
    d["row_id"] = range(1, len(d) + 1)
    imp = cfd.merge(d[["row_id", "Impact", "Rating", "Pos"]], on="row_id", how="left") if len(cfd) else cfd
    if len(imp):
        imp["p"] = imp["status"].map({k: v[0] for k, v in rate26.items()}).fillna(0.0)
        imp["impact_out"] = imp["p"] * pd.to_numeric(imp["Impact"], errors="coerce").fillna(0.0)
        imp = imp.groupby(["game_id", "team_id"], as_index=False)["impact_out"].sum()

    tf = A.team_features(po, mu)
    tfl = A.team_features(po.assign(p_out=np.where(po["src"].isin(["none", "streak"]), 0.0, po["p_out"])), mu)
    lcols = ["qb_out", "rb_out", "wr_out", "te_out"]
    if len(imp):
        tf = tf.merge(imp, on=["game_id", "team_id"], how="left")
    tf["impact_out"] = tf.get("impact_out", 0.0)
    tf["impact_out"] = tf["impact_out"].fillna(0.0)
    tf.to_parquet(EXP / "avail_team_features.parquet", index=False)

    cols = sorted({c for v in SETS.values() for c in v} | {"impact_out", "qb_out_orc", "rb_out_orc", "wr_out_orc",
                                                          "te_out_orc", "def_out_orc", "key_multi_orc", "qb_vor",
                                                          "listed_n"})
    h = tf.rename(columns={c: f"h_{c}" for c in cols})[["game_id", "team_id"] + [f"h_{c}" for c in cols]]
    w = tf.rename(columns={c: f"a_{c}" for c in cols})[["game_id", "team_id"] + [f"a_{c}" for c in cols]]
    g = base.merge(h.rename(columns={"team_id": "home_id"}), on=["game_id", "home_id"], how="left")
    g = g.merge(w.rename(columns={"team_id": "away_id"}), on=["game_id", "away_id"], how="left")
    for c in cols:
        g[f"h_{c}"] = g[f"h_{c}"].fillna(0.0)
        g[f"a_{c}"] = g[f"a_{c}"].fillna(0.0)
        g[f"d_{c}"] = g[f"h_{c}"] - g[f"a_{c}"]
    hl = tfl[["game_id", "team_id"] + lcols].rename(columns={"team_id": "home_id"} | {c: f"hl_{c}" for c in lcols})
    al = tfl[["game_id", "team_id"] + lcols].rename(columns={"team_id": "away_id"} | {c: f"al_{c}" for c in lcols})
    g = g.merge(hl, on=["game_id", "home_id"], how="left").merge(al, on=["game_id", "away_id"], how="left")
    for c in lcols:
        g[f"dl_{c}"] = g[f"hl_{c}"].fillna(0.0) - g[f"al_{c}"].fillna(0.0)
    g["resid"] = g["margin"] - g["pred"]
    conf = teams[teams["season"] == 2026].set_index("team_id")["conference"]
    nd = teams.loc[(teams["season"] == 2026) & (teams["team"] == "Notre Dame"), "team_id"]
    p4 = set(conf[conf.isin(P4)].index) | set(nd)
    g["p4"] = g["home_id"].isin(p4) | g["away_id"].isin(p4)

    # ---- oracle sizes (2018-2025): realized absences, residual vs model_v2
    tr = g[g["season"] <= 2025]
    oc = ["d_qb_out_orc", "d_rb_out_orc", "d_wr_out_orc", "d_te_out_orc", "d_def_out_orc", "d_key_multi_orc"]
    b = ridge(tr[oc].to_numpy(), tr["resid"].to_numpy(), a.lam)
    print("\nOracle (realized absences) points per unit, 2018-2025 residual vs model_v2:",
          dict(zip([c[2:] for c in oc], np.round(b, 2))))

    # ---- season-forward evaluation
    res = {}
    rows = []
    for name, feats in list(SETS.items()) + [("fixed", None), ("fixed_listed", "listed")]:
        preds = []
        for s in range(2020, 2027):
            trn = g[(g["season"] < s) & (g["season"] >= 2018)]
            te = g[g["season"] == s].copy()
            if feats in (None, "listed"):
                # QB value is a loss when out; skill coefficients are already negative per share
                pre = "d_" if feats is None else "dl_"
                adj = -te[pre + "qb_out"] + sum(te[f"{pre}{c}"] * v for c, v in FIXED.items())
                beta = None
            else:
                # fit on streak-based p_out in training seasons (what was knowable then)
                X = trn[[f"d_{c}" for c in feats]].to_numpy()
                beta = ridge(X, trn["resid"].to_numpy(), a.lam)
                adj = te[[f"d_{c}" for c in feats]].to_numpy() @ beta
            te["adj"] = adj
            te["p_new"] = sigmoid(te["lin"] + te["calib_s"] * te["adj"])
            te["beta"] = [None] * len(te) if beta is None or isinstance(feats, str) else [dict(zip(feats, np.round(beta, 3)))] * len(te)
            preds.append(te)
        res[name] = pd.concat(preds, ignore_index=True)

    def score(name, d, label):
        p0 = sigmoid(d["lin"])
        p1 = d["p_new"]
        bb = boot(d, p1, p0)
        acc0 = ((p0 >= 0.5) == (d["y"] == 1)).mean()
        acc1 = ((p1 >= 0.5) == (d["y"] == 1)).mean()
        rm0 = np.sqrt(((d["margin"] - d["pred"]) ** 2).mean())
        rm1 = np.sqrt(((d["margin"] - d["pred"] - d["adj"]) ** 2).mean())
        rows.append({"model": name, "set": label, "n": len(d), "ll_v2": round(log_loss(d["y"], p0), 4),
                     "ll_test": round(log_loss(d["y"], p1), 4), "diff": round(bb["diff"], 4), "se": round(bb["se"], 4),
                     "acc_v2": round(acc0, 4), "acc_test": round(acc1, 4), "rmse_v2": round(rm0, 2),
                     "rmse_test": round(rm1, 2)})

    for name, d in res.items():
        old = d[d["season"] <= 2025]
        score(name, old, "2020-2025 all")
        rows[-1]["seasons_better"] = "%d/6" % sum(
            log_loss(old.loc[old.season == s, "y"], old.loc[old.season == s, "p_new"])
            < log_loss(old.loc[old.season == s, "y"], sigmoid(old.loc[old.season == s, "lin"])) for s in range(2020, 2026))
        c25 = old[(old["season"] == 2025) & ((old["h_listed_n"] + old["a_listed_n"]) > 0)]
        if len(c25):
            score(name, c25, "2025 games with a conf-report listing")
        n26 = d[d["season"] == 2026]
        score(name, n26, "2026 all FBS")
        score(name, n26[n26["p4"]], "2026 P4/ND games")
        score(name, n26[(n26["h_listed_n"] + n26["a_listed_n"]) > 0], "2026 games with a listed role player")
        qb = n26[(n26["h_qb_out"] > 0) | (n26["a_qb_out"] > 0)]
        if len(qb):
            score(name, qb, "2026 games with p_out > 0 for a QB1")
        print(f"{name:<22} beta fitted 2018-2025: {d.loc[d['season'] == 2026, 'beta'].iloc[0]}")

    # ---- 2026 walk-forward on its own earlier weeks: CFBDepth Impact column (one coefficient)
    n26 = g[g["season"] == 2026].copy().sort_values("week")
    adj = np.zeros(len(n26))
    for i, wk in enumerate(sorted(n26["week"].unique())):
        prev = n26[n26["week"] < wk]
        m = (n26["week"] == wk).to_numpy()
        if len(prev) < 100 or prev["d_impact_out"].abs().sum() == 0:
            continue
        bimp = ridge(prev[["d_impact_out"]].to_numpy(), prev["resid"].to_numpy(), a.lam)[0]
        adj[m] = n26.loc[m, "d_impact_out"].to_numpy() * bimp
    n26["adj"] = adj
    n26["p_new"] = sigmoid(n26["lin"] + n26["calib_s"] * n26["adj"])
    res["cfbdepth_impact_wf"] = n26
    score("cfbdepth_impact_wf", n26, "2026 all FBS")
    score("cfbdepth_impact_wf", n26[n26["p4"]], "2026 P4/ND games")
    if len(n26):
        bfull = ridge(n26[["d_impact_out"]].to_numpy(), n26["resid"].to_numpy(), a.lam)[0]
        print(f"CFBDepth Impact: points per Impact unit (all 2026, in sample) {bfull:+.3f}")

    # ---- OL and defensive starters out (CFBDepth season exports), 2026 walk-forward on top of fixed_listed
    if a.exports.exists():
        vals = A.cfbdepth_unit_values(a.exports, DATA, teams)
        uo = A.unit_out(l26, vals, rate26)
        print(f"CFBDepth unit values: {len(vals)} OL/DL/LB/DB players; listed team-games with a unit starter out "
              f"{int((uo[['ol_out', 'dfs_out']].sum(axis=1) > 0).sum())}")
        fl = res["fixed_listed"]
        fl26 = fl[fl["season"] == 2026].copy()
        for side, key in (("h", "home_id"), ("a", "away_id")):
            fl26 = fl26.merge(uo.rename(columns={"team_id": key} | {c: f"{side}u_{c}" for c in uo.columns
                                                                   if c not in ("game_id", "team_id")}),
                              on=["game_id", key], how="left")
        ucols = ["ol_out", "dfs_out", "ol_out_r", "dfs_out_r"]
        for c in ucols:
            fl26[f"du_{c}"] = fl26[f"hu_{c}"].fillna(0.0) - fl26[f"au_{c}"].fillna(0.0)
        fl26["resid2"] = fl26["margin"] - fl26["pred"] - fl26["adj"]
        for lab, uc in (("count", ["du_ol_out", "du_dfs_out"]), ("rating-weighted", ["du_ol_out_r", "du_dfs_out_r"])):
            d = fl26.sort_values("week").copy()
            extra = np.zeros(len(d))
            for wk in sorted(d["week"].unique()):
                prev = d[d["week"] < wk]
                msk = (d["week"] == wk).to_numpy()
                if len(prev) < 100:
                    continue
                bu = ridge(prev[uc].to_numpy(), prev["resid2"].to_numpy(), a.lam)
                extra[msk] = d.loc[msk, uc].to_numpy() @ bu
            p_fl = d["p_new"].to_numpy()
            d["adj"] = d["adj"] + extra
            d["p_new"] = sigmoid(d["lin"] + d["calib_s"] * d["adj"])
            name = f"fixed_listed+units_{lab}"
            res[name] = d
            for sub_lab, msk in (("2026 all FBS", np.ones(len(d), bool)), ("2026 P4/ND games", d["p4"].to_numpy())):
                score(name, d[msk], sub_lab)
                bb = boot(d[msk], d["p_new"][msk], p_fl[msk])
                rows[-1]["vs_fixed_listed"] = f"{bb['diff']:+.4f} ({bb['se']:.4f})"
            bu_all = ridge(d[uc].to_numpy(), d["resid2"].to_numpy(), a.lam)
            print(f"{name}: points per starter out (all 2026, in sample) OL {bu_all[0]:+.2f}, defense {bu_all[1]:+.2f}")

    # ---- 2026 source ablation, fixed coefficients (no fitting): which listings carry the gain
    base26 = g[g["season"] == 2026][["game_id", "season", "week", "home_id", "away_id", "margin", "y", "pred", "lin",
                                     "calib_s", "p4"]]
    ro26 = ro[ro["season"] == 2026]
    st26 = A.streak_p_out(ro)
    st26 = st26[st26["season"] == 2026]
    srcs = {
        "streak only (no listings)": None,
        "pregame only, no streak: conf reports + CFBDepth report entries": ["cfbdepth_report"],
        "+ CFBDepth current status (Last Update to download)": ["cfbdepth_report", "cfbdepth_current"],
        "+ per-player check windows (all listings), no streak": ["cfbdepth_report", "cfbdepth_current", "check_window"],
        "all listings + streak (main run)": "all",
    }
    hdi26 = hdi[hdi["source"] == "hdi2026"]
    for lab, keep in srcs.items():
        if keep == "all":
            pp = A.overlay_listings(st26, l26, rate26)
        elif keep is None:
            pp = st26
        else:
            # built from the raw sources, so a conference-report row is kept even where the CFBDepth
            # list also covers that player and game
            sub = pd.concat([cfd[cfd["source"].isin(keep)], hdi26]).drop_duplicates(["game_id", "team_id", "athlete_id"])
            pp = st26.assign(p_out=0.0, src="none")
            pp = A.overlay_listings(pp, sub, rate26)
        f = A.team_features(pp, mu)
        hh = f.add_prefix("h_").rename(columns={"h_game_id": "game_id", "h_team_id": "home_id"})
        aa = f.add_prefix("a_").rename(columns={"a_game_id": "game_id", "a_team_id": "away_id"})
        d = base26.merge(hh, on=["game_id", "home_id"], how="left").merge(aa, on=["game_id", "away_id"], how="left")
        for c in ("qb_out",) + tuple(FIXED):
            d[f"d_{c}"] = d[f"h_{c}"].fillna(0.0) - d[f"a_{c}"].fillna(0.0)
        d["adj"] = -d["d_qb_out"] + sum(d[f"d_{c}"] * v for c, v in FIXED.items())
        d["p_new"] = sigmoid(d["lin"] + d["calib_s"] * d["adj"])
        score("fixed", d, f"2026 all FBS | {lab}")
        score("fixed", d[d["p4"]], f"2026 P4/ND | {lab}")

    out = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out / "results.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + out.to_string(index=False))
    keep = ["game_id", "season", "week", "home_id", "away_id", "margin", "y", "pred", "lin", "adj", "p_new",
            "d_qb_out", "d_qb_baked", "d_rb_out", "d_wr_out", "d_te_out", "d_def_out", "d_key_multi", "d_impact_out",
            "h_listed_n", "a_listed_n", "p4"]
    for name, d in res.items():
        d[[c for c in keep if c in d]].to_parquet(a.out / f"preds_{name}.parquet", index=False)
    po[po["season"] == 2026].to_parquet(a.out / "roles_2026_p_out.parquet", index=False)


if __name__ == "__main__":
    sys.exit(main())

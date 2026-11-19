"""Do team playbook tendencies explain what model_v2 misses? (second stage, 2026 walk-forward)

    python -m cfb.core.build && python -m cfb.core.playtext
    python -m cfb.predict --model model_v2 season-forward          # 2025 predictions (late-2025 training rows)
    python -m cfb.predict --model model_v2 walk-forward --seasons 2026
    python scripts/playbook_check.py [--data-dir data] [--scope P4]

Tendencies come from the detailed play text only (2025 week 9 on), so the rows are FBS vs FBS
games from 2025 week 10 on. For each game, each team's tendencies use only its earlier games
(2025 plays weighted 0.5 for 2026 games), shrunk toward the FBS mean with K_PLAYS plays.
Garbage time, kneels, spikes and no-plays are left out (cfb.lab.playbooks).

Columns (home minus away): offense under-center share, raw no-huddle share, pass rate over
expected, screen share, deep share, QB-run share, play-mix entropy; defense sack rate and hurry
rate per dropback.

Second stage: logistic regression with offset logit(model_v2 p_home_win), no intercept, ridge on
standardised columns, refit before every 2026 week on all earlier rows (late 2025 included). Scored on 2026
weeks against model_v2 alone: log loss and Brier, with a paired SE over weeks. Also: one-column
slopes of the margin residual (home margin - model_v2 pred_margin) on every row, in points per SD.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.lab import playbooks as pb

SCOPE = "P4"
K_PLAYS = 150.0
W_PRIOR_SEASON = 0.5
OFF_COLS = ("uc", "nh", "proe", "screen", "deep", "qbrun", "entropy")
DEF_COLS = ("sack", "hurry")


def team_games(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = pb.load(data_dir, (2025, 2026))
    p = pb.label_plays(b)
    p = p[~p["garbage_time"].fillna(False).astype(bool)].reset_index(drop=True)
    # situation model on 2025 only, so 2026 tendencies use no 2026 fit
    fit = p[p["season"] == 2025]
    X = pb._proe_design(fit)
    w = pb._logit_fit(X, fit["dropback"].to_numpy(float))
    p["xpass"] = 1 / (1 + np.exp(-np.clip(pb._proe_design(p) @ w, -30, 30)))
    p["uc"] = ~p["shotgun"]
    p["screen"] = p["group"] == "Screen / behind LOS"
    p["deep"] = p["group"] == "Deep 20+"
    p["qbrun"] = p["group"] == "QB run"
    p["sack"] = p["action"] == "sack"
    p["hurry"] = p["qb_hurried"].fillna(False).astype(bool) & p["dropback"]
    off = p.groupby(["game_id", "offense_id"]).agg(
        n=("play_id", "size"), uc=("uc", "sum"), nh=("no_huddle", "sum"), drop=("dropback", "sum"),
        xpass=("xpass", "sum"), screen=("screen", "sum"), deep=("deep", "sum"), qbrun=("qbrun", "sum"))
    mix = pd.crosstab([p["game_id"], p["offense_id"]], p["group"])
    off = off.join(mix.add_prefix("g_")).reset_index().rename(columns={"offense_id": "team_id"})
    dfn = p.groupby(["game_id", "defense_id"]).agg(
        dn=("dropback", "sum"), sack=("sack", "sum"), hurry=("hurry", "sum")).reset_index()
    dfn = dfn.rename(columns={"defense_id": "team_id"})
    return off, dfn


def asof(off: pd.DataFrame, dfn: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per (game, team): tendencies from the team's earlier games."""
    gi = games[["game_id", "season", "start_time_utc"]]
    off = off.merge(gi, on="game_id")
    dfn = dfn.merge(gi, on="game_id")
    gcols = [c for c in off.columns if c.startswith("g_")]
    lg = off[["n", "uc", "nh", "drop", "xpass", "screen", "deep", "qbrun"] + gcols].sum()
    league = {"uc": lg["uc"] / lg["n"], "nh": lg["nh"] / lg["n"], "proe": 0.0,
              "screen": lg["screen"] / lg["n"], "deep": lg["deep"] / lg["n"], "qbrun": lg["qbrun"] / lg["n"]}
    lmix = (lg[gcols] / lg["n"]).to_numpy(float)
    ld = dfn[["dn", "sack", "hurry"]].sum()
    league["sack"] = ld["sack"] / ld["dn"]
    league["hurry"] = ld["hurry"] / ld["dn"]
    rows = []
    sched = pd.concat([off[["game_id", "team_id", "season", "start_time_utc"]],
                       dfn[["game_id", "team_id", "season", "start_time_utc"]]]).drop_duplicates(["game_id", "team_id"])
    off_by = {t: g.sort_values("start_time_utc") for t, g in off.groupby("team_id")}
    def_by = {t: g.sort_values("start_time_utc") for t, g in dfn.groupby("team_id")}
    for r in sched.itertuples(index=False):
        out = {"game_id": r.game_id, "team_id": r.team_id}
        o = off_by.get(r.team_id)
        if o is not None:
            prev = o[o["start_time_utc"] < r.start_time_utc]
            wt = np.where(prev["season"].to_numpy() < r.season, W_PRIOR_SEASON, 1.0)
            n = float((wt * prev["n"]).sum())
            out["prior_plays"] = n
            s = lambda c: float((wt * prev[c]).sum())
            for c in ("uc", "nh", "screen", "deep", "qbrun"):
                out[c] = (s(c) + K_PLAYS * league[c]) / (n + K_PLAYS)
            out["proe"] = (s("drop") - s("xpass")) / (n + K_PLAYS)
            mix = ((wt[:, None] * prev[gcols].to_numpy(float)).sum(0) + K_PLAYS * lmix) / (n + K_PLAYS)
            mix = mix[mix > 0] / mix.sum()
            out["entropy"] = float(-(mix * np.log(mix)).sum())
        d = def_by.get(r.team_id)
        if d is not None:
            prev = d[d["start_time_utc"] < r.start_time_utc]
            wt = np.where(prev["season"].to_numpy() < r.season, W_PRIOR_SEASON, 1.0)
            dn = float((wt * prev["dn"]).sum())
            for c in DEF_COLS:
                out[c] = (float((wt * prev[c]).sum()) + K_PLAYS * league[c]) / (dn + K_PLAYS)
        rows.append(out)
    return pd.DataFrame(rows)


def load_predictions(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(data_dir) / "predictions" / "model_v2" / "*.parquet")))
    p = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return p.sort_values("cutoff_utc").drop_duplicates("game_id", keep="last")[["game_id", "p_home_win", "pred_margin"]]


def logit_offset_fit(X, y, off, lam):
    w = np.zeros(X.shape[1] + 1)
    Xb = np.column_stack([np.ones(len(X)), X])
    pen = np.full(Xb.shape[1], lam)
    pen[0] = 1e6  # no intercept: model_v2 is already calibrated, so only the columns are fitted
    for _ in range(50):
        z = off + Xb @ w
        q = 1 / (1 + np.exp(-z))
        g = Xb.T @ (q - y) + pen * w
        H = (Xb * (q * (1 - q))[:, None]).T @ Xb + np.diag(pen)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ridge", default="5,50,200", help="comma-separated ridge penalties to score")
    ap.add_argument("--cols", default=None, help="comma-separated subset of columns for the second stage")
    ap.add_argument("--scope", default="P4", help="subset to report: P4 (68 teams incl. Notre Dame) or a conference")
    a = ap.parse_args(argv)
    games = pd.read_parquet(Path(a.data_dir) / "core" / "game_index.parquet")
    teams = pd.read_parquet(Path(a.data_dir) / "core" / "teams.parquet")
    off, dfn = team_games(a.data_dir)
    t = asof(off, dfn, games)
    g = games[games["completed"].astype(bool) & games["home_fbs"].astype(bool) & games["away_fbs"].astype(bool)]
    g = g[((g["season"] == 2025) & (g["week"] >= 10)) | (g["season"] == 2026)]
    g = g.merge(load_predictions(a.data_dir), on="game_id")
    cols = OFF_COLS + DEF_COLS
    h = t.add_prefix("h_").rename(columns={"h_game_id": "game_id", "h_team_id": "home_id"})
    w = t.add_prefix("a_").rename(columns={"a_game_id": "game_id", "a_team_id": "away_id"})
    g = g.merge(h, on=["game_id", "home_id"]).merge(w, on=["game_id", "away_id"])
    g = g.dropna(subset=[f"h_{c}" for c in cols] + [f"a_{c}" for c in cols])
    g = g[(g["h_prior_plays"] > 0) & (g["a_prior_plays"] > 0)].reset_index(drop=True)
    for c in cols:
        g[f"d_{c}"] = g[f"h_{c}"] - g[f"a_{c}"]
    g["margin"] = g["home_points"] - g["away_points"]
    g = g[g["margin"] != 0].reset_index(drop=True)
    g["y"] = (g["margin"] > 0).astype(float)
    g["resid"] = g["margin"] - g["pred_margin"]
    ids = set(pb.scope_ids(teams[teams["season"] == 2026], a.scope))
    g["sec"] = g["home_id"].isin(ids) | g["away_id"].isin(ids)
    g["both"] = g["home_id"].isin(ids) & g["away_id"].isin(ids)
    global SCOPE
    SCOPE = a.scope
    print(f"rows {len(g)}: 2025 wk10+ {int((g.season == 2025).sum())}, 2026 {int((g.season == 2026).sum())}; "
          f"{a.scope} ({len(ids)} teams) games {int(g.sec.sum())}, both teams {int(g.both.sum())}")

    print("\nOne-column slopes of the margin residual (points per SD of the home-minus-away value), all rows")
    print(f"{'column':<10}{'pts/SD':>8}{'t':>7}{'  ' + a.scope + ' pts/SD':>12}{'t':>7}")
    for c in cols:
        for sub, label in ((g, "all"), (g[g.sec], "sec")):
            x = sub[f"d_{c}"].to_numpy(float)
            x = (x - x.mean()) / x.std()
            yv = sub["resid"].to_numpy(float)
            beta = (x * (yv - yv.mean())).sum() / (x * x).sum()
            se = np.sqrt(((yv - yv.mean() - beta * x) ** 2).sum() / (len(x) - 2) / (x * x).sum())
            if label == "all":
                line = f"{c:<10}{beta:>8.2f}{beta / se:>7.2f}"
            else:
                line += f"{beta:>12.2f}{beta / se:>7.2f}"
        print(line)

    use = tuple(a.cols.split(",")) if a.cols else cols
    for lam in (float(v) for v in a.ridge.split(",")):
        print(f"\nSecond stage on {', '.join(use)}, ridge {lam:g}")
        score(g, use, lam)
    return 0


def score(g: pd.DataFrame, cols, lam: float) -> None:
    X_all = g[[f"d_{c}" for c in cols]].to_numpy(float)
    off_all = np.log(g["p_home_win"] / (1 - g["p_home_win"])).to_numpy()
    out = []
    for wk in sorted(g.loc[g.season == 2026, "week"].unique()):
        tr = (g["season"] == 2025) | ((g["season"] == 2026) & (g["week"] < wk))
        te = (g["season"] == 2026) & (g["week"] == wk)
        mu, sd = X_all[tr].mean(0), X_all[tr].std(0)
        Xt = (X_all - mu) / sd
        coef = logit_offset_fit(Xt[tr], g.loc[tr, "y"].to_numpy(), off_all[tr], lam)
        z = off_all[te] + np.column_stack([np.ones(te.sum()), Xt[te]]) @ coef
        q2 = 1 / (1 + np.exp(-z))
        q1 = g.loc[te, "p_home_win"].to_numpy()
        yy = g.loc[te, "y"].to_numpy()
        ll = lambda q: -(yy * np.log(q) + (1 - yy) * np.log(1 - q))
        out.append(pd.DataFrame({"week": wk, "sec": g.loc[te, "sec"].to_numpy(), "both": g.loc[te, "both"].to_numpy(),
                                 "ll1": ll(q1), "ll2": ll(q2),
                                 "b1": (q1 - yy) ** 2, "b2": (q2 - yy) ** 2}))
    r = pd.concat(out)
    for sub, label in ((r, "all 2026 FBS games"), (r[r.sec], f"2026 games with a {SCOPE} team"),
                       (r[r.both], f"2026 {SCOPE} vs {SCOPE}")):
        wk = sub.groupby("week").agg(d=("ll2", "sum"), d1=("ll1", "sum"), n=("ll1", "size"))
        diff = (wk["d"] - wk["d1"]) / wk["n"].sum()
        se = np.sqrt(len(wk) * ((diff - diff.mean()) ** 2).sum() / max(len(wk) - 1, 1))
        print(f"  {label}: n {len(sub)}  log loss model_v2 {sub.ll1.mean():.4f}  + playbook {sub.ll2.mean():.4f}  "
              f"diff {sub.ll2.mean() - sub.ll1.mean():+.4f} (SE {se:.4f})  Brier {sub.b1.mean():.4f} -> {sub.b2.mean():.4f}")
    mu, sd = X_all.mean(0), X_all.std(0)
    coef = logit_offset_fit((X_all - mu) / sd, g["y"].to_numpy(), off_all, lam)
    print("  full-sample coefficients (log-odds per SD):", {c: round(float(v), 3) for c, v in zip(("const",) + tuple(cols), coef)})


if __name__ == "__main__":
    raise SystemExit(main())

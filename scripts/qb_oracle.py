"""Oracle backtest of a quarterback change (diagnostic only; never a model input).

For every game of 2018-2025 in the baseline set, compare the expected starter at the
cutoff (cfb.features.strength.qb) with the actual starter (the player with the most
passing plays in that game, known only afterwards). Measures, against model_v1's
season-forward predicted margin:

  * the mean margin residual of a team whose actual starter was not the expected one,
    split by whether the expected starter was the team's primary QB (share >= 0.5);
  * the points per unit of quality difference (actual minus expected, per QB play),
    by least squares of the residual on the home-minus-away quality change;
  * the size of a typical starter-to-backup drop in points: that slope times the median
    quality drop when a primary starter was replaced;
  * log loss of model_v1 with and without the oracle term (in-sample slope; upper bound).

    python scripts/qb_oracle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_select import SEASONS, load  # noqa: E402

from cfb.core.asof import Warehouse  # noqa: E402
from cfb.model import config as C  # noqa: E402
from cfb.model.models import MarginModel, log_loss, sigmoid  # noqa: E402
from cfb.features.strength import qb as Q  # noqa: E402

OUT = Path("data/exploration/qb_oracle.parquet")


def v1_margins(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = C.model_columns()
    for s in SEASONS:
        tr = X[(X["season"] >= C.TRAIN_START) & (X["season"] < s) & X["y"].notna()]
        te = X[(X["season"] == s) & X["in_eval"] & X["y"].notna()]
        m = MarginModel(cols, lam=C.LAM).fit(tr, tr["y"].to_numpy(), tr["margin"].to_numpy())
        rows.append(pd.DataFrame({"game_id": te["game_id"].to_numpy(), "pred": m.predict_margin(te),
                                  "lin": m.linear(te), "calib_s": m.calib[1]}))
    return pd.concat(rows, ignore_index=True)


def actual_starters(wh: Warehouse) -> pd.DataFrame:
    view = wh.view(wh.cutoffs()[-1])
    pg = Q.player_games(view)
    pg = pg.sort_values(["game_id", "team_id", "pass_plays", "athlete"], ascending=[True, True, False, True])
    return pg.drop_duplicates(["game_id", "team_id"])[["game_id", "team_id", "athlete"]].rename(
        columns={"athlete": "actual"})


def per_team(wh: Warehouse, games: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cutoff, g in games.groupby("cutoff_utc"):
        view = wh.view(cutoff)
        st, q = Q.qb_state(view)
        qi = q.set_index("athlete")["q"]
        sides = pd.concat([
            pd.DataFrame({"game_id": g["game_id"], "team_id": g["home_id"], "home": 1}),
            pd.DataFrame({"game_id": g["game_id"], "team_id": g["away_id"], "home": 0}),
        ])
        sides = sides.merge(actual, on=["game_id", "team_id"], how="left")
        need = [a for a in sides["actual"].dropna().unique() if a not in qi.index]
        pri = Q.player_priors(view, view.season, need).set_index("athlete")["prior"] if need else pd.Series(dtype=float)
        sides["q_act"] = sides["actual"].map(lambda a: qi.get(a, pri.get(a, np.nan)) if isinstance(a, str) else np.nan)
        e = st.reindex(sides["team_id"].to_numpy())
        sides["expected"] = e["athlete"].to_numpy()
        sides["q_exp"] = e["q"].to_numpy(dtype=float)
        sides["share"] = e["share"].to_numpy(dtype=float)
        sides["from_roster"] = e["rule"].to_numpy(dtype=float)
        out.append(sides)
    return pd.concat(out, ignore_index=True)


def main():
    wh = Warehouse("data")
    X, _ = load()
    m = v1_margins(X)
    ev = X[X["game_id"].isin(m["game_id"])][["game_id", "season", "week", "season_type", "cutoff_utc", "home_id",
                                             "away_id", "margin", "y"]].merge(m, on="game_id")
    if OUT.exists():
        t = pd.read_parquet(OUT)
    else:
        t = per_team(wh, ev, actual_starters(wh))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        t.to_parquet(OUT, index=False)
    pg = Q.player_games(wh.view(wh.cutoffs()[-1]))
    pp = pg.groupby(["game_id", "team_id", "athlete"])["pass_plays"].sum()
    team_pp = pg.groupby(["game_id", "team_id"])["pass_plays"].sum().rename("team_pp")
    t = t.join(team_pp, on=["game_id", "team_id"])
    t["exp_pp"] = [pp.get((g, tm, a), 0.0) if isinstance(a, str) else np.nan
                   for g, tm, a in zip(t["game_id"], t["team_id"], t["expected"])]
    t["act_share"] = [pp.get((g, tm, a), np.nan) for g, tm, a in zip(t["game_id"], t["team_id"], t["actual"])]
    t["act_share"] = t["act_share"] / t["team_pp"]
    t["changed"] = t["actual"].notna() & t["expected"].notna() & (t["actual"] != t["expected"])
    t["dq"] = t["q_act"] - t["q_exp"]
    t = t.merge(ev[["game_id", "margin", "pred", "season"]], on="game_id")
    sign = np.where(t["home"] == 1, 1.0, -1.0)
    t["resid"] = sign * (t["margin"] - t["pred"])  # from the team's side
    known = t[t["actual"].notna() & t["expected"].notna()]
    print(f"team-games {len(t):,}; both starters known {len(known):,}; changed {known['changed'].sum():,} "
          f"({known['changed'].mean():.1%})")
    for label, sub in (("in-season (last game)", known[known["from_roster"] == 0]),
                       ("before first game (roster)", known[known["from_roster"] == 1])):
        print(f"  {label}: {len(sub):,} team-games, changed {sub['changed'].mean():.1%}")
    def summary(name, s):
        se = s["resid"].std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
        print(f"  {name:<52} n {len(s):5d}  resid {s['resid'].mean():+6.2f} (SE {se:.2f})  "
              f"median dq {s['dq'].median():+.3f}")
    print("\nmargin residual vs model_v1, team's side (points):")
    summary("starter as expected", known[~known["changed"]])
    summary("changed", known[known["changed"]])
    prim = known[known["changed"] & (known["share"] >= 0.5) & (known["from_roster"] == 0)]
    summary("changed, expected was primary (share >= 0.5), in-season", prim)
    summary("  ... and new starter lower quality (dq < 0)", prim[prim["dq"] < 0])
    summary("changed, before the first game", known[known["changed"] & (known["from_roster"] == 1)])
    # Clean "QB out": the expected primary starter threw no pass in the game and one
    # replacement took >= 80% of the passing plays. In-game changes are left out: an
    # injury or benching during the game is partly a result of the game.
    clean = prim[(prim["exp_pp"] == 0) & (prim["act_share"] >= 0.8)]
    print("\nclean QB out (expected primary did not play, replacement >= 80% of pass plays):")
    summary("all", clean)
    summary("expected starter share >= 0.8", clean[clean["share"] >= 0.8])
    summary("expected starter share 0.5-0.8", clean[clean["share"] < 0.8])
    summary("in-game change (primary played, someone else led)", prim[prim["exp_pp"] > 0])
    Z = np.column_stack([np.ones(len(clean)), clean["dq"].fillna(0.0)])
    beta, *_ = np.linalg.lstsq(Z, clean["resid"].to_numpy(), rcond=None)
    sig2 = np.sum((clean["resid"].to_numpy() - Z @ beta) ** 2) / (len(clean) - 2)
    cov = sig2 * np.linalg.inv(Z.T @ Z)
    print(f"  clean: resid = a + b * dq, a {beta[0]:+.2f} (SE {np.sqrt(cov[0, 0]):.2f}), "
          f"b {beta[1]:.1f} (SE {np.sqrt(cov[1, 1]):.1f}); dq quartiles "
          f"{np.round(clean['dq'].quantile([0.25, 0.5, 0.75]).to_numpy(), 3).tolist()}")
    for season, grp in clean.groupby("season"):
        print(f"    {season}: n {len(grp):3d} resid {grp['resid'].mean():+.2f}")

    # Slope of the game residual on the home-minus-away quality change.
    g = t.pivot_table(index="game_id", columns="home", values="dq", aggfunc="first").fillna(0.0)
    g.columns = ["dq_away", "dq_home"]
    g = g.join(ev.set_index("game_id")[["margin", "pred", "y", "lin", "calib_s", "season"]])
    g["ddq"] = g["dq_home"] - g["dq_away"]
    r = g["margin"] - g["pred"]
    x = g["ddq"].to_numpy()
    b = float(np.sum(x * r) / np.sum(x * x))
    se = float(np.sqrt(np.sum((r - b * x) ** 2) / (len(x) - 1) / np.sum(x * x)))
    print(f"\nresidual = b * (dq_home - dq_away): b {b:.1f} points per unit (SE {se:.1f}); "
          f"games with a change {int((g['ddq'] != 0).sum()):,}")
    drop = prim["dq"].median()
    print(f"typical starter-to-backup drop (median dq, primary replaced in-season) {drop:+.3f} per play "
          f"-> {b * drop:+.2f} points per game (SE {se * abs(drop):.2f})")
    lin_or = g["lin"] + g["calib_s"] * b * g["ddq"]
    print(f"log loss model_v1 {log_loss(g['y'], sigmoid(g['lin'])):.4f}, with oracle term "
          f"{log_loss(g['y'], sigmoid(lin_or)):.4f} (in-sample slope, upper bound)")
    for s in sorted(g["season"].unique()):
        gs = g[g["season"] == s]
        print(f"  {s}: n {len(gs)}  changes {int((gs['ddq'] != 0).sum())}  "
              f"{log_loss(gs['y'], sigmoid(gs['lin'])):.4f} -> {log_loss(gs['y'], sigmoid(lin_or[gs.index])):.4f}")


if __name__ == "__main__":
    main()

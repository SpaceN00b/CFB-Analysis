"""Season-forward check of the strength ratings against CFBD pregame Elo (2018-2025).

Games: completed FBS vs FBS games of 2015-2025 with a closing spread and CFBD pregame Elo
(the 6,024-game evaluation set of the build plan for 2018-2025). For each season S from
2018 to 2025 a logistic model (home indicator plus one rating gap, no intercept, as in
the exploration's effects.py) is fitted on seasons 2015 to S-1 and scored on S.

Ratings are taken through the feature code at each game's cutoff, so they see only what
was known then. 2026 is never read.

    python scripts/strength_check.py [--data-dir data] [--lam 4] [--out path.parquet]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from cfb.core import Warehouse
from cfb.features.strength import elo as E
from cfb.features.strength import power as PW

PRIORITY = {"consensus": 0, "Bovada": 1, "DraftKings": 2, "ESPN Bet": 3}


def eval_games(wh: Warehouse) -> pd.DataFrame:
    gi = wh.game_index
    rows = []
    ln = wh.prepared("lines").frame
    for gid, lines in zip(ln["id"], ln["lines"]):
        ls = [x for x in json.loads(lines) if x.get("spread") is not None]
        if ls:
            ls.sort(key=lambda x: PRIORITY.get(x["provider"], 9))
            rows.append((gid, float(ls[0]["spread"])))
    lines = pd.DataFrame(rows, columns=["game_id", "spread"]).drop_duplicates("game_id")
    pg = wh.prepared("game_pregame").frame
    g = gi[gi["completed"] & gi["home_fbs"] & gi["away_fbs"] & (gi["season"].between(2015, 2025))]
    g = g.merge(lines, on="game_id").merge(pg, on="game_id")
    g = g[g["home_pregame_elo"].notna() & g["away_pregame_elo"].notna()].copy()
    g["home"] = 1.0 - g["neutral"].astype(float)
    g["home_win"] = (g["home_points"] > g["away_points"]).astype(float)
    g["elo_gap"] = (g["home_pregame_elo"] - g["away_pregame_elo"]) / 100.0
    return g.reset_index(drop=True)


def logit_fit(y, X, it=50):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b))
        H = (X.T * (p * (1 - p))) @ X + 1e-8 * np.eye(X.shape[1])
        b = b + np.linalg.solve(H, X.T @ (y - p))
    return b


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def forward(d: pd.DataFrame, cols, seasons=range(2018, 2026)):
    ys, ps, per = [], [], {}
    for s in seasons:
        tr, te = d[d["season"] < s], d[d["season"] == s]
        b = logit_fit(tr["home_win"].to_numpy(), tr[cols].to_numpy())
        p = 1 / (1 + np.exp(-te[cols].to_numpy() @ b))
        ys.append(te["home_win"].to_numpy())
        ps.append(p)
        per[s] = (len(te), log_loss(te["home_win"].to_numpy(), p), float(((p > 0.5) == (te["home_win"] > 0.5)).mean()))
    y, p = np.concatenate(ys), np.concatenate(ps)
    return log_loss(y, p), float(((p > 0.5) == (y > 0.5)).mean()), per


def ratings(wh: Warehouse, games: pd.DataFrame, params) -> pd.DataFrame:
    out = []
    for cutoff, slate in games.groupby("cutoff_utc"):
        view = wh.view(cutoff)
        st = PW.power_state(view, params)
        h, a = st.lookup(slate["home_id"]), st.lookup(slate["away_id"])
        teams = view.table("teams", seasons=view.season)
        elo = E.ratings_at(view, view.season, teams)
        out.append(pd.DataFrame({
            "game_id": slate["game_id"].to_numpy(),
            "pwr_gap": (h["net"] - a["net"]).to_numpy() / 10.0,
            "prior_gap": (h["prior_net"] - a["prior_net"]).to_numpy() / 10.0,
            "home_games": h["games"].to_numpy(), "away_games": a["games"].to_numpy(),
            "pred_home_pts": (h["off"] + a["def"]).to_numpy() - st.fit.mu + st.fit.hfa * (1 - slate["neutral"].astype(float)).to_numpy(),
            "pred_away_pts": (a["off"] + h["def"]).to_numpy() - st.fit.mu - st.fit.hfa * (1 - slate["neutral"].astype(float)).to_numpy(),
            "ihelo_gap": (elo.reindex(slate["home_id"]).to_numpy() - elo.reindex(slate["away_id"]).to_numpy()) / 100.0,
        }))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--lam", type=float, nargs="*", default=[PW.POWER.lam])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    wh = Warehouse(args.data_dir)
    games = eval_games(wh)
    n_eval = int((games["season"] >= 2018).sum())
    print(f"evaluation games 2018-2025: {n_eval} (training seasons from 2015)")
    base = forward(games, ["home", "elo_gap"])
    print(f"CFBD pregame Elo + home: log loss {base[0]:.4f}, accuracy {base[1]:.3f}")
    for lam in args.lam:
        t = time.time()
        r = games.merge(ratings(wh, games, replace(PW.POWER, lam=lam)), on="game_id")
        res = forward(r, ["home", "pwr_gap"])
        print(f"\npower net gap + home (lam={lam}): log loss {res[0]:.4f}, accuracy {res[1]:.3f}  ({time.time() - t:.0f}s)")
        print(f"  in-house Elo gap + home:       log loss {forward(r, ['home', 'ihelo_gap'])[0]:.4f}")
        print(f"  preseason prior gap + home:    log loss {forward(r, ['home', 'prior_gap'])[0]:.4f}")
        print("  per season: season n power_ll power_acc elo_ll elo_acc")
        for s in range(2018, 2026):
            n, ll, acc = res[2][s]
            print(f"  {s} {n} {ll:.4f} {acc:.3f} {base[2][s][1]:.4f} {base[2][s][2]:.3f}")
        wins = sum(res[2][s][1] < base[2][s][1] for s in range(2018, 2026))
        print(f"  seasons better than Elo: {wins} of 8")
        b = logit_fit(r["home_win"].to_numpy(), r[["home", "pwr_gap"]].to_numpy())
        print(f"  logistic on all 2015-2025: home {b[0]:.3f}, gap per 10 points {b[1]:.3f} -> WIN_SCALE {10 / b[1]:.2f}")
        resid = np.concatenate([r["home_points"] - r["pred_home_pts"], r["away_points"] - r["pred_away_pts"]])
        print(f"  team points residual SD {np.nanstd(resid):.2f} (POINTS_SD)")
        early = r[(r["home_games"] < 4) | (r["away_games"] < 4)]
        late = r.drop(early.index)
        print(f"  log loss, either team under 4 games: {forward(early, ['home', 'pwr_gap'])[0]:.4f} (n={int((early['season'] >= 2018).sum())});"
              f" both 4+: {forward(late, ['home', 'pwr_gap'])[0]:.4f}")
        if args.out:
            r.to_parquet(args.out)


if __name__ == "__main__":
    main()

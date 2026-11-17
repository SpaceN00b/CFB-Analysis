"""Season-forward model selection for W8 (seasons through 2025 only; 2026 is never read).

Loads the stored feature groups, builds the game matrix, and scores candidate
configurations season-forward: fit on seasons [TRAIN_START, S-1], score S, S = 2018..2025,
on the 6,024 FBS vs FBS games with a closing spread and CFBD pregame Elo (the baseline set).

    python scripts/model_select.py ablation        # group ablation for the logit model
    python scripts/model_select.py tune            # lambda, era half-life, train start, family
    python scripts/model_select.py gbt             # gradient-boosted trees (needs scikit-learn)
    python scripts/model_select.py final           # the frozen configuration, per season
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.core import Warehouse
from cfb.model import config as C
from cfb.model.matrix import feature_columns, game_matrix, groups_by_name, stored_features
from cfb.model.models import LogitModel, MarginModel, log_loss

SEASONS = range(2018, 2026)
LAST_SEASON = 2025
PRIORITY = {"consensus": 0, "Bovada": 1, "DraftKings": 2, "ESPN Bet": 3}


def eval_ids(wh: Warehouse) -> set:
    """Game ids of the baseline set: FBS vs FBS, completed, with a spread and both Elos."""
    ln = wh.prepared("lines").frame
    ids = []
    for gid, lines in zip(ln["id"], ln["lines"]):
        if any(x.get("spread") is not None for x in json.loads(lines)):
            ids.append(gid)
    pg = wh.prepared("game_pregame").frame
    pg = pg[pg["home_pregame_elo"].notna() & pg["away_pregame_elo"].notna()]
    return set(ids) & set(pg["game_id"])


def load(data_dir="data"):
    wh = Warehouse(data_dir)
    gi = wh.game_index
    gi = gi[gi["season"] <= LAST_SEASON].reset_index(drop=True)   # 2026 never enters selection
    groups = [g for g in groups_by_name() if (Path(data_dir) / "features" / f"{g.name}.parquet").exists()]
    missing = sorted({g.name for g in groups_by_name()} - {g.name for g in groups})
    if missing:
        print(f"WARNING: no stored features for {missing}", flush=True)
    feats = stored_features(data_dir, groups)
    X = game_matrix(gi, feats, groups)
    X["completed"] = gi["completed"].to_numpy()
    X["margin"] = (gi["home_points"] - gi["away_points"]).to_numpy(dtype="float64")
    X["y"] = (X["margin"] > 0).astype("float64")
    X.loc[~X["completed"] | (X["margin"] == 0), "y"] = np.nan
    X["in_eval"] = X["game_id"].isin(eval_ids(wh)) & X["completed"] & (X["fcs_game"] == 0)
    return X, groups


def season_forward(X, make, train_filter=None, train_start=2015, seasons=SEASONS):
    """Returns (pooled log loss on the eval set, per-season dict, predictions Series)."""
    preds = []
    for s in seasons:
        tr = X[(X["season"] >= train_start) & (X["season"] < s) & X["y"].notna()]
        if train_filter is not None:
            tr = tr[train_filter(tr)]
        te = X[(X["season"] == s) & X["in_eval"] & X["y"].notna()]
        m = make().fit(tr, tr["y"].to_numpy(), tr["margin"].to_numpy())
        preds.append(pd.Series(m.predict_proba(te), index=te.index))
    p = pd.concat(preds)
    d = X.loc[p.index]
    per = {s: log_loss(d.loc[d.season == s, "y"], p[d.season == s]) for s in seasons}
    return log_loss(d["y"], p), per, p


def fbs_only(tr):
    return tr["fcs_game"] == 0


ELO_PER_SEASON = {2018: 0.5221, 2019: 0.5191, 2020: 0.5532, 2021: 0.5553, 2022: 0.5862,
                  2023: 0.5499, 2024: 0.5669, 2025: 0.5427}   # docs/eval.md (baseline_elo)


def report(name, res, t0=None):
    ll, per, p = res
    wins = sum(per[s] < ELO_PER_SEASON[s] for s in per)
    extra = f" ({time.time() - t0:.0f}s)" if t0 else ""
    print(f"{name:<44} {ll:.4f}  beats Elo {wins}/8  " + " ".join(f"{per[s]:.3f}" for s in sorted(per)) + extra,
          flush=True)


def cmd_ablation(X, groups, args):
    names = [g.name for g in groups]
    base = [n for n in C.CANDIDATE_GROUPS if n in names]
    def cols_for(gs):
        return ["site", "fcs_game"] + feature_columns(groups_by_name(gs), C.FORMS)
    mk = lambda gs: (lambda: LogitModel(cols_for(gs), lam=args.lam))
    print(f"{'config':<44} {'pooled':>6}  per season 2018..2025")
    report("site only", season_forward(X, lambda: LogitModel(["site"], lam=args.lam)))
    report("str_power only", season_forward(X, mk(["str_power"])))
    full = season_forward(X, mk(base))
    report(f"all candidate groups ({len(base)})", full)
    for g in base:
        report(f"  minus {g}", season_forward(X, mk([x for x in base if x != g])))
    for g in base:
        if g != "str_power":
            report(f"  str_power + {g}", season_forward(X, mk(["str_power", g])))


def cmd_tune(X, groups, args):
    gs = C.MODEL_GROUPS
    cols = ["site", "fcs_game"] + feature_columns(groups_by_name(gs), C.FORMS) + C.EXTRA_COLUMNS
    print(f"groups: {gs}; {len(cols)} columns")
    for lam in (0.001, 0.003, 0.01, 0.03, 0.1):
        report(f"logit lam={lam}", season_forward(X, lambda: LogitModel(cols, lam=lam)))
    for hl in (2, 4, 8):
        report(f"logit lam={C.LAM} era half-life {hl}", season_forward(X, lambda: LogitModel(cols, lam=C.LAM, era_half_life=hl)))
    for start in (2014, 2016):
        report(f"logit lam={C.LAM} train from {start}", season_forward(X, lambda: LogitModel(cols, lam=C.LAM), train_start=start))
    report(f"logit lam={C.LAM} FBS vs FBS training only", season_forward(X, lambda: LogitModel(cols, lam=C.LAM), train_filter=fbs_only))
    report(f"logit lam={C.LAM} no missing indicators", season_forward(X, lambda: LogitModel(cols, lam=C.LAM, indicators=None)))
    for lam in (0.003, 0.01, 0.03):
        report(f"margin lam={lam}", season_forward(X, lambda: MarginModel(cols, lam=lam)))


def cmd_gbt(X, groups, args):
    from sklearn.ensemble import HistGradientBoostingClassifier

    gs = C.MODEL_GROUPS
    cols = ["site", "fcs_game"] + feature_columns(groups_by_name(gs), ("h", "a", "d")) + C.EXTRA_COLUMNS

    class GBT:
        def __init__(self, **kw):
            self.m = HistGradientBoostingClassifier(**kw)
        def fit(self, X, y, margin=None):
            self.cols = [c for c in cols if X[c].nunique() > 1]
            self.m.fit(X[self.cols].to_numpy(float), y)
            return self
        def predict_proba(self, X):
            return self.m.predict_proba(X[self.cols].to_numpy(float))[:, 1]

    for lr, it, leaves in ((0.03, 300, 8), (0.03, 600, 8), (0.02, 500, 15)):
        t0 = time.time()
        report(f"gbt lr={lr} iter={it} leaves={leaves}",
               season_forward(X, lambda: GBT(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves,
                                             min_samples_leaf=40, l2_regularization=1.0, random_state=0)), t0)


def cmd_final(X, groups, args):
    make = lambda: C.make_model()
    res = season_forward(X, make, train_start=C.TRAIN_START)
    report(f"FINAL {C.MODEL_ID}", res)
    d = X.loc[res[2].index]
    p = res[2]
    acc = float(((p > 0.5) == (d["y"] > 0.5)).mean())
    brier = float(((p - d["y"]) ** 2).mean())
    print(f"n={len(p)} log loss {res[0]:.4f} brier {brier:.4f} accuracy {acc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ablation", "tune", "gbt", "final"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--lam", type=float, default=C.LAM)
    args = ap.parse_args()
    t = time.time()
    X, groups = load(args.data_dir)
    print(f"matrix {X.shape}, eval games {int(X['in_eval'].sum())} (2018-2025: "
          f"{int((X['in_eval'] & (X['season'] >= 2018)).sum())}), {time.time() - t:.0f}s", flush=True)
    {"ablation": cmd_ablation, "tune": cmd_tune, "gbt": cmd_gbt, "final": cmd_final}[args.cmd](X, groups, args)


if __name__ == "__main__":
    sys.exit(main())

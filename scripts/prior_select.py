"""Selection record for model_v2 (P2, early-season priors). Seasons through 2025 only.

    python scripts/prior_select.py final      # model_v2 vs model_v1 from the stored features
    python scripts/prior_select.py search     # the prior experiments (needs the analysis
                                              # inputs of scripts/prior_harness.py; see docs/priors.md)

`final` scores both frozen configurations season-forward (fit 2015..S-1, score S,
S = 2018..2025) on the 6,024 baseline games and prints pooled, per-season and per-week-bucket
log loss, with paired differences (bootstrap over whole weeks).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import prior_harness as H  # noqa: E402
from model_select import load, season_forward  # noqa: E402

from cfb.model import config as C


def cmd_final(args):
    X, _ = load(args.data_dir)
    early = (X["season_type"] == "regular") & (X["week"] <= 4)
    res = {}
    for mid in (C.MODEL_ID, C.MODEL_V2_ID):
        res[mid] = season_forward(X, lambda: C.make_model(model_id=mid), train_start=C.TRAIN_START)
    v1, v2 = res[C.MODEL_ID], res[C.MODEL_V2_ID]
    print(f"{'model':<10} {'pooled':>7}  " + " ".join(f"{s:>7}" for s in sorted(v1[1])))
    for mid, r in res.items():
        print(f"{mid:<10} {r[0]:7.4f}  " + " ".join(f"{r[1][s]:7.4f}" for s in sorted(r[1])))
    better = sum(v2[1][s] < v1[1][s] for s in v1[1])
    d, se = H.paired(X, v2[2], v1[2])
    de, see = H.paired(X, v2[2], v1[2], early)
    print(f"v2 - v1: all {d:+.4f} (SE {se:.4f}), weeks 1-4 {de:+.4f} (SE {see:.4f}), better in {better}/8 seasons")
    b1, b2 = H.buckets(X, v1[2]), H.buckets(X, v2[2])
    print("bucket   n     v1      v2")
    d1 = X.loc[v1[2].index]
    wk = np.where(d1["season_type"] == "regular", d1["week"], 99)
    counts = {"w1": (wk == 1).sum(), "w2": (wk == 2).sum(), "w3-4": ((wk >= 3) & (wk <= 4)).sum(),
              "w5-8": ((wk >= 5) & (wk <= 8)).sum(), "w9+": ((wk >= 9) & (wk < 99)).sum(), "post": (wk == 99).sum(),
              "w1-4": (wk <= 4).sum(), "all": len(wk)}
    for k in ("w1", "w2", "w3-4", "w1-4", "w5-8", "w9+", "post", "all"):
        print(f"{k:<6} {counts[k]:>5} {b1[k]:.4f}  {b2[k]:.4f}")
    p, y = v2[2].to_numpy(), X.loc[v2[2].index, "y"].to_numpy()
    print(f"model_v2 brier {np.mean((p - y) ** 2):.4f} accuracy {np.mean((p > 0.5) == (y > 0.5)):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["final"])
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    {"final": cmd_final}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

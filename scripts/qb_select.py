"""Ablation of the quarterback group against model_v1 (seasons through 2025 only).

Season-forward (fit 2015..S-1, score S = 2018..2025) on the baseline set of 6,024 FBS vs
FBS games, the frozen model_v1 family (margin ridge, lam 0.03), with str_qb columns added
as home-minus-away differences. Paired bootstrap over whole weeks against model_v1.

    python scripts/qb_select.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_select import load, report, season_forward  # noqa: E402

from cfb.eval.bootstrap import paired_bootstrap  # noqa: E402
from cfb.model import config as C  # noqa: E402
from cfb.model.matrix import feature_columns, groups_by_name  # noqa: E402
from cfb.model.models import MarginModel  # noqa: E402

QB_ALL = ["str_qb_q", "str_qb_mix_q", "str_qb_diff", "str_qb_plays", "str_qb_share", "str_qb_prior",
          "str_qb_from_roster"]
VARIANTS = {
    "+ str_qb (all 7)": QB_ALL,
    "+ diff": ["str_qb_diff"],
    "+ q": ["str_qb_q"],
    "+ q, mix": ["str_qb_q", "str_qb_mix_q"],
    "+ diff, share": ["str_qb_diff", "str_qb_share"],
    "+ diff, plays": ["str_qb_diff", "str_qb_plays"],
    "+ diff, from_roster": ["str_qb_diff", "str_qb_from_roster"],
}


def main():
    X, groups = load()
    base_cols = ["site", "fcs_game"] + feature_columns(groups_by_name(C.MODEL_GROUPS), C.FORMS)
    have = set(X.columns)

    def run(extra):
        cols = base_cols + [f"d_{c}" if f"d_{c}" in have else c for c in extra]
        return season_forward(X, lambda: MarginModel(cols, lam=C.LAM), train_start=C.TRAIN_START)

    base = run([])
    report("model_v1 (str_power + str_elo)", base)
    d = X.loc[base[2].index]
    clusters = d["season"].astype(str) + "_" + d["season_type"].astype(str) + "_" + d["week"].astype(str)
    for name, extra in VARIANTS.items():
        res = run(extra)
        report(name, res)
        b = paired_bootstrap(d["y"].to_numpy(), res[2].loc[d.index].to_numpy(), base[2].to_numpy(),
                             clusters=clusters.to_numpy())
        better = sum(res[1][s] < base[1][s] for s in base[1])
        print(f"    vs model_v1: diff {b['diff']:+.4f} (SE {b['se']:.4f}), better in {better}/8 seasons")


if __name__ == "__main__":
    main()

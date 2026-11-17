"""Evaluation CLI.

    python -m cfb.eval baselines [--data-dir data] [--seasons 2018-2026] [--score-seasons 2018-2025]
        Writes the four baselines to data/predictions/<model_id>/, their metrics to
        data/eval/<model_id>.json, and prints the reference table: season-forward log loss,
        Brier and accuracy on the FBS vs FBS games that have a closing spread and Elo.

    python -m cfb.eval score <model_id> [--seasons 2018-2025] [--scope fbs_vs_fbs|all]
                             [--common-with M ...] [--compare M ...] [--cluster-weeks] [--no-write]
        Scores stored predictions, writes data/eval/<model_id>.json and prints the summary.
        --common-with keeps only games every named model also predicted.
        --compare runs a paired bootstrap against each named model on common games.

The default scoring seasons stop at 2025: 2026 is the test season and is scored only when
asked for (--seasons 2026).
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from ..core.asof import Warehouse
from ..core.schemas import read_predictions, write_metrics
from ..features.build import open_warehouse, parse_seasons
from .baselines import BASELINES, DEFAULT_BASELINE_SEASONS, run_baseline
from .bootstrap import common_games, paired_bootstrap
from .metrics import evaluate, score_block
from .splits import DEFAULT_SEASONS, write_weekly

REFERENCE = ("baseline_close", "baseline_elo", "baseline_home")


def _fmt(block: dict) -> str:
    if not block["n"]:
        return "n=0"
    return f"n={block['n']:5d}  log_loss={block['log_loss']:.4f}  brier={block['brier']:.4f}  acc={block['accuracy']:.4f}"


def _restrict(preds: pd.DataFrame, seasons) -> pd.DataFrame:
    return preds[preds["season"].isin(seasons)] if seasons else preds


def reference_table(preds: dict[str, pd.DataFrame], outcomes: pd.DataFrame, seasons) -> list[str]:
    lines = []
    ref = {k: _restrict(preds[k], seasons) for k in REFERENCE}
    g = common_games(ref, outcomes)
    lines.append(f"Season-forward, FBS vs FBS with closing spread and Elo, seasons {min(seasons)}-{max(seasons)}:")
    for k in REFERENCE:
        lines.append(f"  {k:20s} {_fmt(score_block(g['y'], g['p_' + k], False))}")
    lines.append("  by season (log loss): " + "  ".join(f"{k.split('_')[1]}" for k in REFERENCE))
    for s, d in g.groupby("season"):
        cells = "  ".join(f"{score_block(d['y'], d['p_' + k], False)['log_loss']:.4f}" for k in REFERENCE)
        lines.append(f"    {s}: n={len(d):4d}  {cells}")
    ml = common_games({**ref, "baseline_moneyline": _restrict(preds["baseline_moneyline"], seasons)}, outcomes)
    if len(ml):
        lines.append(f"Same games with a moneyline ({ml['season'].min()}-{ml['season'].max()}):")
        for k in ("baseline_moneyline",) + REFERENCE:
            lines.append(f"  {k:20s} {_fmt(score_block(ml['y'], ml['p_' + k], False))}")
    return lines


def cmd_baselines(args) -> int:
    wh = open_warehouse(args.data_dir)
    seasons = parse_seasons(args.seasons) or list(DEFAULT_BASELINE_SEASONS)
    score_seasons = parse_seasons(args.score_seasons) or list(DEFAULT_SEASONS)
    outcomes = wh.outcomes()
    preds = {}
    for model_id in BASELINES:
        p = run_baseline(wh, model_id, seasons)
        paths = write_weekly(p, args.data_dir, replace=True)
        preds[model_id] = p
        doc = evaluate(_restrict(p, score_seasons), outcomes, model_id)
        write_metrics(doc, args.data_dir)
        print(f"{model_id:20s} {len(p):6d} predictions in {len(paths)} files; "
              f"FBS vs FBS {min(score_seasons)}-{max(score_seasons)}: {_fmt(doc['overall'])}")
    print()
    print("\n".join(reference_table(preds, outcomes, score_seasons)))
    return 0


def cmd_score(args) -> int:
    wh = Warehouse(args.data_dir)
    outcomes = wh.outcomes()
    seasons = parse_seasons(args.seasons) or list(DEFAULT_SEASONS)
    pred = _restrict(read_predictions(args.data_dir, args.model_id), seasons)
    if pred.empty:
        print(f"no predictions for {args.model_id} in seasons {seasons} under {args.data_dir}/predictions",
              file=sys.stderr)
        return 1
    label = args.scope
    if args.common_with:
        others = {m: read_predictions(args.data_dir, m) for m in args.common_with}
        keep = set(pred["game_id"])
        for m, p in others.items():
            keep &= set(p["game_id"])
        pred = pred[pred["game_id"].isin(keep)]
        label = f"{args.scope}; games also predicted by {', '.join(args.common_with)}"
    doc = evaluate(pred, outcomes, args.model_id, scope=args.scope, scope_label=label)
    if not args.no_write:
        print(write_metrics(doc, args.data_dir))
    print(f"{args.model_id}  scope: {doc['scope']}  seasons {min(seasons)}-{max(seasons)}")
    print(f"  overall  {_fmt(doc['overall'])}")
    for row in doc["by_season"]:
        print(f"  {row['season']}     {_fmt(row)}")
    for other in args.compare or []:
        g = common_games({"a": pred, "b": _restrict(read_predictions(args.data_dir, other), seasons)}, outcomes,
                         fbs_only=args.scope == "fbs_vs_fbs")
        if g.empty:
            print(f"  vs {other}: no common games")
            continue
        clusters = (g["season"].astype(str) + g["season_type"] + g["week"].astype(str)) if args.cluster_weeks else None
        for metric in ("log_loss", "brier", "accuracy"):
            r = paired_bootstrap(g["y"], g["p_a"], g["p_b"], metric, n_boot=args.n_boot, clusters=clusters)
            print(f"  vs {other:20s} {metric:9s} n={r['n']}  {r['a']:.4f} - {r['b']:.4f} = {r['diff']:+.4f}  "
                  f"se {r['se']:.4f}  95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
                  f"P({args.model_id} better) {r['p_a_better']:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cfb.eval", description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines", help="write and score the four baselines")
    b.add_argument("--seasons", help="seasons to predict, default 2018-2026")
    b.add_argument("--score-seasons", help="seasons in the metrics files, default 2018-2025")
    s = sub.add_parser("score", help="score stored predictions of one model")
    s.add_argument("model_id")
    s.add_argument("--seasons", help="default 2018-2025; pass 2026 to score the test season")
    s.add_argument("--scope", default="fbs_vs_fbs", choices=("fbs_vs_fbs", "all"))
    s.add_argument("--common-with", nargs="+", metavar="MODEL")
    s.add_argument("--compare", nargs="+", metavar="MODEL")
    s.add_argument("--cluster-weeks", action="store_true", help="bootstrap whole weeks instead of games")
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--no-write", action="store_true")
    for p in (b, s):
        p.add_argument("--data-dir", default=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    return cmd_baselines(args) if args.cmd == "baselines" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())

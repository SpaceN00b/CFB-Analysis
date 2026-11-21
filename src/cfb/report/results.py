"""A week's published picks next to the final scores.

    graded = grade(bundle, results)          # bundle = that week's stored bundle.json
    print(render_results(graded))

The picks are the ones the weekly run stored before the games (the week's `bundle.json`:
win probability per model, the pick, and the spread and total picks against the line at that
run). Nothing is recomputed, so later data, refits or a model switch do not change them.
Results come from the warehouse's completed games (`results_from_outcomes`) or from a later
week's bundle, whose last-week section holds the scores (`results_from_bundles`).

Command line (reads data/reports/<key>/bundle.json and data/ for the scores):

    python -m cfb.results --season 2026 --week 12
    python -m cfb.results --season 2026 --week 12 --out week12_results.md
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from ..eval.metrics import score_block

Score = tuple[float, float]  # (home_points, away_points)


# ---- results --------------------------------------------------------------------------------

def results_from_outcomes(outcomes: pd.DataFrame) -> dict[int, Score]:
    """game_id -> (home_points, away_points) for completed games (`Warehouse.outcomes()`)."""
    done = outcomes[outcomes["home_points"].notna() & outcomes["away_points"].notna()]
    return {int(g): (float(h), float(a))
            for g, h, a in zip(done["game_id"], done["home_points"], done["away_points"])}


def results_from_bundles(bundles) -> dict[int, Score]:
    """game_id -> score from the last-week sections of later bundles."""
    out: dict[int, Score] = {}
    for b in bundles:
        for g in (b.get("previous") or {}).get("games", []):
            r = g.get("result") or {}
            if r.get("home_points") is not None and r.get("away_points") is not None:
                out[int(g["game_id"])] = (float(r["home_points"]), float(r["away_points"]))
    return out


# ---- grading --------------------------------------------------------------------------------

def _wlp(diff: float | None, picked_positive: bool) -> str | None:
    """'W', 'L' or 'P' (push) for a pick against a line; None without a pick or a score."""
    if diff is None:
        return None
    if diff == 0:
        return "P"
    return "W" if (diff > 0) == picked_positive else "L"


def _round(x, digits=4):
    return None if x is None else round(float(x), digits)


def grade_game(g: dict, score: Score | None, primary: str) -> dict:
    """One stored game row with its final score and whether each stored pick was right."""
    pick, market = g.get("pick") or {}, g.get("market") or {}
    ats, ou = g.get("ats") or {}, g.get("ou") or {}
    p_home = (g.get("p") or {}).get(primary)
    row = {
        "game_id": int(g["game_id"]), "kickoff": g.get("kickoff_local"), "kickoff_utc": g.get("kickoff_utc"),
        "away": g["away"]["name"], "home": g["home"]["name"], "neutral": bool(g.get("neutral")),
        "fbs_vs_fbs": bool(g.get("fbs_vs_fbs")),
        "conferences": sorted({c for c in (g["home"].get("conference"), g["away"].get("conference")) if c}),
        "p": dict(g.get("p") or {}), "pick": pick.get("team"), "pick_side": pick.get("side"),
        "p_pick": pick.get("p"),
        "market_p_pick": (None if market.get("p_home") is None or not pick
                          else (market["p_home"] if pick["side"] == "home" else 1 - market["p_home"])),
        "model_spread": ats.get("model_line"), "home_line": market.get("spread"), "spread_line": ats.get("line"),
        "ats_pick": ats.get("team"), "pred_total": g.get("pred_total"),
        "total_line": ou.get("line", market.get("total")), "ou_pick": ou.get("side"),
        "home_points": None, "away_points": None, "final": False, "winner": None,
        "pick_correct": None, "ats_result": None, "ou_result": None, "log_loss": None,
    }
    if score is None or score[0] == score[1]:
        return row
    hp, ap = score
    home_won = hp > ap
    row.update(home_points=hp, away_points=ap, final=True, winner=row["home"] if home_won else row["away"])
    if pick:
        row["pick_correct"] = pick["side"] == ("home" if home_won else "away")
    if ats.get("side") and ats.get("line") is not None:
        picked_margin = hp - ap if ats["side"] == "home" else ap - hp   # the line is the picked team's
        row["ats_result"] = _wlp(picked_margin + ats["line"], True)
    if ou.get("side") and ou.get("line") is not None:
        row["ou_result"] = _wlp(hp + ap - ou["line"], ou["side"] == "over")
    if p_home is not None:
        p = min(max(p_home if home_won else 1 - p_home, 1e-15), 1.0)
        row["log_loss"] = round(-math.log(p), 4)
    return row


def _record(rows: list[dict], key: str) -> dict | None:
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None
    w, l, p = (sum(v == x for v in vals) for x in "WLP")
    return {"won": w, "lost": l, "push": p, "n": len(vals), "hit_rate": _round(w / (w + l)) if w + l else None}


def grade(bundle: dict, results: dict[int, Score]) -> dict:
    """The week's stored picks graded against `results` (game_id -> score).

    The summary is FBS vs FBS (the report's headline scope); `all` covers every graded game."""
    meta = bundle["meta"]
    primary = meta["primary"]
    rows = [grade_game(g, results.get(int(g["game_id"])), primary) for g in bundle.get("games", [])]
    final = [r for r in rows if r["final"]]
    fbs = [r for r in final if r["fbs_vs_fbs"]]
    labels = {m["id"]: m.get("label", m["id"]) for m in bundle.get("models", [])}

    def models_block(scope: list[dict]) -> dict:
        out = {}
        for m in labels:
            pairs = [(1.0 if r["home_points"] > r["away_points"] else 0.0, r["p"][m])
                     for r in scope if r["p"].get(m) is not None]
            if not pairs:
                continue
            y, p = zip(*pairs)
            b = score_block(y, p, with_calibration=False)
            out[m] = {"label": labels[m], "primary": m == primary, "n": b["n"],
                      "correct": int(sum((pp >= 0.5) == (yy == 1) for yy, pp in pairs)),
                      **{k: _round(b[k]) for k in ("accuracy", "log_loss", "brier")}}
        return out

    picks = [r for r in fbs if r["pick_correct"] is not None]
    return {
        "key": meta.get("key"), "label": meta.get("label"), "primary": primary,
        "primary_label": meta.get("primary_label", labels.get(primary, primary)),
        "created_local": meta.get("created_local"), "lines_asof_local": meta.get("lines_asof_local"),
        "phase2": bool(meta.get("phase2")),
        "counts": {"games": len(rows), "final": len(final), "pending": len(rows) - len(final),
                   "fbs_vs_fbs_final": len(fbs)},
        "picks": {"n": len(picks), "correct": sum(r["pick_correct"] for r in picks)},
        "models": models_block(fbs),
        "all_games": models_block(final),
        "ats": _record(fbs, "ats_result"), "ou": _record(fbs, "ou_result"),
        "games": rows,
    }


def frame(graded: dict) -> pd.DataFrame:
    """One row per game (kickoff order) for tables."""
    cols = ["game_id", "kickoff", "away", "home", "fbs_vs_fbs", "conferences", "pick", "p_pick", "market_p_pick",
            "score", "pick_correct", "log_loss", "model_spread", "home_line", "spread_line", "ats_pick", "ats_result",
            "pred_total", "total_line", "ou_pick", "ou_result", "final", "kickoff_utc"]
    rows = []
    for r in graded["games"]:
        score = f"{r['away']} {r['away_points']:.0f}, {r['home']} {r['home_points']:.0f}" if r["final"] else None
        rows.append({**r, "score": score})
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)


# ---- Markdown -------------------------------------------------------------------------------

def _pct(p) -> str:
    return "" if p is None else f"{100 * p:.0f}%"


def _num(x, fmt=".4f") -> str:
    return "" if x is None else format(x, fmt)


def _rec(r: dict | None) -> str:
    if not r:
        return "none graded"
    return f"{r['won']}-{r['lost']}-{r['push']} ({_pct(r['hit_rate'])} of decided)"


def _md_table(head: list[str], rows: list[list], align: str) -> list[str]:
    sep = {"l": ":---", "r": "---:"}
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join(sep[a] for a in align) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def render_results(graded: dict) -> str:
    c = graded["counts"]
    out = [f"# {graded['label']}: published picks vs results", "",
           f"Picks as stored by the weekly run of {graded.get('created_local') or '?'} (lines as of "
           f"{graded.get('lines_asof_local') or '?'}); primary model {graded['primary_label']}. "
           f"Final: {c['final']} of {c['games']} games ({c['pending']} pending).", ""]
    if graded["models"]:
        out += ["FBS vs FBS games with a final score:", ""]
        rows = [[v["label"] + (" (primary)" if v["primary"] else ""), v["n"], f"{v['correct']}/{v['n']}",
                 _pct(v["accuracy"]), _num(v["log_loss"]), _num(v["brier"])] for v in graded["models"].values()]
        out += _md_table(["Model", "Games", "Correct", "Accuracy", "Log loss", "Brier"], rows, "lrrrrr") + [""]
        if graded["phase2"]:
            out += [f"Against the line at the run: spread {_rec(graded['ats'])}, total {_rec(graded['ou'])} "
                    "(won-lost-push).", ""]
    rows = []
    for r in sorted(graded["games"], key=lambda r: (r["kickoff_utc"] or "", r["game_id"])):
        score = f"{r['away']} {r['away_points']:.0f}, {r['home']} {r['home_points']:.0f}" if r["final"] else "pending"
        mark = "" if r["pick_correct"] is None else ("yes" if r["pick_correct"] else "no")
        row = [r["kickoff"] or "", f"{r['away']} at {r['home']}",
               f"{r['pick']} {_pct(r['p_pick'])}" if r["pick"] else "", score, mark, _num(r["log_loss"], ".3f")]
        if graded["phase2"]:
            row += [f"{r['ats_pick']} ({_num(r['spread_line'], '+.1f')})" if r["ats_pick"] else "",
                    r["ats_result"] or "", f"{r['ou_pick']} {_num(r['total_line'], '.1f')}" if r["ou_pick"] else "",
                    r["ou_result"] or ""]
        rows.append(row)
    head = ["Kickoff", "Game", "Pick", "Final", "Right", "Log loss"]
    align = "lllllr"
    if graded["phase2"]:
        head += ["Spread pick (its line)", "ATS", "Total pick", "O/U"]
        align += "llll"
    out += _md_table(head, rows, align)
    return "\n".join(out) + "\n"


# ---- command line ---------------------------------------------------------------------------

def bundle_path(reports_dir: Path, key: str) -> Path:
    return Path(reports_dir) / key / "bundle.json"


def later_bundles(reports_dir: Path, key: str) -> list[dict]:
    """Stored bundles of weeks after `key` (their last-week sections hold scores)."""
    import json

    from ..ui.data import report_sort_key  # plain helper, no Streamlit

    k, root = report_sort_key(key), Path(reports_dir)
    if not root.exists():
        return []
    folders = [f for f in root.iterdir() if (f / "bundle.json").exists() and report_sort_key(f.name) > k]
    return [json.loads((f / "bundle.json").read_text(encoding="utf-8")) for f in folders]


def collect_results(data_dir: Path | None, reports_dir: Path, key: str, log=print) -> dict[int, Score]:
    """Scores from later bundles, then the warehouse (which wins where both have a game)."""
    results = results_from_bundles(later_bundles(reports_dir, key))
    if data_dir is not None:
        try:
            from ..core.asof import Warehouse
            results.update(results_from_outcomes(Warehouse(data_dir).outcomes()))
        except Exception as exc:  # no core tables: later bundles only
            log(f"warehouse scores unavailable ({exc}); using later reports only")
    return results


def main(argv: list[str] | None = None) -> int:
    import json

    ap = argparse.ArgumentParser(prog="python -m cfb.results", description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season-type", default="regular", choices=["regular", "postseason"])
    ap.add_argument("--data", default="data", help="warehouse folder with core tables (default data)")
    ap.add_argument("--reports", default=None, help="folder of report folders (default <data>/reports)")
    ap.add_argument("--out", default=None, help="write Markdown here instead of printing it")
    ap.add_argument("--json", default=None, help="also write the graded rows as JSON here")
    args = ap.parse_args(argv)
    key = f"{args.season}_{args.season_type}_{args.week}"
    reports = Path(args.reports) if args.reports else Path(args.data) / "reports"
    path = bundle_path(reports, key)
    if not path.exists():
        print(f"no stored report for {key} at {path}", file=sys.stderr)
        return 2
    bundle = json.loads(path.read_text(encoding="utf-8"))
    log = lambda m: print(m, file=sys.stderr)  # noqa: E731
    graded = grade(bundle, collect_results(Path(args.data), reports, key, log=log))
    text = render_results(graded)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.json:
        Path(args.json).write_text(json.dumps(graded, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


__all__ = ["collect_results", "frame", "grade", "grade_game", "render_results", "results_from_bundles",
           "results_from_outcomes"]

"""Scores for point predictions (phase 2): the home margin against the closing spread and
the game total against the closing total.

    doc = evaluate_points(predictions, outcomes, lines)

`predictions` needs `game_id`, `pred_margin`, `pred_total` (either may be empty).
`outcomes` are game_index rows with results; `lines` holds `game_id`, `mkt_spread_close`,
`mkt_total_close` (home-oriented: a spread of -7 means the home team gives 7).

For each target the document has, on the games with a result, a prediction and a closing
line: MAE, RMSE and bias (mean of actual minus predicted) of the model and the line's MAE
on the same games; and the hit rate against the line: the model's side (home cover when
pred_margin > -spread, over when pred_total > total) against the result, pushes and games
where the model equals the line left out. Hit rates are given for every game and for games
where the model is at least 3 and 7 points away from the line. At -110 prices a side
needs 52.4% to break even.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLDS = (0.0, 3.0, 7.0)
BREAK_EVEN_110 = 110 / 210
_R = 4

TARGETS = {
    "margin": ("pred_margin", "mkt_spread_close"),
    "total": ("pred_total", "mkt_total_close"),
}


def points_frame(predictions: pd.DataFrame, outcomes: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Predictions joined to results and closing lines. One row per game."""
    pred = predictions[[c for c in ("game_id", "pred_margin", "pred_total") if c in predictions.columns]]
    if pred["game_id"].duplicated().any():
        raise ValueError("predictions hold more than one row per game")
    cols = ["game_id", "season", "season_type", "week", "home_fbs", "away_fbs", "home_points", "away_points"]
    out = outcomes[[c for c in cols if c in outcomes.columns]]
    out = out[out["home_points"].notna() & out["away_points"].notna()]
    g = pred.merge(out, on="game_id", how="inner")
    ln = lines[[c for c in ("game_id", "mkt_spread_close", "mkt_total_close") if c in lines.columns]]
    g = g.merge(ln, on="game_id", how="left")
    g["margin"] = (g["home_points"] - g["away_points"]).astype("float64")
    g["total"] = (g["home_points"] + g["away_points"]).astype("float64")
    g["fbs_vs_fbs"] = g["home_fbs"].astype(bool) & g["away_fbs"].astype(bool)
    return g.reset_index(drop=True)


def line_prediction(g: pd.DataFrame, target: str) -> np.ndarray:
    """The closing line as a prediction of the target (minus the spread for the margin)."""
    col = TARGETS[target][1]
    v = g[col].to_numpy(dtype="float64") if col in g else np.full(len(g), np.nan)
    return -v if target == "margin" else v


def hit_rate(pred, line, actual, threshold: float = 0.0) -> tuple[float | None, int]:
    """Share of games where the model's side of the line won; pushes and zero edges left out."""
    pred, line, actual = (np.asarray(x, float) for x in (pred, line, actual))
    edge, res = pred - line, actual - line
    k = np.isfinite(edge) & np.isfinite(res) & (res != 0) & (np.abs(edge) >= max(threshold, 1e-9))
    if not k.any():
        return None, 0
    return float(np.mean(np.sign(edge[k]) == np.sign(res[k]))), int(k.sum())


def block(g: pd.DataFrame, target: str) -> dict:
    pcol = TARGETS[target][0]
    line = line_prediction(g, target)
    k = g[pcol].notna().to_numpy() & np.isfinite(line)
    y, p, ln = g[target].to_numpy(float)[k], g[pcol].to_numpy(float)[k], line[k]
    out = {"n": int(k.sum()), "mae": None, "rmse": None, "bias": None, "line_mae": None, "line_rmse": None}
    if k.any():
        out.update(mae=round(float(np.mean(np.abs(y - p))), _R), rmse=round(float(np.sqrt(np.mean((y - p) ** 2))), _R),
                   bias=round(float(np.mean(y - p)), _R), line_mae=round(float(np.mean(np.abs(y - ln))), _R),
                   line_rmse=round(float(np.sqrt(np.mean((y - ln) ** 2))), _R))
    hits = []
    for t in THRESHOLDS:
        rate, n = hit_rate(p, ln, y, t)
        hits.append({"min_edge": t, "n": n, "hit_rate": None if rate is None else round(rate, _R)})
    out["vs_line"] = hits
    return out


def evaluate_points(predictions: pd.DataFrame, outcomes: pd.DataFrame, lines: pd.DataFrame,
                    model_id: str = "", scope: str = "fbs_vs_fbs") -> dict:
    g = points_frame(predictions, outcomes, lines)
    s = g[g["fbs_vs_fbs"]] if scope == "fbs_vs_fbs" else g
    doc = {"model_id": model_id, "scope": scope, "break_even_110": round(BREAK_EVEN_110, _R)}
    for target in TARGETS:
        doc[target] = {
            "overall": block(s, target),
            "by_season": [{"season": int(k), **block(d, target)} for k, d in s.groupby("season", sort=True)],
        }
    return doc


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def format_points(doc: dict) -> str:
    """Plain-text summary of an evaluate_points document."""
    lines = [f"{doc['model_id']} ({doc['scope']}), break-even at -110: {100 * doc['break_even_110']:.1f}%"]
    for target, name in (("margin", "Spread (home margin)"), ("total", "Total")):
        lines.append(f"{name}:")
        rows = [("all", doc[target]["overall"])] + [(str(b["season"]), b) for b in doc[target]["by_season"]]
        for label, b in rows:
            if not b["n"]:
                continue
            hits = "  ".join(f"{h['min_edge']:g}+: {_pct(h['hit_rate'])} ({h['n']})" for h in b["vs_line"])
            lines.append(f"  {label:>5} n {b['n']:>5}  MAE {b['mae']:.2f} (line {b['line_mae']:.2f})  "
                         f"RMSE {b['rmse']:.2f} (line {b['line_rmse']:.2f})  bias {b['bias']:+.2f}  vs line {hits}")
    return "\n".join(lines)


__all__ = ["BREAK_EVEN_110", "THRESHOLDS", "block", "evaluate_points", "format_points", "hit_rate",
           "line_prediction", "points_frame"]

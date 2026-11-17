"""Write the sample prediction and metrics files that show the two file contracts.

Usage:
    python -m cfb.core.samples --data-dir data --out samples --season 2025 --weeks 1,2

The sample model `sample_elo` turns CFBD pregame Elo into a home win probability with a
fixed 55-point home edge. It exists to show the format; the real baselines belong to
cfb.eval. Games without Elo for both teams are left out.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .asof import Warehouse
from .schemas import METRICS_SCHEMA_VERSION, reasons_json, write_metrics, write_predictions

SAMPLE_MODEL = "sample_elo"
HOME_EDGE_ELO = 55.0
_EPS = 1e-12


def sample_predictions(warehouse: Warehouse, season: int, season_type: str, week: int,
                       created_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    games = warehouse.schedule([season])
    games = games[(games["season_type"] == season_type) & (games["week"] == week)]
    if games.empty:
        raise ValueError(f"no games for {season} {season_type} week {week}")
    view = warehouse.view(games["cutoff_utc"].max())
    elo = view.table("game_pregame")
    g = games.merge(elo, on="game_id", how="inner").dropna(subset=["home_pregame_elo", "away_pregame_elo"])
    diff = g["home_pregame_elo"] - g["away_pregame_elo"] + np.where(g["neutral"], 0.0, HOME_EDGE_ELO)
    created = created_utc if created_utc is not None else pd.Timestamp.now(tz="UTC")
    reasons = [
        reasons_json([{"text": f"Elo gap of {d:+.0f} points including home field", "feature": "elo_diff",
                       "value": round(float(d), 1), "effect": round(float(d) * np.log(10) / 400, 4)}])
        for d in diff
    ]
    return pd.DataFrame({
        "game_id": g["game_id"].to_numpy(),
        "model_id": SAMPLE_MODEL,
        "p_home_win": (1.0 / (1.0 + 10 ** (-diff / 400.0))).round(6).to_numpy(),
        "pred_margin": np.nan,
        "pred_total": np.nan,
        "cutoff_utc": g["cutoff_utc"].array,
        "created_utc": created,
        "reasons": reasons,
    })


def _block(p: np.ndarray, y: np.ndarray, calibration: bool = True, bins: int = 10) -> dict:
    n = int(len(p))
    block = {"n": n, "log_loss": None, "brier": None, "accuracy": None}
    if n:
        q = np.clip(p, _EPS, 1 - _EPS)
        block["log_loss"] = round(float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))), 6)
        block["brier"] = round(float(np.mean((p - y) ** 2)), 6)
        block["accuracy"] = round(float(np.mean((p >= 0.5) == (y == 1))), 6)
    if calibration:
        edges = np.linspace(0, 1, bins + 1)
        which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1) if n else np.array([], dtype=int)
        block["calibration"] = [
            {"p_lo": round(float(edges[i]), 4), "p_hi": round(float(edges[i + 1]), 4), "n": int((which == i).sum()),
             "p_mean": round(float(p[which == i].mean()), 6) if (which == i).any() else None,
             "win_rate": round(float(y[which == i].mean()), 6) if (which == i).any() else None}
            for i in range(bins)
        ]
    return block


def sample_metrics(predictions: pd.DataFrame, warehouse: Warehouse, created_utc: pd.Timestamp | None = None) -> dict:
    """Metrics document for `predictions`, scored on completed FBS vs FBS games."""
    out = warehouse.outcomes()
    scored = predictions.merge(out, on="game_id", how="inner")
    scored["y"] = (scored["home_points"] > scored["away_points"]).astype("float64")
    fbs = scored[scored["home_fbs"] & scored["away_fbs"]]
    block = lambda d, cal=True: _block(d["p_home_win"].to_numpy(), d["y"].to_numpy(), cal)
    segments = {
        "fbs_vs_fbs": fbs, "conference": fbs[fbs["conference_game"]],
        "non_conference": fbs[~fbs["conference_game"]], "neutral": fbs[fbs["neutral"]],
    }
    created = created_utc if created_utc is not None else pd.Timestamp.now(tz="UTC")
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "model_id": str(predictions["model_id"].iloc[0]),
        "created_utc": created.isoformat(),
        "scope": "fbs_vs_fbs",
        "overall": block(fbs),
        "by_season": [{"season": int(s), **block(d)} for s, d in fbs.groupby("season")],
        "by_week": [
            {"season": int(s), "season_type": str(t), "week": int(w), **block(d, False)}
            for (s, t, w), d in fbs.groupby(["season", "season_type", "week"])
        ],
        "by_segment": [{"segment": name, **block(d, False)} for name, d in segments.items()],
    }


def write_samples(warehouse: Warehouse, out_dir: str | Path, season: int, weeks: list[int],
                  created_utc: pd.Timestamp | None = None) -> list[Path]:
    paths, frames = [], []
    for week in weeks:
        frame = sample_predictions(warehouse, season, "regular", week, created_utc)
        frames.append(frame)
        paths.append(write_predictions(frame, out_dir, season, "regular", week))
    paths.append(write_metrics(sample_metrics(pd.concat(frames, ignore_index=True), warehouse, created_utc), out_dir))
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="samples")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--weeks", default="1,2")
    args = ap.parse_args(argv)
    weeks = [int(w) for w in args.weeks.split(",")]
    for path in write_samples(Warehouse(args.data_dir), args.out, args.season, weeks):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

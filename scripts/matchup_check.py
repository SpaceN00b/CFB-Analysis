"""Coverage of the matchup features and their relation to the margin left after Elo and home field.

    python -m cfb.features.build --groups matchup_style,matchup_history --seasons 2014-2025
    python scripts/matchup_check.py [--data-dir data] [--out report.md]

Seasons 2014 to 2025 only; 2026 is the test season and is never read here.

Games: completed FBS vs FBS games with CFBD pregame Elo for both teams. Base model:
    home margin = a * home_field + b * (home Elo - away Elo) / 100      (OLS, no intercept)
with home_field 0 at neutral sites. The residual is what the features should explain.

For each feature the game value is home row minus away row; the report gives the slope
of the residual on that value scaled to one standard deviation (points per SD), its t
statistic, and the correlation. Variance inputs (mat_var_*) are the same on both rows,
so the report relates them to the absolute residual instead. Count columns get coverage
only. Coverage: share of the season's games where both rows have a value.

This is a single-feature check. The ablation against the strength group is W8's.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.core import Warehouse
from cfb.features.base import run_group
from cfb.features.matchup import GROUPS

LAST_SEASON = 2025
COUNT_COLUMNS = ("mat_style_games",)


def load_features(wh: Warehouse, data_dir: Path, rebuild: bool) -> pd.DataFrame:
    games = wh.schedule(range(2014, LAST_SEASON + 1))
    frames = []
    for group in GROUPS:
        path = data_dir / "features" / f"{group.name}.parquet"
        if path.exists() and not rebuild:
            frame = pd.read_parquet(path)
            frame = frame[frame["game_id"].isin(games["game_id"])]
            missing = set(group.column_names) - set(frame.columns)
            if missing:
                raise SystemExit(f"{path} lacks {sorted(missing)}; rebuild the group")
        else:
            frame = run_group(group, wh, games)
        frames.append(frame.set_index(["game_id", "team_id"]))
    return pd.concat(frames, axis=1).reset_index()


def base_residuals(wh: Warehouse) -> pd.DataFrame:
    out = wh.outcomes()
    out = out[(out["season"] <= LAST_SEASON) & out["home_fbs"] & out["away_fbs"]]
    elo = wh.view(out["cutoff_utc"].max()).table("game_pregame")
    g = out.merge(elo, on="game_id", how="inner").dropna(subset=["home_pregame_elo", "away_pregame_elo"])
    g = g.assign(margin=g["home_points"] - g["away_points"],
                 home_field=(~g["neutral"].astype(bool)).astype(float),
                 elo_diff=(g["home_pregame_elo"] - g["away_pregame_elo"]) / 100)
    x = g[["home_field", "elo_diff"]].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(x, g["margin"].to_numpy(dtype=float), rcond=None)
    g["resid"] = g["margin"] - x @ coef
    print(f"base model on {len(g)} games: home field {coef[0]:.2f} pts, {coef[1]:.2f} pts per 100 Elo, "
          f"residual SD {g['resid'].std():.2f}")
    return g[["game_id", "season", "home_id", "away_id", "resid"]]


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Slope per SD of x, its t statistic, and the correlation (with an intercept)."""
    xs = (x - x.mean()) / x.std()
    X = np.column_stack([np.ones_like(xs), xs])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / (len(y) - 2)
    se = np.sqrt(s2 / (xs @ xs))
    return float(beta[1]), float(beta[1] / se), float(np.corrcoef(xs, y)[0, 1])


def markdown(frame: pd.DataFrame) -> str:
    """A pipe table without extra dependencies."""
    fmt = lambda v: "" if isinstance(v, float) and np.isnan(v) else str(v)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def report(wh: Warehouse, feats: pd.DataFrame) -> str:
    g = base_residuals(wh)
    cols = [c for grp in GROUPS for c in grp.column_names]
    home = feats.rename(columns={"team_id": "home_id"})
    away = feats.rename(columns={"team_id": "away_id"})
    g = g.merge(home, on=["game_id", "home_id"], how="left").merge(
        away, on=["game_id", "away_id"], how="left", suffixes=("_h", "_a"))
    seasons = sorted(g["season"].unique())
    rows = []
    for c in cols:
        h, a = g[f"{c}_h"].to_numpy(dtype=float), g[f"{c}_a"].to_numpy(dtype=float)
        both = np.isfinite(h) & np.isfinite(a)
        cov = g.assign(ok=both).groupby("season")["ok"].mean().reindex(seasons)
        row = {"column": c, **{str(s): round(float(v), 2) for s, v in cov.items()}, "n": int(both.sum())}
        is_count = c in COUNT_COLUMNS or c.startswith("mat_hist_n_")
        if is_count or both.sum() < 50:
            row.update(kind="count" if is_count else "-", pts_per_sd=np.nan, t=np.nan, r=np.nan)
        elif c.startswith("mat_var_"):
            slope, t, r = ols(np.abs(g["resid"].to_numpy()[both]), h[both])
            row.update(kind="|resid|", pts_per_sd=slope, t=t, r=r)
        else:
            x = h[both] - a[both]
            if np.nanstd(x) == 0:
                row.update(kind="diff", pts_per_sd=np.nan, t=np.nan, r=np.nan)
            else:
                slope, t, r = ols(g["resid"].to_numpy()[both], x)
                row.update(kind="diff", pts_per_sd=slope, t=t, r=r)
        rows.append(row)
    table = pd.DataFrame(rows)
    rel = table[["column", "kind", "n", "pts_per_sd", "t", "r"]].round({"pts_per_sd": 2, "t": 1, "r": 3})
    cov = table[["column"] + [str(s) for s in seasons]]
    text = ["## Relation to the residual margin after Elo and home field (2014-2025)", "",
            markdown(rel), "",
            "## Coverage by season (share of FBS vs FBS games with both rows filled)", "",
            markdown(cov), ""]
    return "\n".join(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--rebuild", action="store_true", help="build the groups instead of reading data/features")
    ap.add_argument("--out", default="", help="also write the report to this markdown file")
    args = ap.parse_args(argv)
    wh = Warehouse(args.data_dir)
    feats = load_features(wh, Path(args.data_dir), args.rebuild)
    text = report(wh, feats)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

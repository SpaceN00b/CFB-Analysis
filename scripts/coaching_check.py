"""Reliability of the coaching metrics on seasons 2014 to 2025, and the numbers behind the
constants in cfb.features.coaching.metrics.

    python scripts/coaching_check.py [--data-dir data] [--seed 0]

Everything is read through one view taken on 2026-07-01, after the 2025 postseason and
before any 2026 game, so no 2026 result enters. For each metric it prints:
    * coaches with enough data (thresholds below),
    * split-half reliability: each coach's games are split at random into two halves,
      the raw metric is computed on each half, the halves are correlated across coaches
      that meet the threshold in both, and Spearman-Brown gives the full-sample figure,
    * the method-of-moments split of the spread of coach rates into true variance and
      sampling noise, which sets the shrinkage prior,
    * for the half-lives, how well an estimate through season S-1 predicts season S
      (weighted correlation across coaches, S = 2016 to 2025).
It also checks the coach-to-game map: agreement with CFBD's per-season game counts and
the first-year effect against pregame Elo (plan doc: -1.6 points).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb.core import Warehouse  # noqa: E402
from cfb.features.coaching import metrics as M  # noqa: E402
from cfb.features.coaching.mapping import coach_game_map, season_counts  # noqa: E402

VIEW_AT = "2026-07-01T00:00:00Z"
LAST_SEASON = 2025
MIN = {"ats": 25, "fourth": 100, "run": 1000, "flex": 30}  # per half for split-half


def wcorr(x, y, w=None) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 3:
        return float("nan")
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return float(cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)))


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")


def mom(rate: pd.Series, n: pd.Series, noise: pd.Series) -> tuple[float, float]:
    """(true variance, mean sampling variance) of coach rates; noise = per-unit variance."""
    w = n / n.sum()
    mean = float((rate * w).sum())
    observed = float((w * (rate - mean) ** 2).sum())
    sampling = float((w * noise / n).sum())
    return observed - sampling, sampling


def split_half(frame: pd.DataFrame, value: str, rng, min_n: int, unit: str = "game_id") -> tuple[float, int]:
    """Split-half reliability of a per-coach mean of `value`, splitting by `unit`."""
    units = frame[["coach_id", unit]].drop_duplicates()
    units = units.assign(half=rng.integers(0, 2, len(units)))
    f = frame.merge(units, on=["coach_id", unit])
    agg = f.groupby(["coach_id", "half"])[value].agg(["mean", "size"]).unstack("half")
    ok = (agg[("size", 0)] >= min_n) & (agg[("size", 1)] >= min_n)
    r = wcorr(agg.loc[ok, ("mean", 0)], agg.loc[ok, ("mean", 1)])
    return spearman_brown(r), int(ok.sum())


def predictive(frame: pd.DataFrame, half_life: float | None, prior: float) -> float:
    """Weighted correlation between a coach's shrunk estimate through S-1 and their mean
    residual in S, pooled over S = 2016..2025."""
    xs, ys, ws = [], [], []
    by_season = frame.groupby(["coach_id", "season"])["resid"].agg(["mean", "size"]).reset_index()
    for s in range(2016, LAST_SEASON + 1):
        past = frame[frame["season"] < s]
        est = M.shrunk_rate(past, ["coach_id"], s, half_life, prior)
        now = by_season[by_season["season"] == s].merge(est, on="coach_id")
        xs += list(now["value"]); ys += list(now["mean"]); ws += list(now["size"])
    return wcorr(xs, ys, ws)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    wh = Warehouse(args.data_dir)
    view = wh.view(VIEW_AT)
    assert view.table("team_game")["season"].max() <= LAST_SEASON

    cmap = coach_game_map(view)
    tg = view.table("team_game", columns=["game_id", "team_id", "opp_id", "is_home", "margin", "team_fbs", "opp_fbs"])
    hist = tg.merge(cmap.drop(columns="start_time_utc"), on=["game_id", "team_id"]).dropna(subset=["coach_id"])
    print(f"== coach-to-game map (view {VIEW_AT})")
    fbs = tg[tg["team_fbs"]].merge(cmap, on=["game_id", "team_id"], how="left")
    print(f"FBS team-games with a coach: {fbs['coach_id'].notna().mean():.4f} of {len(fbs)}")
    counts = season_counts(view)
    mine = hist.groupby(["coach_id", "team_id", "season"]).size().rename("mapped").reset_index()
    cmp_ = counts[counts["games"] > 0].merge(mine, on=["coach_id", "team_id", "season"], how="outer").fillna(0)
    cmp_ = cmp_[cmp_["season"] <= LAST_SEASON]
    print(f"coach-team-seasons where mapped games == CFBD games: {(cmp_['mapped'] == cmp_['games']).mean():.3f} "
          f"(n={len(cmp_)}); within 1 game: {((cmp_['mapped'] - cmp_['games']).abs() <= 1).mean():.3f}")
    # Point-in-time map against the final one: map each season at its last cutoff.
    agree = []
    for s in range(2014, LAST_SEASON + 1):
        last = wh.schedule([s])["cutoff_utc"].max()
        live = coach_game_map(wh.view(last))
        live = live[live["season"] == s].merge(cmap[["game_id", "team_id", "coach_id"]], on=["game_id", "team_id"],
                                               suffixes=("", "_final")).dropna(subset=["coach_id_final"])
        agree.append((live["coach_id"] == live["coach_id_final"]).mean())
    print("point-in-time vs final coach, by season 2014-2025: " + " ".join(f"{a:.3f}" for a in agree))

    # First-year effect: home margin minus pregame Elo expectation and home edge.
    pre = view.table("game_pregame")
    gi = view.table("game_index", columns=["game_id", "season", "home_id", "away_id", "neutral", "home_points",
                                           "away_points", "home_fbs", "away_fbs"])
    g = gi[gi["home_fbs"] & gi["away_fbs"]].merge(pre, on="game_id")
    fy = cmap.assign(fy=(cmap["tenure_seasons"] <= 1).astype(float))[["game_id", "team_id", "fy"]]
    g = g.merge(fy.rename(columns={"team_id": "home_id", "fy": "h_fy"}), on=["game_id", "home_id"])
    g = g.merge(fy.rename(columns={"team_id": "away_id", "fy": "a_fy"}), on=["game_id", "away_id"])
    g = g.dropna(subset=["home_pregame_elo", "away_pregame_elo", "h_fy", "a_fy", "home_points", "away_points"])
    X = np.column_stack([(~g["neutral"]).astype(float), (g["home_pregame_elo"] - g["away_pregame_elo"]) / 100,
                         g["h_fy"] - g["a_fy"]])
    y = (g["home_points"] - g["away_points"]).to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var())
    print(f"first-year coach effect: {beta[2]:+.2f} pts (se {se[2]:.2f}, n={len(g)}); "
          f"share of team-games flagged {(cmap['tenure_seasons'] <= 1).mean():.3f}")

    # ---- ATS ----------------------------------------------------------------------
    print("\n== ATS (cover rate against the closing spread)")
    spreads = M.closing_spreads(view.table("lines", columns=["id", "lines"]))
    ats = M.ats_results(hist, spreads)
    ats = ats.assign(resid=ats["cover"] - 0.5)
    per = ats.groupby("coach_id")["cover"].agg(["mean", "size"])
    tv, sv = mom(per["mean"], per["size"], pd.Series(0.25, index=per.index))
    rel, k = split_half(ats, "cover", rng, MIN["ats"])
    print(f"games with a line: {len(ats)}; coaches >= {2 * MIN['ats']} games: {(per['size'] >= 2 * MIN['ats']).sum()}")
    print(f"split-half reliability (>= {MIN['ats']} games per half, {k} coaches): {rel:.3f}")
    print(f"true variance {tv:.5f}, sampling {sv:.5f} -> prior games 0.25/true = "
          f"{(0.25 / tv) if tv > 0 else float('inf'):.0f}")
    for prior in (25, 100, 200, 400):
        print(f"  predictive r (career through S-1 vs S), prior {prior}: {predictive(ats, None, prior):.3f}")

    # ---- fourth down -------------------------------------------------------------------
    pbp = view.table("pbp", columns=M.PBP_COLUMNS)
    coach_of = hist[["game_id", "team_id", "coach_id"]]
    for label, plays, min_n, key in (("fourth-down go rate over expected", M.fourth_down_plays(pbp), MIN["fourth"],
                                      "fourth"),
                                     ("neutral run rate over expected", M.neutral_plays(pbp), MIN["run"], "run")):
        print(f"\n== {label}")
        f = M.add_expectation(plays).merge(coach_of, left_on=["game_id", "offense_id"],
                                           right_on=["game_id", "team_id"])
        f = f[f["season"] <= LAST_SEASON]
        per = f.groupby("coach_id").agg(rate=("resid", "mean"), n=("resid", "size"),
                                        noise=("exp", lambda e: float((e * (1 - e)).mean())))
        tv, sv = mom(per["rate"], per["n"], per["noise"])
        noise = float((f["exp"] * (1 - f["exp"])).mean())
        rel, k = split_half(f, "resid", rng, min_n)
        print(f"plays {len(f)}; mean exp {f['exp'].mean():.3f}; coaches >= {2 * min_n}: {(per['n'] >= 2 * min_n).sum()}")
        print(f"split-half reliability (>= {min_n} per half, {k} coaches): {rel:.3f}")
        print(f"true variance {tv:.6f}, sampling {sv:.6f}, per-play noise {noise:.3f} -> prior plays noise/true = "
              f"{(noise / tv) if tv > 0 else float('inf'):.0f}")
        prior = M.FOURTH_PRIOR if key == "fourth" else M.RUN_PRIOR
        for hl in (0.5, 1.0, 2.0, 4.0, None):
            print(f"  predictive r, half-life {hl}, prior {prior:.0f}: {predictive(f, hl, prior):.3f}")
        hl = M.FOURTH_HALF_LIFE if key == "fourth" else M.RUN_HALF_LIFE
        for p in (prior / 4, prior / 2, prior, prior * 2):
            print(f"  predictive r, half-life {hl}, prior {p:.0f}: {predictive(f, hl, p):.3f}")
        if key == "run":
            neutral = f

    # ---- flexibility ---------------------------------------------------------------------
    print("\n== flexibility (run rate vs opponent rush-minus-pass defensive gap)")
    gaps = M.defensive_gaps(view.table("unit_ratings", columns=["team_id", "cutoff_utc", "unit", "epa_adj",
                                                                "n_plays"]))
    cut = view.schedule()[["game_id", "cutoff_utc"]]
    games = M.game_run_residuals(neutral.drop(columns=["team_id", "coach_id"])).merge(
        hist[["game_id", "team_id", "opp_id", "coach_id", "season"]], on=["game_id", "team_id"])
    games = games.merge(cut, on="game_id").merge(gaps.rename(columns={"team_id": "opp_id", "gap": "x"}),
                                                 on=["opp_id", "cutoff_utc"], how="left")
    rated = games.dropna(subset=["x"])
    print(f"team-games with neutral plays: {len(games)}; with a rated opponent: {len(rated)}; sd(x) {rated['x'].std():.4f}")
    # Game-level extra variance after removing coach-season means.
    grp = rated.groupby(["coach_id", "season"])
    dev = rated["r"] - grp["r"].transform("mean")
    size = grp["r"].transform("size")
    tau2 = float((dev ** 2 * size / (size - 1)).where(size > 1).mean() - (0.25 / rated["n"]).mean())
    print(f"game-level extra variance of run rate over expected: {tau2:.4f} (constant {M.FLEX_GAME_VAR})")
    slopes, pooled = M.flex_slopes(rated)
    print(f"pooled within coach-season slope: {pooled:.4f}")
    sl = slopes.dropna(subset=["raw_slope"])
    # Sampling variance of each raw slope is 1/Sxx; recover Sxx from the shrink factor.
    g2 = rated.assign(w=1.0 / (0.25 / rated["n"] + M.FLEX_GAME_VAR))
    g2 = g2.assign(wx=g2["w"] * g2["x"])
    gg = g2.groupby(["coach_id", "season"])
    xc = g2["x"] - gg["wx"].transform("sum") / gg["w"].transform("sum")
    sxx = (g2["w"] * xc * xc).groupby(g2["coach_id"]).sum()
    sl = sl.set_index("coach_id").join(sxx.rename("sxx"))
    sl = sl[sl["sxx"] > 0]
    wt = sl["sxx"] / sl["sxx"].sum()
    obs = float((wt * (sl["raw_slope"] - pooled) ** 2).sum())
    samp = float((wt / sl["sxx"]).sum())
    print(f"slope variance: observed {obs:.4f}, sampling {samp:.4f}, true {obs - samp:.4f} "
          f"(constant {M.FLEX_SLOPE_VAR})")
    # Split-half: random halves of games within each coach.
    units = rated[["coach_id", "game_id"]].drop_duplicates()
    units = units.assign(half=rng.integers(0, 2, len(units)))
    h = rated.merge(units, on=["coach_id", "game_id"])
    halves = []
    for k_ in (0, 1):
        part = h[h["half"] == k_]
        s_, _ = M.flex_slopes(part)
        halves.append(s_.set_index("coach_id"))
    both = halves[0].join(halves[1], lsuffix="_0", rsuffix="_1", how="inner")
    enough = both[(both["games_0"] >= MIN["flex"]) & (both["games_1"] >= MIN["flex"])]
    r_raw = wcorr(enough["raw_slope_0"], enough["raw_slope_1"])
    print(f"split-half reliability of the raw slope (>= {MIN['flex']} rated games per half, {len(enough)} coaches): "
          f"{spearman_brown(r_raw):.3f}")
    print(f"coaches with >= {2 * MIN['flex']} rated games: {(slopes['games'] >= 2 * MIN['flex']).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

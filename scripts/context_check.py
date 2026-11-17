"""Check the context features (W6) against the effects measured for the plan doc.

    python scripts/context_check.py [--data-dir data] [--out docs/context_check.txt]

Seasons 2014-2025 only (2026 is the test season and is never read here). Sample and method
follow /mnt/project-files/exploration/effect-sizes/effects.py: completed FBS vs FBS games
with CFBD pregame Elo for both teams; OLS of the home margin on 1, on-campus indicator,
Elo gap, plus one candidate variable, HC1 standard errors. Each row prints the effect
measured with the ctx_ columns next to the value in the plan doc, and z = difference
over the plan doc's standard error. The two runs share almost all games, so |z| < 2 is
"within sampling error"; a larger gap points to a definition difference.

Also prints the shrinkage behind ctx_home_resid / ctx_road_resid and the split-half
reliability of those team values, and effects of the ctx_ columns the plan doc did not
measure (QB change, crowd, portal before bowls, weather on totals).

This script reads outcomes (Warehouse.outcomes), which feature code may not.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.core import Warehouse
from cfb.features.base import run_group
from cfb.features.context import Crowd, HomeRoad, Portal, QBContinuity, ScheduleContext, SeasonPhase, Travel, \
    Venue, Weather

LAST_SEASON = 2025

# Plan doc / effects_output.txt (2026-11-16 snapshot): name -> (coef, se)
MEASURED = {
    "home field (on-campus)": (2.191, None),
    "rest diff (days, capped 21)": (0.107, 0.058),
    "home off bye, away not": (1.864, 0.644),
    "away off bye, home not": (-0.515, 0.664),
    "away travel (per 1000 km), on-campus": (0.682, 0.213),
    "away travel > 2000 km, on-campus": (1.469, 0.626),
    "away crosses >= 2 tz eastward": (1.515, 0.900),
    "away crosses >= 2 tz westward": (2.097, 0.888),
    "venue >= 1200 m, away home < 600 m": (1.557, 0.969),
    "away surface mismatch": (0.686, 0.392),
    "indoor venue, away home outdoor": (-0.756, 1.274),
    "away body clock kickoff before 11:00": (3.033, 1.479),
    "night kickoff (local >= 18:00)": (0.517, 0.373),
    "conference game (home edge shift)": (-2.973, 0.431),
}


def ols(y: np.ndarray, X: np.ndarray):
    m = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[m], X[m]
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    V = XtXi @ (X.T * e ** 2) @ X @ XtXi * len(y) / (len(y) - X.shape[1])
    return b, np.sqrt(np.maximum(np.diag(V), 0.0)), len(y), e  # collinear subsets give ~0


def pivot(team: pd.DataFrame, games: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Team-level rows to h_<col> and a_<col> columns per game."""
    out = games[["game_id", "home_id", "away_id"]]
    for side, key in (("h", "home_id"), ("a", "away_id")):
        part = team[["game_id", "team_id", *cols]].rename(columns={"team_id": key, **{c: f"{side}_{c}" for c in cols}})
        out = out.merge(part, on=["game_id", key], how="left")
    return out.drop(columns=["home_id", "away_id"])


def build_frame(wh: Warehouse) -> pd.DataFrame:
    sched = wh.schedule(range(2014, LAST_SEASON + 1))
    res = wh.outcomes()[["game_id", "home_points", "away_points"]]
    elo = wh.view(sched["cutoff_utc"].max()).table("game_pregame")
    games = sched[sched["home_fbs"] & sched["away_fbs"]].merge(res, on="game_id").merge(elo, on="game_id")
    games = games.dropna(subset=["home_pregame_elo", "away_pregame_elo"]).reset_index(drop=True)
    games["margin"] = games["home_points"] - games["away_points"]
    games["total"] = games["home_points"] + games["away_points"]
    games["elo_diff"] = games["home_pregame_elo"] - games["away_pregame_elo"]
    games["home"] = (~games["neutral"].astype(bool)).astype(float)
    target = sched[sched["game_id"].isin(games["game_id"])]
    frame = games
    for group in (ScheduleContext(), Travel(), HomeRoad(), Crowd(), Portal(), QBContinuity()):
        t0 = time.time()
        out = run_group(group, wh, target)
        frame = frame.merge(pivot(out, target, group.column_names), on="game_id", how="left")
        print(f"built {group.name} in {time.time() - t0:.0f}s", file=sys.stderr)
    for group in (SeasonPhase(), Venue(), Weather()):
        t0 = time.time()
        frame = frame.merge(run_group(group, wh, target), on="game_id", how="left")
        print(f"built {group.name} in {time.time() - t0:.0f}s", file=sys.stderr)
    return frame


def report(G: pd.DataFrame) -> list[str]:
    lines = [f"n games (FBS vs FBS, 2014-{LAST_SEASON}, Elo present) = {len(G)}"]
    base = lambda d: [np.ones(len(d)), d["home"].to_numpy(), d["elo_diff"].to_numpy()]
    b, s, n, _ = ols(G["margin"].to_numpy(float), np.column_stack(base(G)))
    lines.append(f"base: margin = {b[0]:.2f} + {b[1]:.2f}*home + {b[2] * 100:.2f}*(elo_diff/100); n={n}")
    rows = [("home field (on-campus)", b[1], s[1], n)]

    def eff(name, x, df=G):
        x = np.asarray(x, float)
        bb, ss, nn, _ = ols(df["margin"].to_numpy(float), np.column_stack(base(df) + [x]))
        rows.append((name, bb[-1], ss[-1], nn))

    NN = G[G["home"] == 1]
    hr, ar = G["h_ctx_rest_days"], G["a_ctx_rest_days"]
    both = hr.notna() & ar.notna()
    eff("rest diff (days, capped 21)", hr.clip(upper=21) - ar.clip(upper=21))
    eff("home off bye, away not", ((G["h_ctx_off_bye"] == 1) & (G["a_ctx_off_bye"] == 0)).astype(float).where(both))
    eff("away off bye, home not", ((G["a_ctx_off_bye"] == 1) & (G["h_ctx_off_bye"] == 0)).astype(float).where(both))
    eff("away travel (per 1000 km), on-campus", NN["a_ctx_travel_km"] / 1000, NN)
    eff("away travel > 2000 km, on-campus", (NN["a_ctx_travel_km"] > 2000).astype(float).where(NN["a_ctx_travel_km"].notna()), NN)
    tz = NN["a_ctx_tz_shift_h"]
    eff("away crosses >= 2 tz eastward", (tz >= 2).astype(float).where(tz.notna()), NN)
    eff("away crosses >= 2 tz westward", (tz <= -2).astype(float).where(tz.notna()), NN)
    a_elev = NN["ctx_venue_elev_m"] - NN["a_ctx_elev_gain_m"]
    eff("venue >= 1200 m, away home < 600 m",
        ((NN["ctx_venue_elev_m"] >= 1200) & (a_elev < 600)).astype(float).where(NN["ctx_venue_elev_m"].notna() & a_elev.notna()), NN)
    eff("away surface mismatch", NN["a_ctx_surface_change"], NN)
    eff("indoor venue, away home outdoor",
        ((NN["ctx_venue_dome"] == 1) & (NN["a_ctx_dome_change"] == 1)).astype(float).where(NN["ctx_venue_dome"].notna()), NN)
    body = NN["a_ctx_body_clock_hour"]
    eff("away body clock kickoff before 11:00", (body < 11).astype(float).where(body.notna()), NN)
    loc = NN["ctx_kickoff_local_hour"]
    eff("night kickoff (local >= 18:00)", (loc >= 18).astype(float).where(loc.notna()), NN)
    eff("conference game (home edge shift)", NN["ctx_conference_game"], NN)

    lines.append("")
    lines.append(f"{'effect on home margin (points)':42s} {'ctx_ value':>18s} {'plan doc':>16s} {'z':>6s}  n")
    worst = 0.0
    for name, coef, se, nn in rows:
        ref, ref_se = MEASURED[name]
        z = (coef - ref) / ref_se if ref_se else np.nan
        worst = max(worst, abs(z)) if np.isfinite(z) else worst
        ok = "" if not np.isfinite(z) else ("  ok" if abs(z) < 2 else "  CHECK")
        lines.append(f"{name:42s} {coef:+8.3f} (se {se:.3f}) {ref:+8.3f}" + (f" ({ref_se:.3f})" if ref_se else "        ")
                     + (f" {z:+6.2f}" if np.isfinite(z) else "       ") + f"  {nn}{ok}")
    lines.append(f"largest |z| = {worst:.2f}")

    # ---- new effects (no plan-doc value) ----
    lines.append("")
    lines.append("== ctx_ variables the plan doc did not measure (same model; home minus away unless noted) ==")
    new = []

    def neff(name, x, df=G, y="margin", controls=True):
        x = np.asarray(x, float)
        X = np.column_stack((base(df) if controls else [np.ones(len(df))]) + [x])
        bb, ss, nn, _ = ols(df[y].to_numpy(float), X)
        new.append(f"{name:52s} coef={bb[-1]:+7.3f} se={ss[-1]:.3f} t={bb[-1] / ss[-1]:+5.1f} n={nn}")

    neff("QB leader changed in last game (diff)", G["h_ctx_qb_changed"] - G["a_ctx_qb_changed"])
    neff("QB last-game leader share of attempts (diff, per 0.1)",
         (G["h_ctx_qb_leader_share"] - G["a_ctx_qb_leader_share"]) / 0.1)
    neff("passers with 10+ attempts (diff)", G["h_ctx_qb_passers_10"] - G["a_ctx_qb_passers_10"])
    neff("home team prior-season fill rate (per 0.1), on-campus", NN["h_ctx_crowd_prior_fill"] / 0.1, NN)
    neff("home team prior-season attendance (per 10k), on-campus", NN["h_ctx_crowd_prior_att"] / 10, NN)
    neff("team home resid minus opp road resid, on-campus", NN["h_ctx_home_resid"] - NN["a_ctx_road_resid"], NN)
    post = G[G["ctx_postseason"] == 1]
    neff("bowls: portal entries since last game (diff)",
         post["h_ctx_portal_out_since_last"] - post["a_ctx_portal_out_since_last"], post)
    neff("bowls: rated (3*+) portal entries since last game (diff)",
         post["h_ctx_portal_out_rated_since_last"] - post["a_ctx_portal_out_rated_since_last"], post)
    neff("bowls: days since last regular-season game (diff)",
         post["h_ctx_days_since_regular"] - post["a_ctx_days_since_regular"], post)
    neff("travel km difference at neutral sites (per 1000)", G[G["home"] == 0]["ctx_travel_km_diff"] / 1000,
         G[G["home"] == 0])
    out = G[(G["ctx_wx_indoors"] == 0)]
    neff("weather: wind (per 10 mph) on total points", out["ctx_wx_wind_mph"] / 10, out, "total", False)
    neff("weather: precipitation > 0 on total points", (out["ctx_wx_precip_in"] > 0).astype(float), out, "total", False)
    neff("weather: temperature < 32F on total points", (out["ctx_wx_temp_f"] < 32).astype(float), out, "total", False)
    lines += new
    lines.append("plan doc, weather on totals (outdoor games with an O/U): wind -1.04 (t=-2.6), precip>0 -2.68 (t=-4.0), "
                 "temp<32F -4.49 (t=-3.1)")
    return lines


def shrinkage(G: pd.DataFrame, wh: Warehouse) -> list[str]:
    """Between-team variance of 4-season home and road residual means, and split-half reliability."""
    base = np.column_stack([np.ones(len(G)), G["home"], G["elo_diff"]])
    b = np.linalg.lstsq(base, G["margin"].to_numpy(float), rcond=None)[0]
    r = G.assign(resid=G["margin"] - base @ b)
    r = r[r["home"] == 1]
    long = pd.concat([
        pd.DataFrame({"team": r["home_id"], "season": r["season"], "rt": r["resid"], "where": "home"}),
        pd.DataFrame({"team": r["away_id"], "season": r["season"], "rt": -r["resid"], "where": "road"}),
    ])
    s2 = float(long["rt"].var())
    lines = ["", "== Team home/road residuals (shrinkage and reliability) ==", f"per-game residual SD = {np.sqrt(s2):.2f}"]
    long["blk"] = (long["season"] - 2014) // 4
    long["half"] = long["season"] % 2
    for where in ("home", "road"):
        d = long[long["where"] == where]
        g = d.groupby(["team", "blk"])["rt"].agg(["mean", "count"])
        g = g[g["count"] >= 10]
        tau2 = g["mean"].var() - (s2 / g["count"]).mean()
        k = s2 / tau2 if tau2 > 0 else np.inf
        sh = d.groupby(["team", "half"])["rt"].mean().unstack().dropna()
        lines.append(f"{where}: {len(g)} team-blocks, between-team variance {tau2:+.2f} pts^2 -> k = {k:.0f} games; "
                     f"split-half (odd vs even seasons) r = {sh.corr().iloc[0, 1]:+.3f} over {len(sh)} teams")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args(argv)
    wh = Warehouse(args.data_dir)
    G = build_frame(wh)
    assert G["season"].max() <= LAST_SEASON
    lines = report(G) + shrinkage(G, wh)
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

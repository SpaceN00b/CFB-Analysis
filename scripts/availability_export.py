"""Pack this week's player-availability package for the local app's Players page (test model).

Run after the depth-impact week steps (docs/depth-impact.md):

    python scripts/depth_impact_select.py --week 12      # writes data/exploration/depth_impact/app_week12_2026/
    python scripts/depth_impact_week.py --week 12 --out data/exploration/depth_impact_week
    python scripts/availability_export.py --week 12 [--publish]

Checks that the app's per-team recompute (cfb.ui.availability, no overrides) gives the cloud's
team costs and test-model probabilities for every game, then writes
<out>/availability_<key>.zip and, with --publish, uploads it to the R2 bucket under
extras/availability/ (outside the warehouse manifest; the weekly publish never deletes it).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.ui import availability as AV

DATA = Path("data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--inputs", type=Path, help="default data/exploration/depth_impact/app_week<N>_<season>")
    ap.add_argument("--games", type=Path, help="default data/exploration/depth_impact_week/week<N>_<season>.csv")
    ap.add_argument("--out", type=Path, default=Path("/mnt/project-files/reports/availability"))
    ap.add_argument("--listings", default="CFBDepth list downloaded 2026-11-18")
    ap.add_argument("--charts", default="team depth charts 2026-11-19")
    ap.add_argument("--publish", action="store_true", help="upload to the R2 bucket (extras/availability/)")
    a = ap.parse_args()
    S, W = a.season, a.week
    inp = a.inputs or DATA / "exploration" / "depth_impact" / f"app_week{W}_{S}"
    games_csv = a.games or DATA / "exploration" / "depth_impact_week" / f"week{W}_{S}.csv"

    T = {n: pd.read_parquet(inp / f"{n}.parquet") for n in AV.TABLES if n != "games"}
    g = pd.read_csv(games_csv)
    need = {"game_id", "home_id", "away_id", "lin", "calib_s", "p_v2", "p_dc", "home_team", "away_team"}
    if need - set(g.columns):
        raise SystemExit(f"{games_csv} lacks {sorted(need - set(g.columns))}: re-run depth_impact_week.py")
    g["start_time_utc"] = g["start_time_utc"].astype(str)
    g["neutral"] = g["neutral"].astype(bool)
    g["p4"] = g["p4"].astype(bool)
    for c in ("home_out", "away_out"):
        g[c] = g[c].fillna("")
    T["games"] = g
    t26 = T["teams"][T["teams"]["season"] == S]
    tid = dict(zip(t26["team"], t26["team_id"]))
    cs = T["cloud_scores"].copy()
    cs["team_id"] = cs["team"].map(tid)
    T["cloud_scores"] = cs
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        commit = None
    key = f"{S}_regular_{W}"
    meta = {"schema": AV.SCHEMA, "key": key, "season": S, "week": W,
            "label": f"{S} week {W}: model_v2 + depth-chart injury term (test model, PR 35)",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_commit": commit,
            "mu": json.loads((inp / "mu.json").read_text()),
            "sources": {"listings": a.listings, "charts": a.charts}}
    pkg = AV.WeekPackage(meta, T)
    pkg = AV.WeekPackage.from_bytes(pkg.to_bytes())  # what the app reads

    # the app's recompute with no overrides must give the cloud's numbers
    m = AV.WeekModel(pkg)
    m.cloud_cost = {}  # force the recompute
    worst, n = 0.0, 0
    cloud = cs.groupby("team_id")["cost"].sum()
    for t in sorted(m.week_game):
        c = m.team(t, {}).cost
        worst = max(worst, abs(c - float(cloud.get(t, 0.0))))
        n += 1
    p = m.picks({})
    dp = float(np.abs(p["p_mine"] - p["p_dc"]).max())
    print(f"{n} teams, {len(p)} games: max |team cost - cloud| {worst:.2e} pts, max |P(home) - cloud| {dp:.2e}")
    if worst > 1e-6 or dp > 1e-6:
        print("recompute does not match the cloud run", file=sys.stderr)
        return 1

    # one override per kind, as a smoke check
    lst = T["listings"]
    out_qb = lst.merge(T["lab"][["athlete_id", "grp"]], on="athlete_id")
    out_qb = out_qb[(out_qb["grp"] == "QB") & (out_qb["p"] >= 0.5)]
    for r in out_qb.head(3).itertuples(index=False):
        b, back = m.team(r.team_id, {}).cost, m.team(r.team_id, {int(r.athlete_id): "available"}).cost
        print(f"  {m.team_name.get(int(r.team_id))}: {m.names.get(int(r.athlete_id))} listed {r.status}: "
              f"cost {b:.2f} -> {back:.2f} when marked available")

    a.out.mkdir(parents=True, exist_ok=True)
    data = pkg.to_bytes()
    dest = a.out / f"availability_{key}.zip"
    dest.write_bytes(data)
    print(f"wrote {dest} ({len(data) / 1e6:.2f} MB)")
    if a.publish:
        from cfb import warehouse_sync as WS

        info = AV.publish(WS.open_store(WS.r2_settings()), pkg)
        print(f"published {info['object']} ({info['bytes'] / 1e6:.2f} MB, sha256 {info['sha256'][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Headline cases for the player-impact study: margin residual vs model_v2 (season-forward
2018-2025 and 2026) when a team's lead player at a position is out, from the team's side.

Lead player = highest share at the position among the team's role players before the game.
"New out" = no box-score line this game, played the previous one (known only afterwards).
"Missed last game" = no line in the previous game but a line in the one before (streak 1;
known before kickoff); his status in this game is not used.

    python scripts/impact_cases.py [--plays-stats]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

CASES = [("QB", "pass", 0.8), ("RB", "car", 0.4), ("WR", "rec", 0.25), ("TE", "rec", 0.15),
         ("DL", "tkl", 0.06), ("LB", "tkl", 0.10), ("DB", "tkl", 0.08)]


def main():
    base = pd.read_parquet("data/exploration/impact_base.parquet")
    if "--plays-stats" in sys.argv:   # target share (plays_stats) in place of reception share
        r = pd.read_parquet("data/exploration/impact_roles_ps.parquet")
        r["sh_rec"] = r["sh_tgt"]
        print("roles with plays_stats: 'rec' = target share, presence = box-score line or plays_stats row")
    else:
        r = pd.read_parquet("data/exploration/impact_roles.parquet")
    side = pd.concat([base.assign(team_id=base["home_id"], sgn=1.0), base.assign(team_id=base["away_id"], sgn=-1.0)])
    side["resid"] = side["sgn"] * (side["margin"] - side["pred"])
    print("margin residual vs model_v2 from the team's side, points (SE); n = team-games\n")
    print(f"{'position (share >=)':<22} {'period':<10} {'all team-games':>16} {'lead new out':>22} {'lead missed last game':>24}")
    for pos, c, th in CASES:
        s = r[(r["pos"] == pos)].sort_values(f"sh_{c}", ascending=False).drop_duplicates(["game_id", "team_id"])
        s = s[s[f"sh_{c}"] >= th]
        t = side.merge(s[["game_id", "team_id", "absent", "absent_prev", "streak", f"sh_{c}"]], on=["game_id", "team_id"])
        for lab, tt in (("2018-25", t[t.season <= 2025]), ("2026", t[t.season == 2026])):
            def fmt(x):
                return f"{x['resid'].mean():+5.2f} ({x['resid'].std() / np.sqrt(max(len(x), 1)):.2f}) n {len(x):4d}"
            new = tt[(tt["absent"] == 1) & (tt["absent_prev"] == 0)]
            prev = tt[tt["streak"] == 1]
            print(f"{pos + ' ' + c + ' ' + str(th):<22} {lab:<10} {fmt(tt):>16}   {fmt(new):>22}   {fmt(prev):>24}")
    # How often a pregame-known absence exists.
    big = r[(r["streak"] == 1) & (
        ((r["pos"] == "QB") & (r["sh_pass"] >= 0.5)) | ((r["pos"] == "RB") & (r["sh_car"] >= 0.3)) |
        (r["pos"].isin(["WR", "TE"]) & (r["sh_rec"] >= 0.15)))]
    tg = side[side.season <= 2025]
    k = tg.merge(big[["game_id", "team_id"]].drop_duplicates(), on=["game_id", "team_id"])
    print(f"\nteam-games 2018-2025 with a QB>=50% / RB>=30% carries / WR-TE>=15% catches who missed only the last game: "
          f"{len(k):,} of {len(tg):,} ({len(k) / len(tg):.1%})")


if __name__ == "__main__":
    main()

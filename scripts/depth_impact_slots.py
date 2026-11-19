"""Cost of a full-strength starter's absence by depth chart slot (diagnostic for docs/depth-impact.md).

Uses the realized-absence details written by scripts/depth_impact_select.py (Josiah's assumption: a
starter who did not play a regular-season game was out injured) and model_v2's season-forward margins.
Team-side residual = team margin minus model_v2's predicted margin. For each slot: mean residual in
team-games with that starter out and no other starter out, the mean depth-chart cost (value over the
slot's replacement), and points lost per point of cost (minus the slope of residual on cost over all
team-games; 1.0 means the cost is right on average).

    python scripts/depth_impact_slots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path("data/exploration")


def main():
    det = pd.read_parquet(EXP / "depth_impact" / "dc_details_realized.parquet")
    base = pd.read_parquet(EXP / "impact_base.parquet")
    base = base[base["season"].isin([2025, 2026]) & (base["season_type"] == "regular")]
    h = base.assign(team_id=base["home_id"], resid=base["margin"] - base["pred"])
    a = base.assign(team_id=base["away_id"], resid=-(base["margin"] - base["pred"]))
    tg = pd.concat([h, a])[["season", "game_id", "team_id", "resid"]]
    out = det[det["p_out"] >= 0.5].copy()
    out["slot"] = np.where(out["grp"] == "WR", "WR-" + out["align"].astype(str), out["grp"])
    out["cost"] = out["vor"] * out["p_out"]
    n_out = out.groupby(["game_id", "team_id"]).size().rename("n_out")
    rows = []
    for slot, g in [("all starters", out)] + list(out.groupby("slot")):
        one = g.merge(n_out.reset_index(), on=["game_id", "team_id"])
        one = one[one["n_out"] == 1].merge(tg, on=["season", "game_id", "team_id"])
        c = g.groupby(["game_id", "team_id"])["cost"].sum().rename("c").reset_index()
        m = tg.merge(c, on=["game_id", "team_id"], how="left").fillna({"c": 0.0})
        x, y = m["c"].to_numpy(), m["resid"].to_numpy()
        b = (x * y).sum() / max((x * x).sum(), 1e-9)
        se_b = np.sqrt(((y - b * x) ** 2).sum() / max(len(y) - 1, 1) / max((x * x).sum(), 1e-9))
        rows.append({"slot": slot, "starter-games out": len(g), "only starter out: n": len(one),
                     "mean residual": round(one["resid"].mean(), 2) if len(one) else None,
                     "se": round(one["resid"].std() / np.sqrt(len(one)), 2) if len(one) > 1 else None,
                     "mean cost": round(one["cost"].mean(), 2) if len(one) else None,
                     "lost per cost pt": round(-b, 2), "se ": round(se_b, 2)})
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "depth_impact" / "slots.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())

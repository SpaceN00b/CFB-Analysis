"""Do teams play down to weaker opponents and up to stronger ones? (seasons <= 2025 only)

Residual of a past game for team T: actual margin minus expected margin, where expected =
(T's power net - opponent's power net) + league home edge x site, all from the stored
features at that game's own cutoff (str_power, ctx_home_field), so every input is point in time.

Two questions:
1. League-wide: is the residual related to the size of the rating gap (do favourites as
   a group fall short of the gap)? Slope of the residual on the gap, and log loss with a
   non-linear gap term added to the model.
2. Team-specific: does a team's past "vs weaker" vs "vs stronger" residual predict its
   residual in the next game of the same kind? Feature for a game at cutoff c: the team's
   shrunk mean residual over its past games (this and last season, visible at c) in the
   same tier as this game (favourite by more than TIER points / underdog by more / close).
   Split-half reliability, and the change in season-forward log loss when added.

    python scripts/model_opponent_tier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_select as M  # noqa: E402

from cfb.core.timeutil import RESULT_LAG  # noqa: E402
from cfb.model import config as C  # noqa: E402
from cfb.model.matrix import feature_columns, groups_by_name  # noqa: E402
from cfb.model.models import LogitModel  # noqa: E402

TIER = 7.0     # points of power-rating gap that make a game a "favourite" or "underdog" game
SHRINK = 8.0   # pseudo-games at residual 0


def team_rows(X: pd.DataFrame) -> pd.DataFrame:
    """Two rows per completed game: team view of gap and residual."""
    g = X[X["completed"] & X["d_str_pwr_net"].notna()].copy()
    hfa = g["ctx_league_hfa"].fillna(g["ctx_league_hfa"].median())
    exp = g["d_str_pwr_net"] + hfa * g["site"]
    res = g["margin"] - exp
    home = pd.DataFrame({"game_id": g["game_id"], "team": g["home_id"], "season": g["season"],
                         "t": g["start_time_utc"], "gap": g["d_str_pwr_net"], "resid": res})
    away = pd.DataFrame({"game_id": g["game_id"], "team": g["away_id"], "season": g["season"],
                         "t": g["start_time_utc"], "gap": -g["d_str_pwr_net"], "resid": -res})
    r = pd.concat([home, away], ignore_index=True)
    r["tier"] = np.where(r["gap"] > TIER, 1, np.where(r["gap"] < -TIER, -1, 0))
    return r.sort_values("t").reset_index(drop=True)


def tier_feature(X: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Home minus away shrunk same-tier past residual, per game of X (point in time)."""
    by_team = {t: d for t, d in rows.groupby("team")}
    out = np.full(len(X), np.nan)
    for i, (gid, s, cut, h, a, gap) in enumerate(zip(X["game_id"], X["season"], X["cutoff_utc"],
                                                     X["home_id"], X["away_id"], X["d_str_pwr_net"])):
        if pd.isna(gap):
            continue
        vals = []
        for team, tg in ((h, gap), (a, -gap)):
            d = by_team.get(team)
            tier = 1 if tg > TIER else (-1 if tg < -TIER else 0)
            if d is None:
                vals.append(0.0)
                continue
            past = d[(d["t"] + RESULT_LAG <= cut) & (d["season"] >= s - 1) & (d["tier"] == tier)]
            vals.append(past["resid"].sum() / (len(past) + SHRINK))
        out[i] = vals[0] - vals[1]
    return pd.Series(out, index=X.index)


def reliability(rows: pd.DataFrame) -> None:
    """Split-half: per team and two-season window, (mean resid as favourite) - (as underdog)
    on odd vs even games."""
    out = []
    for end in range(2015, 2026):
        w = rows[rows["season"].between(end - 1, end)]
        for team, d in w.groupby("team"):
            d = d.reset_index(drop=True)
            halves = []
            for par in (0, 1):
                h = d[d.index % 2 == par]
                fav, dog = h[h["tier"] == 1]["resid"], h[h["tier"] == -1]["resid"]
                if len(fav) >= 3 and len(dog) >= 3:
                    halves.append(fav.mean() - dog.mean())
            if len(halves) == 2:
                out.append(halves)
    a = np.array(out)
    r = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
    print(f"split-half r of (resid as favourite - resid as underdog): {r:+.3f} over {len(a)} team-windows "
          f"(Spearman-Brown {2 * r / (1 + r):+.3f})")


def main():
    X, groups = M.load()
    rows = team_rows(X)
    fb = rows[rows["season"] <= 2025]
    print(f"team-game rows {len(fb)}; residual SD {fb['resid'].std():.2f} pts")
    for tier, name in ((1, "favourite by >7"), (0, "within 7"), (-1, "underdog by >7")):
        d = fb[fb["tier"] == tier]
        print(f"  {name:<18} n={len(d):5d} mean resid {d['resid'].mean():+.2f} (se {d['resid'].std() / np.sqrt(len(d)):.2f})")
    b = np.polyfit(fb["gap"], fb["resid"], 1)
    print(f"league-wide slope of residual on gap: {b[0]:+.4f} pts per pt of gap")
    reliability(fb)

    X["d_tier_resid"] = tier_feature(X, rows)
    X["gap_nl"] = X["d_str_pwr_net"] * X["d_str_pwr_net"].abs() / 20.0
    ev = X[X["in_eval"] & (X["season"] >= 2018)]
    c = np.corrcoef(ev["d_tier_resid"].fillna(0), (ev["margin"] - ev["d_str_pwr_net"] - ev["ctx_league_hfa"] * ev["site"]))[0, 1]
    print(f"corr of d_tier_resid with the game's own residual (eval games 2018-2025): {c:+.3f}")

    base = ["site", "fcs_game"] + feature_columns(groups_by_name(["str_power"]), ("d",))
    full = C.model_columns()
    for name, cols in (("str_power", base), ("config groups", full)):
        M.report(f"{name}", M.season_forward(X, lambda: LogitModel(cols, lam=C.LAM)))
        M.report(f"{name} + tier residual", M.season_forward(X, lambda: LogitModel(cols + ["d_tier_resid"], lam=C.LAM)))
        M.report(f"{name} + non-linear gap", M.season_forward(X, lambda: LogitModel(cols + ["gap_nl"], lam=C.LAM)))


if __name__ == "__main__":
    main()

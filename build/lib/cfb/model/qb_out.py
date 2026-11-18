"""Manual "QB out" override for the weekly run and on-demand matchups.

The model cannot see injury news. When the expected starter (cfb.features.strength.qb: the
leading passer of the team's last game) will not play, the team's predicted margin moves by

    points = A + B * (q_replacement - q_starter),   clipped to [MIN_POINTS, 0]

with q the shrunk, opponent-adjusted PPA per QB play of each player at the cutoff. A and B
are the least-squares fit of model_v1's margin residual on the quality gap, over the 529
games of 2018-2025 in which a team's primary starter (>= 50% of its QB plays) did not throw
a pass and one replacement took >= 80% of the passing plays (scripts/qb_oracle.py). The
win probability moves by the model's log-odds per point (LOGIT_PER_POINT unless the fitted
model's calibration is given).

Configuration (dashboard/config.toml):

    [qb_out]
    "Texas" = "out"                 # expected starter out, the next QB starts
    "Ohio State" = "Julian Sayin"   # expected starter out, this player starts

On demand: python -m cfb.predict matchup "Texas" "Texas A&M" --qb-out "Texas"
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..core.schemas import reasons_json
from ..features.strength import qb as Q

# scripts/qb_oracle.py, clean QB-out cases 2018-2025: a -1.23 (SE 0.71), b 24.2 (SE 8.1).
A = -1.23
B = 24.2
MIN_POINTS = -6.0
LOGIT_PER_POINT = 0.109  # model_v1 calibration slope (0.1085 fitted through 2021, 0.1097 through 2025)


def _names(view, season: int) -> pd.DataFrame:
    if not view.has("roster"):
        return pd.DataFrame(columns=["athlete", "team", "position", "full"])
    r = view.table("roster", seasons=[season]).reindex(columns=["id", "firstName", "lastName", "team", "position"])
    full = (r["firstName"].fillna("") + " " + r["lastName"].fillna("")).str.strip().str.lower()
    return pd.DataFrame({"athlete": r["id"].astype(str), "team": r["team"], "position": r["position"], "full": full})


def team_adjustment(view, team_id: int, replacement: str | None = None) -> dict:
    """Points to add to `team_id`'s margin when its expected starter is out.

    `replacement`: a player's name (roster of the cutoff season) or athlete id; default is the
    team's QB with the most passing plays this season other than the starter, then the roster
    QB with the most career plays, then the best recruit.
    """
    st, q = Q.qb_state(view)
    if team_id not in st.index or st.at[team_id, "athlete"] is None:
        return {"team_id": team_id, "points": 0.0, "note": "no expected starter known"}
    starter = st.at[team_id, "athlete"]
    q_start = float(st.at[team_id, "q"])
    names = _names(view, view.season)
    label = dict(zip(names["athlete"], names["full"].str.title()))
    team_name = Q.team_names(view)
    team_name = {v: k for k, v in team_name.items()}.get(team_id)
    repl = None
    if replacement:
        key = str(replacement).strip()
        hit = names[(names["full"] == key.lower()) & (names["team"] == team_name)]
        if hit.empty:
            hit = names[names["full"] == key.lower()]
        repl = hit["athlete"].iat[0] if len(hit) else (key if key in set(q["athlete"]) else None)
        if repl is None:
            raise ValueError(f"no player named {replacement!r} on a {view.season} roster")
    else:
        pg = Q.player_games(view)
        cur = pg[(pg["season"] == view.season) & (pg["team_id"] == team_id) & (pg["athlete"] != starter)] \
            if len(pg) else pg
        if len(cur):
            repl = cur.groupby("athlete")["pass_plays"].sum().sort_values(ascending=False).index[0]
        else:
            cand = names[(names["team"] == team_name) & (names["position"] == "QB") & (names["athlete"] != starter)]
            if len(cand):
                c = pd.DataFrame({"athlete": cand["athlete"].drop_duplicates()})
                pri = Q.player_priors(view, view.season, c["athlete"])
                c = c.merge(q[["athlete", "plays"]], on="athlete", how="left").merge(
                    pri[["athlete", "prior"]], on="athlete", how="left").fillna({"plays": 0.0})
                repl = c.sort_values(["plays", "prior"], ascending=False)["athlete"].iat[0]
    if repl is None:
        q_repl = float(Q.prior_mean(np.array([np.nan]), np.array([0.0]), np.array([np.nan]))[0])
    else:
        qi = q.set_index("athlete")["q"]
        q_repl = float(qi[repl]) if repl in qi.index else float(
            Q.player_priors(view, view.season, [repl])["prior"].iat[0])
    points = float(np.clip(A + B * (q_repl - q_start), MIN_POINTS, 0.0))
    return {"team_id": int(team_id), "starter": starter, "starter_name": label.get(starter, starter),
            "replacement": repl, "replacement_name": label.get(repl, repl or "unknown"),
            "q_starter": q_start, "q_replacement": q_repl, "points": points}


def resolve(view, overrides: dict) -> dict[int, dict]:
    """{team name or id: "out" | replacement name} -> {team_id: adjustment}."""
    ids = Q.team_names(view)
    lower = {k.lower(): v for k, v in ids.items()}
    out = {}
    for team, value in (overrides or {}).items():
        tid = int(team) if str(team).isdigit() else lower.get(str(team).lower())
        if tid is None:
            raise ValueError(f"unknown team {team!r} in [qb_out]")
        repl = None if str(value).strip().lower() in ("", "out") else str(value)
        out[int(tid)] = team_adjustment(view, int(tid), repl)
    return out


def apply(preds: pd.DataFrame, games: pd.DataFrame, adjustments: dict[int, dict],
          logit_per_point: float = LOGIT_PER_POINT) -> pd.DataFrame:
    """Shift p_home_win (and pred_margin when present) of `games` for the adjusted teams and
    add a reason. `preds` has game_id; `games` has game_id, home_id, away_id."""
    if not adjustments or preds.empty:
        return preds
    out = preds.copy()
    sides = games.set_index("game_id")[["home_id", "away_id"]]
    for i, gid in enumerate(out["game_id"].to_numpy()):
        if gid not in sides.index:
            continue
        h, a = (int(x) for x in sides.loc[gid])
        delta = adjustments.get(h, {}).get("points", 0.0) - adjustments.get(a, {}).get("points", 0.0)
        if delta == 0.0:
            continue
        p = float(np.clip(out["p_home_win"].iat[i], 1e-6, 1 - 1e-6))
        out.iloc[i, out.columns.get_loc("p_home_win")] = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p))
                                                                               + logit_per_point * delta)))
        if "pred_margin" in out.columns and pd.notna(out["pred_margin"].iat[i]):
            out.iloc[i, out.columns.get_loc("pred_margin")] = out["pred_margin"].iat[i] + delta
        if "reasons" in out.columns:
            items = json.loads(out["reasons"].iat[i]) if isinstance(out["reasons"].iat[i], str) else []
            for tid, sign in ((h, 1.0), (a, -1.0)):
                adj = adjustments.get(tid)
                if adj and adj.get("points"):
                    items.insert(0, {"text": f"QB out (manual): {adj['starter_name']} out, "
                                             f"{adj['replacement_name']} starts ({adj['points']:+.1f} pts)",
                                     "feature": "qb_out", "effect": round(sign * logit_per_point * adj["points"], 4)})
            out.iloc[i, out.columns.get_loc("reasons")] = reasons_json(items)
    return out

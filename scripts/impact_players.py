"""Player usage roles and absences from box scores (diagnostic; writes data/exploration/impact_roles.parquet).

One row per (team-game, player with an established role before that game). A role is the
player's share of the team's production in the team's earlier games of the same season in
which he recorded a box-score line ("share when playing"), over at least MIN_GAMES such games:

    pass   = passing attempts / team attempts
    car    = carries / team carries                (QBs included)
    rec    = receptions / team receptions          (a stand-in for target share; CFBD box
                                                    scores have no targets)
    ryd    = rushing + receiving yards / team scrimmage yards
    tkl    = total tackles / team tackles
    dplay  = TFL + sacks + passes defended + interceptions / team total of the same

For each row: `absent` = no box-score line in this game (any category); `absent_prev` = no
line in the team's previous game of the season (known before this game); `absent_next` = no
line in the next game (diagnostic only, -1 if none); `streak` = the
number of consecutive team games, ending with the previous one, that he had no line.
Position from the season roster (QB, RB, WR, TE, OL, DL, LB, DB, ST, other).

    python scripts/impact_players.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cfb.core import Warehouse

OUT = Path("data/exploration/impact_roles.parquet")
MIN_GAMES = 2
MIN_SHARE = 0.03   # keep rows with at least this share in some category

POS = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
       "OL": "OL", "OT": "OL", "G": "OL", "C": "OL", "OG": "OL",
       "DL": "DL", "DE": "DL", "DT": "DL", "NT": "DL", "EDGE": "DL",
       "LB": "LB", "ILB": "LB", "OLB": "LB",
       "DB": "DB", "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB",
       "PK": "ST", "K": "ST", "P": "ST", "LS": "ST"}
CATS = ["pass", "car", "rec", "ryd", "tkl", "dplay"]


def box_scores(root: Path) -> pd.DataFrame:
    frames = []
    for f in sorted((root / "tables" / "games_players").glob("*.parquet")):
        d = pd.read_parquet(f)
        if "gameId" in d:                                    # weeks not yet played are empty files
            frames.append(d[["gameId", "homeAway", "category", "statType", "athleteId", "stat", "season"]])
    d = pd.concat(frames, ignore_index=True)
    d = d[d["athleteId"].notna() & (d["athleteId"].astype(str) != "") & ~d["athleteId"].astype(str).str.startswith("-")]
    d["athleteId"] = d["athleteId"].astype(str)
    d["key"] = d["category"] + ":" + d["statType"]
    keep = {"passing:C/ATT", "rushing:CAR", "rushing:YDS", "receiving:REC", "receiving:YDS", "defensive:TOT",
            "defensive:TFL", "defensive:SACKS", "defensive:PD", "interceptions:INT"}
    appear = d[["gameId", "homeAway", "athleteId", "season"]].drop_duplicates()
    v = d[d["key"].isin(keep)].copy()
    att = v["key"] == "passing:C/ATT"
    v.loc[att, "stat"] = v.loc[att, "stat"].astype(str).str.split("/").str[-1]
    v["val"] = pd.to_numeric(v["stat"], errors="coerce").fillna(0.0)
    w = v.pivot_table(index=["gameId", "homeAway", "athleteId"], columns="key", values="val", aggfunc="sum",
                      fill_value=0.0).reset_index()
    for k in keep:
        if k not in w:
            w[k] = 0.0
    w["pass"] = w["passing:C/ATT"]
    w["car"] = w["rushing:CAR"]
    w["rec"] = w["receiving:REC"]
    w["ryd"] = w["rushing:YDS"] + w["receiving:YDS"]
    w["tkl"] = w["defensive:TOT"]
    w["dplay"] = w["defensive:TFL"] + w["defensive:SACKS"] + w["defensive:PD"] + w["interceptions:INT"]
    w = appear.merge(w[["gameId", "homeAway", "athleteId"] + CATS], on=["gameId", "homeAway", "athleteId"],
                     how="left").fillna({c: 0.0 for c in CATS})
    return w.drop(columns="season").rename(columns={"gameId": "game_id", "athleteId": "athlete"})


def positions(root: Path) -> pd.DataFrame:
    frames = []
    for f in sorted((root / "tables" / "roster").glob("*.parquet")):
        r = pd.read_parquet(f, columns=["id", "position", "season"])
        frames.append(r)
    r = pd.concat(frames, ignore_index=True)
    r["athlete"] = r["id"].astype(str)
    r["pos"] = r["position"].map(POS).fillna("other")
    return r.drop_duplicates(["season", "athlete"])[["season", "athlete", "pos"]]


def main():
    wh = Warehouse("data")
    gi = wh.game_index
    gi = gi[gi["completed"]]
    root = Path("data")
    bs = box_scores(root)
    side = pd.concat([
        gi[["game_id", "season", "start_time_utc", "home_id"]].rename(columns={"home_id": "team_id"}).assign(homeAway="home"),
        gi[["game_id", "season", "start_time_utc", "away_id"]].rename(columns={"away_id": "team_id"}).assign(homeAway="away"),
    ])
    bs = bs.merge(side[["game_id", "homeAway", "team_id"]], on=["game_id", "homeAway"])
    have_box = bs[["game_id", "team_id"]].drop_duplicates()
    tg = side.merge(have_box, on=["game_id", "team_id"])            # team-games with a box score
    tg = tg.sort_values(["team_id", "season", "start_time_utc"]).reset_index(drop=True)
    tg["gno"] = tg.groupby(["team_id", "season"]).cumcount()
    tot = bs.groupby(["game_id", "team_id"])[CATS].sum().add_prefix("team_").reset_index()
    tg = tg.merge(tot, on=["game_id", "team_id"])
    print(f"team-games with box scores {len(tg):,}; player-game lines {len(bs):,}", flush=True)

    # Player x team-season: a row for every team game of the season (played or not), from his
    # first appearance for that team on.
    pg = bs.merge(tg[["game_id", "team_id", "season", "gno"] + [f"team_{c}" for c in CATS]],
                  on=["game_id", "team_id"])
    first = pg.groupby(["team_id", "season", "athlete"])["gno"].min().rename("first").reset_index()
    grid = first.merge(tg[["team_id", "season", "gno", "game_id"] + [f"team_{c}" for c in CATS]],
                       on=["team_id", "season"])
    grid = grid[grid["gno"] >= grid["first"]]
    grid = grid.merge(pg[["game_id", "team_id", "athlete"] + CATS].assign(played=1),
                      on=["game_id", "team_id", "athlete"], how="left")
    grid["played"] = grid["played"].fillna(0).astype(int)
    for c in CATS:
        grid[c] = grid[c].fillna(0.0)
        grid[f"team_{c}"] = grid[f"team_{c}"].where(grid["played"] == 1, 0.0)   # team totals in games he played
    grid = grid.sort_values(["team_id", "season", "athlete", "gno"]).reset_index(drop=True)
    g = grid.groupby(["team_id", "season", "athlete"], sort=False)
    # Cumulative before this game (shift 1).
    cum = {}
    for c in ["played"] + CATS + [f"team_{c}" for c in CATS]:
        cum[c] = g[c].cumsum() - grid[c]
    out = grid[["team_id", "season", "athlete", "gno", "game_id", "played"]].copy()
    out["games_played"] = cum["played"]
    for c in CATS:
        den = cum[f"team_{c}"]
        out[f"sh_{c}"] = np.where(den > 0, cum[c] / den.where(den > 0, 1.0), 0.0)
    out["absent"] = 1 - out["played"]
    out["absent_prev"] = 1 - g["played"].shift(1).fillna(1).astype(int)
    # Diagnostic only (uses the next game): no line in the team's next game; -1 = no next game.
    out["absent_next"] = (1 - g["played"].shift(-1)).fillna(-1).astype(int)
    # Consecutive misses ending with the previous game.
    key = g.ngroup()
    miss = 1 - grid["played"]
    block = grid["played"].groupby(key).cumsum()
    run = miss.groupby([key, block]).cumsum()
    out["streak"] = run.groupby(key).shift(1).fillna(0).astype(int)
    out = out[out["games_played"] >= MIN_GAMES]
    out = out[(out[[f"sh_{c}" for c in CATS]].max(axis=1) >= MIN_SHARE)]
    out = out.merge(positions(root), on=["season", "athlete"], how="left")
    out["pos"] = out["pos"].fillna("other")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"role rows {len(out):,}; absent {out['absent'].mean():.1%}; by pos:")
    print(out.groupby("pos").agg(n=("absent", "size"), absent=("absent", "mean")).to_string())


if __name__ == "__main__":
    main()

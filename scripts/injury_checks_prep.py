"""Inputs for the per-player injury checks (date of injury, expected return).

One row per CFBDepth player-updates row (all FBS teams), with the matched CFBD athlete id and
the team's 2026 schedule marked by whether the player had a box-score line or a /plays/stats
row in each game (offensive linemen have neither, so their marks are "n/a"). Written in
batches for the checkers.

    python scripts/injury_checks_prep.py [--src CSV] [--out DIR] [--batches 12]
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

SEASON = 2026
SRC = Path("/mnt/project-files/injury-reports/cfbdepth_2026-11-18/cfbdepth-player-updates.csv")
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
P4 = {"ACC", "Big Ten", "Big 12", "SEC"}
OL = {"OL", "OT", "OG", "C", "G", "T", "IOL", "LS"}


def _norm(name) -> str:
    toks = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " ")).split()
    return " ".join(t for t in toks if t not in SUFFIXES)


def match_ids(d: pd.DataFrame, roster: pd.DataFrame) -> pd.Series:
    """team + full name, then team + first initial + last name (unique keys only)."""
    r = roster.assign(full=(roster["firstName"].fillna("") + " " + roster["lastName"].fillna("")).map(_norm))
    r["key2"] = r["full"].str.split().map(lambda t: f"{t[0][:1]} {t[-1]}" if t else "")
    k1 = r.drop_duplicates(["team", "full"], keep=False).set_index(["team", "full"])["id"]
    k2 = r.drop_duplicates(["team", "key2"], keep=False).set_index(["team", "key2"])["id"]
    full = d["Name"].map(_norm)
    key2 = full.str.split().map(lambda t: f"{t[0][:1]} {t[-1]}" if t else "")
    a = [k1.get((t, n)) for t, n in zip(d["Team"], full)]
    b = [k2.get((t, n)) for t, n in zip(d["Team"], key2)]
    return pd.Series([x if x is not None else y for x, y in zip(a, b)], index=d.index, dtype="object")


def _ids(data: Path, table: str) -> pd.DataFrame:
    """(gameId, athleteId) rows of a season's partitions; empty partitions skipped."""
    out = []
    for f in glob.glob(str(data / "tables" / table / f"{SEASON}_*.parquet")):
        t = pd.read_parquet(f)
        if {"gameId", "athleteId"} <= set(t.columns):
            out.append(t[["gameId", "athleteId"]])
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=12)
    a = ap.parse_args()
    data = a.data_dir
    d = pd.read_csv(a.src)
    d.insert(0, "row_id", range(1, len(d) + 1))
    roster = pd.read_parquet(data / "tables" / "roster" / f"{SEASON}.parquet")
    d["athlete_id"] = match_ids(d, roster)
    pos = roster.drop_duplicates("id").set_index("id")["position"]
    d["roster_pos"] = d["athlete_id"].map(pos)

    g = pd.read_parquet(data / "tables" / "games" / f"{SEASON}_regular.parquet")
    g = g[g["completed"].fillna(False).astype(bool)]
    g["date"] = pd.to_datetime(g["startDate"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%m-%d")
    conf = dict(zip(g["homeTeam"], g["homeConference"])) | dict(zip(g["awayTeam"], g["awayConference"]))
    sched = {}
    for r in g.itertuples(index=False):
        for team, opp, ha, pf, pa in ((r.homeTeam, r.awayTeam, "vs", r.homePoints, r.awayPoints),
                                      (r.awayTeam, r.homeTeam, "at", r.awayPoints, r.homePoints)):
            sched.setdefault(team, []).append((r.startDate, r.id, f"{r.date} {ha} {opp} {'W' if pf > pa else 'L'} {int(pf)}-{int(pa)}"))
    pres = pd.concat([_ids(data, "games_players"), _ids(data, "plays_stats")]).dropna()
    pres["gameId"] = pd.to_numeric(pres["gameId"], errors="coerce").astype("Int64")
    pres = set(zip(pres["gameId"], pres["athleteId"].astype(str)))

    lines = []
    for r in d.itertuples(index=False):
        games = sorted(sched.get(r.Team, []))
        aid = None if r.athlete_id is None or pd.isna(r.athlete_id) else str(r.athlete_id)
        is_ol = str(r.Pos).upper() in OL or str(r.roster_pos).upper() in OL
        marks = []
        for _, gid, lab in games:
            if aid is None:
                m = "?"
            elif is_ol:
                m = "n/a"
            else:
                m = "played" if (int(gid), aid) in pres else "no stat line"
            marks.append(f"{lab}: {m}")
        lines.append("; ".join(marks))
    d["schedule_2026"] = lines
    d["conference"] = d["Team"].map(conf)
    d["p4_nd"] = d["conference"].isin(P4) | (d["Team"] == "Notre Dame")
    a.out.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out / "checks_input_all.csv", index=False)
    # P4 + Notre Dame first, then the rest, team by team so a batch keeps a team together
    d = d.sort_values(["p4_nd", "Team", "row_id"], ascending=[False, True, True]).reset_index(drop=True)
    size = -(-len(d) // a.batches)
    for i in range(a.batches):
        part = d.iloc[i * size:(i + 1) * size]
        part.to_csv(a.out / f"batch_{i + 1:02d}.csv", index=False)
        print(f"batch {i + 1:02d}: {len(part)} rows, P4+ND {int(part['p4_nd'].sum())}, teams {part['Team'].nunique()}")
    print("matched athlete ids:", int(d["athlete_id"].notna().sum()), "of", len(d))


if __name__ == "__main__":
    main()

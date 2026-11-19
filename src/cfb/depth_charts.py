"""Import team depth chart CSVs into the local store as a dated table.

Input: one CSV per team with the columns
    Team, Unit, Scheme, Position, Depth, Number, Player
for example `Georgia Bulldogs,Offense,Pro Spread,WR-X,First Team,6,"Canion, Isiah"`.
Units are Offense, Defense, Special Teams and Bench; depth runs First Team to Fifth Team
(Bench rows have no depth).

Output:
    data/raw/depth_charts/<asof_date>/<file>.csv         the CSV exactly as given
    data/tables/depth_charts/<asof_date>_<team>.parquet  one row per listed player

Each import is a snapshot as of one time (`--asof`, default: the file's modified time).
Importing a team again for the same date replaces its partition; another date adds one.
Team names are mapped to CFBD school names through `teams_fbs`, and players are matched
to CFBD athlete ids through `roster` (team, jersey and name).

Usage:
    python -m cfb.depth_charts import PATH [PATH ...] [--asof 2026-11-19T04:01:16Z]
    python -m cfb.depth_charts summary
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import pandas as pd

from . import settings
from .store import Store

DATASET = "depth_charts"
COLUMNS = ("Team", "Unit", "Scheme", "Position", "Depth", "Number", "Player")
UNITS = {"offense": "offense", "defense": "defense", "special teams": "special_teams", "bench": "bench"}
DEPTHS = {"first team": 1, "second team": 2, "third team": 3, "fourth team": 4, "fifth team": 5,
          "sixth team": 6, "bench": None}

# Position label -> coarse group. Labels are team-specific (JACK, BUCK, LEO, RAZOR are the
# fourth rusher of a four-man front; LOLB/ROLB the outside backers of a 3-4).
POS_GROUPS = {
    "QB": "QB", "RB": "RB", "FB": "RB", "TE": "TE",
    **{p: "WR" for p in ("WR-X", "WR-Z", "WR-SL", "WR-H", "WR-F", "WR-Y", "WR-A", "WR")},
    # Vanderbilt labels its line by quick and strong side (QT QG C SG ST).
    **{p: "OL" for p in ("LT", "LG", "C", "RG", "RT", "QT", "QG", "SG", "ST", "OT", "OG")},
    **{p: "DL" for p in ("DE", "LDE", "RDE", "DT", "LDT", "RDT", "NT", "DL")},
    **{p: "EDGE" for p in ("JACK", "BUCK", "LEO", "RAZOR", "LOLB", "ROLB", "OLB", "EDGE", "VYPER", "WOLF",
                           "RUSH", "DOG", "BAN")},
    **{p: "LB" for p in ("MLB", "WLB", "SLB", "ILB", "LB", "MAC", "MACK", "MONEY", "MIKE", "WILL", "SAM",
                         "STING", "ROVER", "CAT")},
    **{p: "DB" for p in ("LCB", "RCB", "CB", "SS", "FS", "S", "NB", "STAR", "NICKEL", "SPUR", "HUSKY")},
    **{p: "ST" for p in ("PK", "PT", "KO", "LS", "H", "KR", "PR")},
    "Bench": "BENCH",
}

# CFBD roster position -> group, used for a label not in POS_GROUPS when the player matched.
ROSTER_GROUPS = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "OL": "OL", "OT": "OL",
                 "G": "OL", "C": "OL", "DL": "DL", "DT": "DL", "NT": "DL", "DE": "DL", "EDGE": "EDGE",
                 "LB": "LB", "CB": "DB", "S": "DB", "DB": "DB", "PK": "ST", "P": "ST", "LS": "ST", "PR": "ST"}

# Source team names that are not CFBD "school mascot" or "alternate name mascot".
TEAM_OVERRIDES = {"mississippi rebels": "Ole Miss"}

# (CFBD team, jersey, player as given) -> CFBD athlete id, for players the roster cannot
# settle. Utah's Nick Brown (#13 CB) has two identical 2026 roster rows; 5226164 is the one
# with 2026 Utah game stats (and the 2025 Lamar #13 CB row), 518520 has no stats.
PLAYER_OVERRIDES = {("Utah", 13, "Brown, Nick"): "5226164"}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm(text: str) -> str:
    """Lowercase ASCII letters and digits only: 'White-Helton' -> 'whitehelton'."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def norm_words(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.replace("-", " "))).strip()


def split_name(raw: str) -> tuple[str, str, str]:
    """'Greene III, Earnest' -> ('Earnest', 'Greene', 'III'). Names without a comma are
    read as 'First Last'."""
    raw = re.sub(r"\s+", " ", str(raw)).strip()
    if "," in raw:
        last, first = (s.strip() for s in raw.split(",", 1))
    else:
        first, _, last = raw.rpartition(" ")
    words = last.split(" ")
    suffix = ""
    if len(words) > 1 and norm(words[-1]) in SUFFIXES:
        suffix = words[-1]
        last = " ".join(words[:-1])
    return first, last, suffix


def slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", norm_words(team)).strip("-")


def parse_asof(value: str | None, path: Path) -> pd.Timestamp:
    if value:
        ts = pd.Timestamp(value)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")


# ---- team and player matching --------------------------------------------------------

def team_lookup(teams: pd.DataFrame) -> dict[str, str]:
    """normalized 'school mascot' (and 'alternate name mascot') -> CFBD school."""
    out: dict[str, str] = {}
    for row in teams.itertuples(index=False):
        mascot = getattr(row, "mascot", "") or ""
        names = [row.school]
        alts = getattr(row, "alternateNames", None)
        if isinstance(alts, str) and alts.startswith("["):
            names += [a for a in json.loads(alts) if a]
        for name in names:
            out.setdefault(norm_words(f"{name} {mascot}"), row.school)
        out.setdefault(norm_words(row.school), row.school)
    for src, school in TEAM_OVERRIDES.items():
        out[norm_words(src)] = school
    return out


def match_players(frame: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Fill athlete_id and match from the CFBD roster of the same team.

    In order: jersey and last name; full name (first and last run together, so 'Hoyet
    Chance, John' finds 'John Hoyet' 'Chance'); last name unique on the team with the same
    first initial; jersey and first name (last name spelled differently). Otherwise none,
    which includes two roster rows that fit equally well. Also sets roster_position (CFBD).
    """
    frame = frame.copy()
    frame["athlete_id"] = pd.Series([None] * len(frame), dtype="object", index=frame.index)
    frame["match"] = "none"
    frame["roster_position"] = pd.Series([None] * len(frame), dtype="object", index=frame.index)
    if roster is None or roster.empty:
        return frame
    ros = roster.reindex(columns=["id", "team", "firstName", "lastName", "jersey", "position"]).copy()
    ros["id"] = ros["id"].astype(str)
    ros["first_k"] = ros["firstName"].fillna("").map(norm)
    last_words = ros["lastName"].fillna("").map(lambda s: " ".join(
        w for w in str(s).split(" ") if norm(w) not in SUFFIXES))
    ros["last_k"] = last_words.map(norm)
    ros["full_k"] = ros["first_k"] + ros["last_k"]
    ros["jersey"] = pd.to_numeric(ros["jersey"], errors="coerce").astype("Int64")
    by_team = {t: g for t, g in ros.groupby("team")}

    for idx, row in frame.iterrows():
        manual = PLAYER_OVERRIDES.get((row["team"], row.get("jersey"), row.get("player_raw")))
        if manual is not None:
            frame.at[idx, "athlete_id"] = manual
            frame.at[idx, "match"] = "manual"
            hit = ros[ros["id"] == manual]
            frame.at[idx, "roster_position"] = hit["position"].iloc[0] if len(hit) else None
            continue
        team = by_team.get(row["team"])
        if team is None:
            continue
        first_k, last_k = norm(row["first_name"]), norm(row["last_name"])
        jersey = row["jersey"]
        tries = []
        if pd.notna(jersey):
            tries.append(("jersey_last", (team["jersey"] == jersey) & (team["last_k"] == last_k)))
        tries.append(("name", team["full_k"] == first_k + last_k))
        same_last = team["last_k"] == last_k
        if same_last.sum() == 1:
            tries.append(("last_initial", same_last & (team["first_k"].str[:1] == first_k[:1])))
        if pd.notna(jersey):
            tries.append(("jersey_first", (team["jersey"] == jersey) & (team["first_k"] == first_k)))
        for method, mask in tries:
            hit = team[mask.fillna(False).to_numpy()]
            if len(hit) == 1:
                frame.at[idx, "athlete_id"] = hit["id"].iloc[0]
                frame.at[idx, "match"] = method
                frame.at[idx, "roster_position"] = hit["position"].iloc[0]
                break
    return frame


# ---- parsing --------------------------------------------------------------------------

def read_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; expected {list(COLUMNS)}")
    return raw[list(COLUMNS)]


def tidy(raw: pd.DataFrame, *, team_map: dict[str, str], season: int, asof: pd.Timestamp,
         source_file: str) -> tuple[pd.DataFrame, list[str]]:
    """One row per listed player. Returns the frame and a list of warnings."""
    warnings: list[str] = []
    rows = []
    for r in raw.itertuples(index=False):
        team_src = r.Team.strip()
        school = team_map.get(norm_words(team_src))
        if school is None:
            raise ValueError(f"{source_file}: team '{team_src}' is not a CFBD FBS team; "
                             "add it to TEAM_OVERRIDES in cfb/depth_charts.py")
        unit = UNITS.get(r.Unit.strip().lower())
        if unit is None:
            raise ValueError(f"{source_file}: unknown unit '{r.Unit}'")
        depth_label = r.Depth.strip()
        if depth_label.lower() not in DEPTHS:
            raise ValueError(f"{source_file}: unknown depth '{depth_label}'")
        depth = DEPTHS[depth_label.lower()]
        position = r.Position.strip()
        group = POS_GROUPS.get(position)
        if group is None:
            group = "OTHER"
            note = f"{team_src}: position '{position}' is not in POS_GROUPS (group from the CFBD roster)"
            if note not in warnings:
                warnings.append(note)
        first, last, suffix = split_name(r.Player)
        number = r.Number.strip()
        rows.append({
            "season": season, "asof_utc": asof.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "asof_date": asof.strftime("%Y-%m-%d"), "source_file": source_file,
            "source_team": team_src, "team": school, "unit": unit, "scheme": r.Scheme.strip(),
            "position": position, "pos_group": group, "depth_label": depth_label, "depth": depth,
            "starter": depth == 1 and unit in ("offense", "defense"),
            "jersey": int(number) if number.isdigit() else None, "player_raw": r.Player.strip(),
            "first_name": first, "last_name": last, "suffix": suffix,
            "player": " ".join(p for p in (first, last, suffix) if p),
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["depth"] = frame["depth"].astype("Int64")
        frame["jersey"] = frame["jersey"].astype("Int64")
    return frame, warnings


def _season_table(store: Store, dataset: str, season: int) -> pd.DataFrame | None:
    path = store.table_path(dataset, str(season))
    return pd.read_parquet(path) if path.exists() else None


def import_files(paths: list[Path], *, data_dir: Path, asof: str | None = None,
                 season: int | None = None, names: dict[Path, str] | None = None,
                 log=print) -> pd.DataFrame:
    """Import depth chart CSVs. `names` gives an original file name for a path (uploads
    arrive under generated names). Returns the imported rows."""
    store = Store(data_dir)
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.csv")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError("no CSV files given")
    out = []
    for path in files:
        ts = parse_asof(asof, path)
        yr = season or (ts.year if ts.month >= 7 else ts.year - 1)
        teams = _season_table(store, "teams_fbs", yr)
        if teams is None:
            raise FileNotFoundError(f"teams_fbs for {yr} not found under {data_dir}; restore the snapshot first")
        roster = _season_table(store, "roster", yr)
        name = (names or {}).get(path, path.name)
        frame, warnings = tidy(read_csv(path), team_map=team_lookup(teams), season=yr, asof=ts,
                               source_file=name)
        for w in warnings:
            log(f"  warning: {w}")
        frame = match_players(frame, roster)
        other = frame["pos_group"] == "OTHER"
        frame.loc[other, "pos_group"] = frame.loc[other, "roster_position"].map(ROSTER_GROUPS).fillna("OTHER")
        raw_dir = store.root / "raw" / DATASET / ts.strftime("%Y-%m-%d")
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, raw_dir / name)
        for team, part in frame.groupby("team", sort=False):
            key = f"{ts.strftime('%Y-%m-%d')}_{slug(team)}"
            dest = store.table_path(DATASET, key)
            dest.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(dest, index=False)
            matched = (part["match"] != "none").mean()
            log(f"{team:22s} {len(part):4d} rows  starters {int(part['starter'].sum()):2d}  "
                f"roster match {matched:.0%}  -> {dest}")
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def summary(data_dir: Path) -> pd.DataFrame:
    frame = Store(data_dir).load(DATASET)
    g = frame.groupby(["asof_date", "team"])
    return pd.DataFrame({
        "rows": g.size(), "starters": g["starter"].sum(),
        "matched": g["match"].apply(lambda m: round((m != "none").mean(), 3)),
        "other_positions": g["pos_group"].apply(lambda s: int((s == "OTHER").sum())),
    }).reset_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cfb.depth_charts", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="import depth chart CSV files or folders of them")
    imp.add_argument("paths", nargs="+", type=Path)
    imp.add_argument("--asof", help="when the charts were downloaded (ISO time; default: file modified time)")
    imp.add_argument("--season", type=int)
    imp.add_argument("--data-dir", type=Path, default=None)
    summ = sub.add_parser("summary", help="rows, starters and roster match rate by date and team")
    summ.add_argument("--data-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    data_dir = a.data_dir or settings.data_dir()
    if a.cmd == "import":
        rows = import_files(a.paths, data_dir=data_dir, asof=a.asof, season=a.season)
        matched = (rows["match"] != "none").mean()
        print(f"{rows['team'].nunique()} teams, {len(rows)} rows, roster match {matched:.1%}")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(summary(data_dir).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

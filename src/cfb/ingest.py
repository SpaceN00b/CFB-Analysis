"""Pull CollegeFootballData.com datasets into the local store.

Usage:
    python -m cfb.ingest --start 2014 --end 2026
    python -m cfb.ingest --start 2026 --end 2026 --force      # refresh current season
    python -m cfb.ingest --only games,lines --start 2020 --end 2025

Completed pulls are cached; a dataset/key already on disk is skipped unless
--force is given.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

from .client import CFBDClient, CFBDError
from .store import Store

SEASON_TYPES = ("regular", "postseason")
FALLBACK_WEEKS = {"regular": list(range(1, 17)), "postseason": [1]}


@dataclass(frozen=True)
class Dataset:
    name: str
    path: str
    grain: str  # "static" | "year" | "season_type" | "week"
    params: dict = field(default_factory=dict)


DATASETS = [
    Dataset("venues", "/venues", "static"),
    Dataset("teams_fbs", "/teams/fbs", "year"),
    Dataset("coaches", "/coaches", "year"),
    Dataset("records", "/records", "year"),
    Dataset("talent", "/talent", "year"),
    Dataset("returning_production", "/player/returning", "year"),
    Dataset("ratings_sp", "/ratings/sp", "year"),
    Dataset("ratings_elo", "/ratings/elo", "year"),
    Dataset("ratings_fpi", "/ratings/fpi", "year"),
    Dataset("ratings_srs", "/ratings/srs", "year"),
    Dataset("stats_season_advanced", "/stats/season/advanced", "year"),
    Dataset("games", "/games", "season_type", {"classification": "fbs"}),
    Dataset("lines", "/lines", "season_type"),
    Dataset("drives", "/drives", "season_type", {"classification": "fbs"}),
    Dataset("stats_game_advanced", "/stats/game/advanced", "season_type"),
    Dataset("ppa_games", "/ppa/games", "season_type"),
    Dataset("games_teams", "/games/teams", "week", {"classification": "fbs"}),
    Dataset("plays", "/plays", "week", {"classification": "fbs"}),
]
BY_NAME = {d.name: d for d in DATASETS}


def weeks_for(client: CFBDClient, store: Store, year: int, force: bool) -> dict[str, list[int]]:
    """Weeks per season type from /calendar, with a fixed fallback."""
    key = str(year)
    try:
        if force or not store.has("calendar", key):
            store.write("calendar", key, client.get("/calendar", year=year), {"season": year})
        cal = store.read_raw("calendar", key)
    except CFBDError as exc:
        print(f"  calendar {year}: {exc}; using fallback weeks", file=sys.stderr)
        return dict(FALLBACK_WEEKS)
    weeks: dict[str, set[int]] = {s: set() for s in SEASON_TYPES}
    for row in cal:
        stype = row.get("seasonType") or row.get("season_type")
        if stype in weeks and row.get("week") is not None:
            weeks[stype].add(int(row["week"]))
    return {s: sorted(w) or FALLBACK_WEEKS[s] for s, w in weeks.items()}


def tasks(ds: Dataset, year: int, weeks: dict[str, list[int]]):
    """Yield (key, params, extra columns) for one dataset and season."""
    if ds.grain == "year":
        yield str(year), {"year": year}, {"season": year}
    elif ds.grain == "season_type":
        for st in SEASON_TYPES:
            yield f"{year}_{st}", {"year": year, "seasonType": st}, {"season": year, "season_type": st}
    elif ds.grain == "week":
        for st in SEASON_TYPES:
            for wk in weeks[st]:
                yield (
                    f"{year}_{st}_{wk:02d}",
                    {"year": year, "seasonType": st, "week": wk},
                    {"season": year, "season_type": st, "week": wk},
                )


def run(client: CFBDClient, store: Store, years: list[int], names: list[str], force: bool = False) -> list[str]:
    """Run the pull. Returns a list of failure descriptions (empty on full success)."""
    failures: list[str] = []
    selected = [BY_NAME[n] for n in names]

    def pull(ds: Dataset, key: str, params: dict, extra: dict):
        if not force and store.has(ds.name, key):
            return
        try:
            rows = store.write(ds.name, key, client.get(ds.path, **ds.params, **params), extra)
            print(f"  {ds.name}/{key}: {rows} rows")
        except CFBDError as exc:
            failures.append(f"{ds.name}/{key}: {exc}")
            print(f"  {ds.name}/{key}: FAILED {exc}", file=sys.stderr)

    for ds in selected:
        if ds.grain == "static":
            pull(ds, "all", {}, {})
    for year in years:
        print(f"season {year}")
        needs_weeks = any(d.grain == "week" for d in selected)
        weeks = weeks_for(client, store, year, force) if needs_weeks else dict(FALLBACK_WEEKS)
        for ds in selected:
            if ds.grain == "static":
                continue
            for key, params, extra in tasks(ds, year, weeks):
                pull(ds, key, params, extra)
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", type=int, default=2014)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--force", action="store_true", help="re-download cached pulls")
    ap.add_argument("--list", action="store_true", help="list dataset names and exit")
    args = ap.parse_args(argv)

    if args.list:
        for d in DATASETS:
            print(f"{d.name:24s} {d.path:26s} per {d.grain}")
        return 0
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(BY_NAME)
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        ap.error(f"unknown datasets: {', '.join(unknown)}")

    client = CFBDClient()
    failures = run(client, Store(args.data_dir), list(range(args.start, args.end + 1)), names, args.force)
    print(f"done: {client.calls} API calls, {len(failures)} failures")
    for f in failures:
        print(f"  {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pull CollegeFootballData.com datasets into the local store.

Usage:
    python -m cfb.ingest --start 2014 --end 2026
    python -m cfb.ingest --start 2026 --end 2026 --force      # refresh current season
    python -m cfb.ingest --only games,lines --start 2020 --end 2025
    python -m cfb.ingest --coverage                           # rows per dataset and season

Completed pulls are cached; a dataset/key already on disk is skipped unless
--force is given. Three exceptions, all driven by the run time:

- Weeks that have not started are never pulled, so nothing empty is cached for them.
- `ratings_elo_weekly` pulls a week only after it has ended, because the API
  answers a later week with the latest ratings it has.
- SP+, FPI, SRS and CORE of the current season are pulled on every run and a
  dated copy is kept under `<dataset>_asof` (see `--asof-archive`).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from . import settings
from .client import CFBDClient, CFBDError
from .store import Store, flatten

SEASON_TYPES = ("regular", "postseason")
FALLBACK_WEEKS = {"regular": list(range(1, 17)), "postseason": [1]}
ASOF_SUFFIX = "_asof"


def poll_rows(payload: list) -> list[dict]:
    """/rankings: one row per season, week, poll and ranked team."""
    return [
        {
            "season": wk.get("season"),
            "seasonType": wk.get("seasonType"),
            "week": wk.get("week"),
            "poll": poll.get("poll"),
            "isFinal": poll.get("isFinal"),
            **rank,
        }
        for wk in payload
        for poll in wk.get("polls") or []
        for rank in poll.get("ranks") or []
    ]


def box_rows(payload: list) -> list[dict]:
    """/games/players: one row per game, team, category, stat type and athlete.

    `stat` stays text because the API mixes forms such as "15/22" and "5.3".
    """
    rows = []
    for game in payload:
        for team in game.get("teams") or []:
            base = {
                "gameId": game.get("id"),
                "team": team.get("team"),
                "conference": team.get("conference"),
                "homeAway": team.get("homeAway"),
            }
            for cat in team.get("categories") or []:
                for typ in cat.get("types") or []:
                    for ath in typ.get("athletes") or []:
                        stat = ath.get("stat")
                        rows.append(
                            {
                                **base,
                                "category": cat.get("name"),
                                "statType": typ.get("name"),
                                "athleteId": None if ath.get("id") is None else str(ath["id"]),
                                "athleteName": ath.get("name"),
                                "stat": None if stat is None else str(stat),
                            }
                        )
    return rows


@dataclass(frozen=True)
class Dataset:
    name: str
    path: str
    grain: str  # "static" | "year" | "season_type" | "week" | "regular_week" | "team"
    params: dict = field(default_factory=dict)
    lag: int = 0  # also pull this many years before the first requested season
    lead: int = 0  # also pull this many years after the last requested season
    asof: bool = False  # keep a dated copy on every run while the season is current
    ended_only: bool = False  # week grains: pull a week only after it has ended
    reshape: Callable[[list], list[dict]] | None = None  # payload -> rows of the table


DATASETS = [
    Dataset("venues", "/venues", "static"),
    Dataset("teams_fbs", "/teams/fbs", "year"),
    Dataset("coaches", "/coaches", "year"),
    Dataset("records", "/records", "year"),
    Dataset("talent", "/talent", "year"),
    Dataset("returning_production", "/player/returning", "year"),
    Dataset("ratings_sp", "/ratings/sp", "year", asof=True),
    Dataset("ratings_elo", "/ratings/elo", "year"),
    Dataset("ratings_fpi", "/ratings/fpi", "year", asof=True),
    Dataset("ratings_srs", "/ratings/srs", "year", asof=True),
    Dataset("ratings_core", "/ratings/core", "year", asof=True),
    Dataset("stats_season_advanced", "/stats/season/advanced", "year"),
    # Recruiting classes of the four prior years are part of a season's roster.
    Dataset("recruiting_teams", "/recruiting/teams", "year", lag=4),
    # Portal season Y+1 opens during season Y (bowl-season entries).
    Dataset("transfer_portal", "/player/portal", "year", lead=1),
    Dataset("ratings_elo_preseason", "/ratings/elo", "year", {"preseason": "true"}),
    # Year-only requests to the next five endpoints return both season types.
    Dataset("weather", "/games/weather", "year", {"classification": "fbs"}),
    Dataset("media", "/games/media", "year", {"classification": "fbs"}),
    Dataset("rankings", "/rankings", "year", reshape=poll_rows),
    Dataset("stats_game_havoc", "/stats/game/havoc", "year"),
    Dataset("punting_games", "/punting/teams/games", "year"),
    Dataset("games", "/games", "season_type", {"classification": "fbs"}),
    Dataset("lines", "/lines", "season_type"),
    Dataset("drives", "/drives", "season_type", {"classification": "fbs"}),
    Dataset("stats_game_advanced", "/stats/game/advanced", "season_type"),
    Dataset("ppa_games", "/ppa/games", "season_type"),
    Dataset("games_teams", "/games/teams", "week", {"classification": "fbs"}),
    Dataset("plays", "/plays", "week", {"classification": "fbs"}),
    Dataset("games_players", "/games/players", "week", {"classification": "fbs"}, reshape=box_rows),
    # Player level (P1). Rosters are as of the pull: the season's final roster.
    Dataset("roster", "/roster", "year"),
    Dataset("player_usage", "/player/usage", "year"),
    # Recruiting classes of the four prior years are on a season's roster.
    Dataset("recruiting_players", "/recruiting/players", "year", lag=4),
    Dataset("recruiting_players_juco", "/recruiting/players", "year", {"classification": "JUCO"}, lag=4),
    # Draft year Y is held in April of Y, before season Y.
    Dataset("draft_picks", "/draft/picks", "year"),
    # Season totals per player; season Y is a prior for Y+1, so one lag year (2013 for 2014).
    Dataset("ppa_players_season", "/ppa/players/season", "year", lag=1),
    # Per player and game. /ppa/players/games has no gameId (joined in core).
    Dataset("ppa_players_games", "/ppa/players/games", "week"),
    Dataset("player_success_games", "/stats/player/success/game", "week"),
    Dataset("passing_players_games", "/passing/players/games", "week"),
    # Week N holds ratings after week N's games. The endpoint ignores seasonType.
    Dataset("ratings_elo_weekly", "/ratings/elo", "regular_week", ended_only=True),
    # One request per FBS team; the response is the team's full head-coach history.
    Dataset("coach_tenures", "/coaches/tenures", "team"),
]
BY_NAME = {d.name: d for d in DATASETS}
WEEK_GRAINS = ("week", "regular_week")


def current_season(now: datetime) -> int:
    """Season in progress or most recently finished: August starts a new one."""
    return now.year if now.month >= 8 else now.year - 1


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


def week_windows(store: Store, year: int) -> dict[tuple[str, int], tuple[datetime, datetime]]:
    """(season type, week) -> (start, end) in UTC from the cached calendar. Empty if unknown."""
    if not store.has("calendar", str(year)):
        return {}
    out = {}
    for row in store.read_raw("calendar", str(year)):
        try:
            start = datetime.fromisoformat(row["startDate"])
            end = datetime.fromisoformat(row["endDate"])
            out[(row["seasonType"], int(row["week"]))] = (start, end)
        except (KeyError, TypeError, ValueError):
            continue
    return out


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
    elif ds.grain == "regular_week":
        for wk in weeks["regular"]:
            yield (
                f"{year}_regular_{wk:02d}",
                {"year": year, "week": wk},
                {"season": year, "season_type": "regular", "week": wk},
            )


def years_for(ds: Dataset, years: list[int]) -> list[int]:
    """Requested seasons plus the dataset's lag and lead years."""
    lo, hi = min(years), max(years)
    return sorted(set(years) | set(range(lo - ds.lag, lo)) | set(range(hi + 1, hi + ds.lead + 1)))


def save(store: Store, name: str, key: str, payload, extra: dict, reshape=None) -> int:
    """Write raw and table. With `reshape`, the table holds the reshaped rows."""
    rows = store.write(name, key, payload, extra)
    if reshape is not None:
        frame = flatten(reshape(payload), extra)
        frame.to_parquet(store.table_path(name, key), index=False)
        rows = len(frame)
    return rows


def sync_asof(src: Path, dst: Path, only: set[tuple[str, str]] | None = None) -> int:
    """Copy dated rating copies from one data root to another. Returns files copied.

    Without `only`, files missing at the destination are copied. With `only`
    (dataset, key) pairs, exactly those are copied and replace older versions.
    """
    copied = 0
    for kind in ("raw", "tables"):
        base = Path(src) / kind
        for folder in sorted(base.glob(f"*{ASOF_SUFFIX}")) if base.exists() else []:
            for path in sorted(folder.iterdir()):
                key = path.name.split(".")[0]
                target = Path(dst) / kind / folder.name / path.name
                if (only is None and target.exists()) or (only is not None and (folder.name, key) not in only):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied += 1
    return copied


def run(
    client: CFBDClient,
    store: Store,
    years: list[int],
    names: list[str],
    force: bool = False,
    now: datetime | None = None,
    asof_archive: str | Path | None = None,
) -> list[str]:
    """Run the pull. Returns a list of failure descriptions (empty on full success)."""
    now = now or datetime.now(timezone.utc)
    live_from = current_season(now)
    failures: list[str] = []
    dated: set[tuple[str, str]] = set()
    selected = [BY_NAME[n] for n in names]

    if asof_archive:
        print(f"dated copies restored from archive: {sync_asof(Path(asof_archive), store.root)}")

    def pull(ds: Dataset, key: str, params: dict, extra: dict, live: bool = False):
        if not force and not live and store.has(ds.name, key):
            return
        try:
            payload = client.get(ds.path, **ds.params, **params)
            rows = save(store, ds.name, key, payload, extra, ds.reshape)
            print(f"  {ds.name}/{key}: {rows} rows")
            if live and payload:
                stamp = {"asof_date": now.strftime("%Y-%m-%d"), "asof_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
                dated_key = f"{key}_{now:%Y%m%d}"
                save(store, ds.name + ASOF_SUFFIX, dated_key, payload, {**extra, **stamp}, ds.reshape)
                dated.add((ds.name + ASOF_SUFFIX, dated_key))
        except CFBDError as exc:
            failures.append(f"{ds.name}/{key}: {exc}")
            print(f"  {ds.name}/{key}: FAILED {exc}", file=sys.stderr)

    for ds in selected:
        if ds.grain == "static":
            pull(ds, "all", {}, {})

    by_year: dict[int, list[Dataset]] = {}
    for ds in selected:
        if ds.grain not in ("static", "team"):
            for year in years_for(ds, years):
                by_year.setdefault(year, []).append(ds)
    for year in sorted(by_year):
        print(f"season {year}")
        todo = by_year[year]
        needs_weeks = any(d.grain in WEEK_GRAINS for d in todo)
        weeks = weeks_for(client, store, year, force) if needs_weeks else dict(FALLBACK_WEEKS)
        windows = week_windows(store, year) if needs_weeks else {}
        for ds in todo:
            for key, params, extra in tasks(ds, year, weeks):
                if ds.grain in WEEK_GRAINS:
                    window = windows.get((extra["season_type"], extra["week"]))
                    if window and window[1 if ds.ended_only else 0] > now:
                        continue  # not started, or not ended for ended_only
                    if ds.ended_only and not window and year >= live_from:
                        continue  # no calendar dates: cannot tell whether the week has ended
                pull(ds, key, params, extra, live=ds.asof and year >= live_from)

    team_sets = [d for d in selected if d.grain == "team"]
    if team_sets:
        teams: dict[int, str] = {}
        for year in sorted(years):
            if not store.has("teams_fbs", str(year)):
                pull(BY_NAME["teams_fbs"], str(year), {"year": year}, {"season": year})
            if store.has("teams_fbs", str(year)):
                teams.update({t["id"]: t["school"] for t in store.read_raw("teams_fbs", str(year))})
        print(f"teams: {len(teams)}")
        for ds in team_sets:
            for team_id, school in sorted(teams.items(), key=lambda kv: kv[1]):
                pull(ds, f"team_{team_id}", {"team": school}, {})

    if asof_archive and dated:
        print(f"dated copies saved to archive: {sync_asof(store.root, Path(asof_archive), dated)}")
    return failures


def coverage(store: Store) -> pd.DataFrame:
    """Rows per dataset (index) and season (columns), from Parquet metadata.

    Keys that do not start with a year are counted under "all".
    """
    import pyarrow.parquet as pq

    counts: dict[str, dict[str, int]] = {}
    base = store.root / "tables"
    for folder in sorted(base.iterdir()) if base.exists() else []:
        for path in sorted(folder.glob("*.parquet")):
            head = path.stem.split("_")[0]
            season = head if len(head) == 4 and head.isdigit() else "all"
            per = counts.setdefault(folder.name, {})
            per[season] = per.get(season, 0) + pq.ParquetFile(path).metadata.num_rows
    frame = pd.DataFrame.from_dict(counts, orient="index").fillna(0).astype(int)
    return frame[sorted(frame.columns)]


def to_markdown(frame: pd.DataFrame, first: str = "dataset") -> str:
    lines = [f"| {first} | " + " | ".join(map(str, frame.columns)) + " |"]
    lines.append("| --- |" + " ---: |" * len(frame.columns))
    for name, row in frame.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{v:,}" for v in row) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", type=int, default=2014)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    ap.add_argument("--data-dir", default=str(settings.data_dir()))
    ap.add_argument("--force", action="store_true", help="re-download cached pulls")
    ap.add_argument("--list", action="store_true", help="list dataset names and exit")
    ap.add_argument("--coverage", action="store_true", help="print rows per dataset and season, then exit")
    ap.add_argument(
        "--asof-archive",
        default=os.environ.get("CFB_ASOF_ARCHIVE", ""),
        help="durable folder for dated rating copies: restored into the data dir before the pull, "
        "new copies saved to it after (default: $CFB_ASOF_ARCHIVE)",
    )
    args = ap.parse_args(argv)

    if args.list:
        for d in DATASETS:
            print(f"{d.name:24s} {d.path:26s} per {d.grain}")
        return 0
    if args.coverage:
        print(to_markdown(coverage(Store(args.data_dir))))
        return 0
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(BY_NAME)
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        ap.error(f"unknown datasets: {', '.join(unknown)}")

    client = CFBDClient()
    failures = run(
        client,
        Store(args.data_dir),
        list(range(args.start, args.end + 1)),
        names,
        args.force,
        asof_archive=args.asof_archive or None,
    )
    print(f"done: {client.calls} API calls, {len(failures)} failures")
    for f in failures:
        print(f"  {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

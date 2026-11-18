"""Pull Sportradar NCAA football (v7) schedules and play-by-play into the local store.

Usage:
    python -m cfb.sportradar --estimate                  # schedules only, then the pending play-by-play calls
    python -m cfb.sportradar --start 2013 --end 2026 --max-calls 900
    python -m cfb.sportradar --coverage                  # rows per Sportradar dataset and season

Auth: the key comes from settings.secret("sportradar_api_key") (SPORTRADAR_API_KEY, then the
OS keyring) and is sent as `x-api-key`. In the cloud a proxy injects it, so None is normal there
and no key header is sent.

Limits: Sportradar documents trial keys at 1 request per second and 1,000 requests per rolling
30 days. The API returns no quota headers, so requests are spaced by `--interval` and a run
stops at `--max-calls` (every HTTP response counts, retries included). A quota error stops the
run at once.

Datasets (same Store layout as cfb.ingest):
    sr_hierarchy/all                 divisions, conferences and teams, as of the pull
    sr_schedule/<year>_REG           one row per game, bowls and playoff included
    sr_pbp/<year>_<type>_<game id>   raw game play-by-play; the table has one row per play
    sr_play_stats/<same key>         one row per play and statistics entry (player, stat type, fields)

Games pulled: status "closed", coverage "full", and at least one team in the hierarchy's FBS
division. The hierarchy is the current one, so teams that moved up to FBS recently also bring
their FCS-era games; filter by season at join time. A closed game's play-by-play is never
re-pulled; past schedules are final; the current season's schedules are re-pulled on every run.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

from . import settings
from .ingest import coverage, current_season, to_markdown
from .store import Store, flatten

BASE_URL = "https://api.sportradar.com/ncaafb"
VERSION = "v7"
LANG = "en"
# Bowls and playoff games are weeks of the REG schedule; PST answers 404 (checked 2013-2026).
SEASON_TYPES = ("REG",)
FIRST_SEASON = 2013  # the 2012 schedule returns 404
RETRY_STATUS = {429, 500, 502, 503, 504}
DATASETS = ("sr_hierarchy", "sr_schedule", "sr_pbp", "sr_play_stats")


class SportradarError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class StopRun(SportradarError):
    """The run cannot continue: call budget reached, quota used up, or the key rejected."""


class SportradarClient:
    def __init__(
        self,
        level: str = "trial",
        api_key: str | None = None,
        base_url: str = BASE_URL,
        session: requests.Session | None = None,
        interval: float = 1.2,
        max_calls: int | None = None,
        max_retries: int = 4,
        backoff: float = 2.0,
        timeout: float = 60.0,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.level = level
        self.api_key = api_key if api_key is not None else settings.secret("sportradar_api_key")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.interval = interval
        self.max_calls = max_calls
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None
        self.calls = 0
        self.throttled = 0  # 429 "Too Many Requests" answers (they count as calls)

    def url(self, path: str) -> str:
        return f"{self.base_url}/{self.level}/{VERSION}/{LANG}/{path.lstrip('/')}"

    def _wait(self) -> None:
        if self._last is not None:
            gap = self._last + self.interval - self._clock()
            if gap > 0:
                self._sleep(gap)
        self._last = self._clock()

    def get(self, path: str) -> Any:
        """GET `path` (relative to /ncaafb/<level>/v7/en/) and return parsed JSON."""
        url = self.url(path)
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        last = ""
        for attempt in range(self.max_retries + 1):
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise StopRun(f"call budget of {self.max_calls} reached")
            self._wait()
            try:
                resp = self.session.get(url, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                last = repr(exc)
            else:
                self.calls += 1
                if resp.status_code == 200:
                    return resp.json()
                body = resp.text[:300]
                last = f"HTTP {resp.status_code}: {body}"
                if resp.status_code == 403 and "Authentication" in body:
                    raise StopRun(f"GET {path}: key rejected. {last}", resp.status_code)
                if resp.status_code == 429 and "Limit Exceeded" in body:
                    raise StopRun(f"GET {path}: quota used up. {last}", resp.status_code)
                if resp.status_code == 429:
                    self.throttled += 1
                if resp.status_code not in RETRY_STATUS:
                    raise SportradarError(f"GET {path} failed. {last}", resp.status_code)
            if attempt < self.max_retries:
                self._sleep(self.backoff * 2**attempt)
        raise SportradarError(f"GET {path} failed after {self.max_retries + 1} attempts. {last}")


# --- reshaping -------------------------------------------------------------------------------


def hierarchy_teams(payload: dict) -> list[dict]:
    """One row per team: division, conference, subdivision and team fields."""
    rows = []
    for div in payload.get("divisions") or []:
        for conf in div.get("conferences") or []:
            groups = [(None, conf.get("teams") or [])]
            groups += [(sub, sub.get("teams") or []) for sub in conf.get("subdivisions") or []]
            for sub, teams in groups:
                for team in teams:
                    rows.append(
                        {
                            "division": div.get("alias"),
                            "conference_id": conf.get("id"),
                            "conference": conf.get("alias") or conf.get("name"),
                            "subdivision": None if sub is None else sub.get("alias") or sub.get("name"),
                            **team,
                        }
                    )
    return rows


def fbs_team_ids(payload: dict) -> set[str]:
    return {t["id"] for t in hierarchy_teams(payload) if t.get("division") == "FBS" and t.get("id")}


def schedule_games(payload: dict) -> list[dict]:
    """One row per game with the week it belongs to."""
    season = payload.get("year")
    stype = payload.get("type")
    return [
        {"season": season, "season_type": stype, "week": wk.get("sequence"), "week_title": wk.get("title"), **game}
        for wk in payload.get("weeks") or []
        for game in wk.get("games") or []
    ]


def wanted(game: dict, fbs: set[str]) -> bool:
    """A finished game with play-by-play coverage and at least one FBS team."""
    teams = {(game.get("home") or {}).get("id"), (game.get("away") or {}).get("id")}
    return game.get("status") == "closed" and game.get("coverage") == "full" and bool(teams & fbs)


def _plays(payload: dict):
    """Yield (period, drive or None, play) for every event of type "play"."""
    for period in payload.get("periods") or []:
        for item in period.get("pbp") or []:
            if item.get("type") == "drive":
                for event in item.get("events") or []:
                    if event.get("type") == "play":
                        yield period, item, event
            elif item.get("type") == "play":
                yield period, None, item


def play_rows(payload: dict) -> list[dict]:
    """sr_pbp table: one row per play. `statistics` is in sr_play_stats; the raw file has the rest."""
    rows = []
    for period, drive, play in _plays(payload):
        drive = drive or {}
        rows.append(
            {
                "game_id": payload.get("id"),
                "period_type": period.get("period_type"),
                "period": period.get("number"),
                "drive_id": drive.get("id"),
                "drive_sequence": drive.get("sequence"),
                "drive_offense_id": (drive.get("offensive_team") or {}).get("id"),
                "drive_defense_id": (drive.get("defensive_team") or {}).get("id"),
                "drive_start_reason": drive.get("start_reason"),
                "drive_end_reason": drive.get("end_reason"),
                **{k: v for k, v in play.items() if k != "statistics"},
            }
        )
    return rows


def play_stat_rows(payload: dict) -> list[dict]:
    """sr_play_stats table: one row per play and statistics entry, with the player and team."""
    rows = []
    for _, _, play in _plays(payload):
        for stat in play.get("statistics") or []:
            player = stat.get("player") or {}
            team = stat.get("team") or {}
            rows.append(
                {
                    "game_id": payload.get("id"),
                    "play_id": play.get("id"),
                    "play_type": play.get("play_type"),
                    "player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "player_position": player.get("position"),
                    "player_jersey": player.get("jersey"),
                    "team_id": team.get("id"),
                    "team_alias": team.get("alias"),
                    **{k: v for k, v in stat.items() if k not in ("player", "team")},
                }
            )
    return rows


def _table(store: Store, dataset: str, key: str, rows: list[dict], extra: dict) -> int:
    frame = flatten(rows, extra)
    out = store.table_path(dataset, key)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    return len(frame)


# --- run -------------------------------------------------------------------------------------


@dataclass
class Result:
    failures: list[str] = field(default_factory=list)
    pending: dict[str, int] = field(default_factory=dict)  # "<year>_<type>" -> games still to pull
    pulled: int = 0
    stopped: str | None = None

    @property
    def pending_total(self) -> int:
        return sum(self.pending.values())


def run(
    client: SportradarClient,
    store: Store,
    years: list[int],
    force: bool = False,
    estimate: bool = False,
    now: datetime | None = None,
    log=print,
) -> Result:
    """Pull the hierarchy, schedules and (unless `estimate`) play-by-play of the wanted games."""
    now = now or datetime.now(timezone.utc)
    live_from = current_season(now)
    result = Result()

    def fetch(dataset: str, key: str, path: str, reshape, extra: dict):
        payload = client.get(path)
        store.write(dataset, key, payload, extra)
        rows = _table(store, dataset, key, reshape(payload), extra)
        log(f"  {dataset}/{key}: {rows} rows")
        return payload

    try:
        if force or not store.has("sr_hierarchy", "all"):
            fetch("sr_hierarchy", "all", "league/hierarchy.json", hierarchy_teams, {})
        fbs = fbs_team_ids(store.read_raw("sr_hierarchy", "all"))
        log(f"FBS teams in hierarchy: {len(fbs)}")

        todo: list[tuple[str, dict, dict]] = []
        for year in sorted(years):
            for stype in SEASON_TYPES:
                key = f"{year}_{stype}"
                if force or year >= live_from or not store.has("sr_schedule", key):
                    try:
                        fetch("sr_schedule", key, f"games/{year}/{stype}/schedule.json", schedule_games, {})
                    except StopRun:
                        raise
                    except SportradarError as exc:  # 404: no schedule for that season type
                        if exc.status == 404:
                            log(f"  sr_schedule/{key}: none (404)")
                            continue
                        result.failures.append(f"sr_schedule/{key}: {exc}")
                        log(f"  sr_schedule/{key}: FAILED {exc}")
                        if not store.has("sr_schedule", key):
                            continue
                games = [g for g in schedule_games(store.read_raw("sr_schedule", key)) if wanted(g, fbs)]
                missing = [g for g in games if force or not store.has("sr_pbp", f"{key}_{g['id']}")]
                result.pending[key] = len(missing)
                log(f"{key}: {len(games)} games with play-by-play, {len(missing)} to pull")
                todo += [(f"{key}_{g['id']}", g, {"season": year, "season_type": stype, "week": g.get("week")}) for g in missing]

        if estimate:
            return result

        for key, game, extra in todo:
            try:
                payload = fetch("sr_pbp", key, f"games/{game['id']}/pbp.json", play_rows, extra)
                _table(store, "sr_play_stats", key, play_stat_rows(payload), extra)
                result.pulled += 1
                result.pending[key.rsplit("_", 1)[0]] -= 1
            except StopRun:
                raise
            except SportradarError as exc:
                result.failures.append(f"sr_pbp/{key}: {exc}")
                log(f"  sr_pbp/{key}: FAILED {exc}")
    except StopRun as exc:
        result.stopped = str(exc)
        log(f"stopped: {exc}")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    now = datetime.now(timezone.utc)
    ap.add_argument("--start", type=int, default=FIRST_SEASON)
    ap.add_argument("--end", type=int, default=current_season(now))
    ap.add_argument("--level", default="trial", choices=("trial", "production"))
    ap.add_argument("--data-dir", default=str(settings.data_dir()))
    ap.add_argument("--interval", type=float, default=1.2, help="seconds between requests (trial: 1 per second)")
    ap.add_argument("--max-calls", type=int, default=900, help="stop after this many requests; 0 = no limit")
    ap.add_argument("--estimate", action="store_true", help="pull hierarchy and schedules only, then report the pending calls")
    ap.add_argument("--force", action="store_true", help="re-download cached pulls")
    ap.add_argument("--coverage", action="store_true", help="print rows per Sportradar dataset and season, then exit")
    args = ap.parse_args(argv)

    store = Store(args.data_dir)
    if args.coverage:
        frame = coverage(store)
        print(to_markdown(frame.loc[[d for d in DATASETS if d in frame.index]]))
        return 0

    client = SportradarClient(level=args.level, interval=args.interval, max_calls=args.max_calls or None)
    result = run(client, store, list(range(args.start, args.end + 1)), args.force, args.estimate, now)
    print(
        f"done: {client.calls} API calls ({client.throttled} throttled), {result.pulled} games pulled, "
        f"{result.pending_total} games pending, {len(result.failures)} failures"
    )
    for f in result.failures:
        print(f"  {f}", file=sys.stderr)
    if result.stopped:
        print(f"stopped early: {result.stopped}", file=sys.stderr)
    return 1 if result.failures or result.stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())

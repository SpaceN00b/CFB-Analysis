"""Pull the newest polls between weekly runs and publish them, without re-running the week.

    python -m cfb.poll_update --publish /mnt/project-files/reports \\
        --polls-archive /mnt/project-files/cfbd-data/polls

Made for Tuesday nights: the playoff committee (CFP) ranking comes out Tuesday evening ET,
after Monday's weekly run (cfb.run_week) has built and published the week's report.

  1. data      download the current published warehouse version from R2 (cfb.warehouse_sync)
               into --work and copy it to a working data dir (or use --data-dir as it is)
  2. pull      /rankings for the season, one API call, into that data dir
  3. rankings  rebuild the rankings section of the published report (reports/<key>/bundle.json)
               at the report's own cutoff and re-render report.md and dashboard.html. A week's
               polls count as known at the week start (cfb.core.policies), so the Tuesday CFP
               ranking shows in that week's report. Predictions, features and the model's
               ratings are not touched.
  4. publish   the R2 warehouse (only changed files upload); the same rankings refresh on the
               report in --publish/<key>/ (and latest/ when latest/KEY is that key), each bundle
               patched in place; and a small tar of the season's rankings files to
               --polls-archive (dated name, LATEST pointer)

When the pull holds no poll week the report does not already have, steps 3 and 4 are skipped.
A short summary is printed at the end.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import warehouse_sync as ws
from .client import CFBDClient
from .core.asof import Warehouse
from .ingest import current_season
from .ingest import run as ingest_run
from .report.bundle import write_bundle
from .report.config import ReportConfig, load_config
from .report.html import render_dashboard
from .report.markdown import render_markdown
from .report.rankings import POLLS, build_rankings
from .store import Store

DATASET = "rankings"
REPORT_FILES = ("bundle.json", "report.md", "dashboard.html")
NOTE_PREFIX = "Polls refreshed"
POLL_SHORT = {name: short for _, name, short in POLLS}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


@dataclass
class PollUpdate:
    season: int
    key: str | None = None
    new_polls: list[str] = field(default_factory=list)
    report_updated: bool = False
    warehouse: object | None = None  # warehouse_sync.PublishResult
    warehouse_error: str | None = None
    published: Path | None = None
    archive: Path | None = None
    rankings: dict | None = None
    notes: list[str] = field(default_factory=list)


def poll_weeks(store: Store, season: int) -> set[tuple[str, int, str]]:
    """(season type, week, poll) of every poll in the season's rankings table."""
    path = store.table_path(DATASET, str(season))
    if not path.exists():
        return set()
    t = pd.read_parquet(path, columns=["seasonType", "week", "poll"])
    return {(str(a), int(b), str(c)) for a, b, c in t.drop_duplicates().itertuples(index=False)}


def poll_label(season_type: str, week: int, poll: str) -> str:
    when = "final" if season_type == "postseason" else f"week {week}"
    return f"{POLL_SHORT.get(poll, poll)} {when}"


def fetch_data_dir(work: Path, store: ws.Store) -> tuple[Path, str | None]:
    """Sync the bucket's current version into work/r2 and copy it to work/data (the synced
    version's files are hard-link targets of later syncs and must stay unchanged)."""
    r = ws.sync(work / "r2", store, log=log)
    data = work / "data"
    if data.exists():
        shutil.rmtree(data)
    shutil.copytree(r.data_dir, data)
    manifest = ws.local_manifest(work / "r2", r.version) or {}
    log(f"  warehouse {r.version} copied to {data}")
    return data, manifest.get("key")


def refresh_report(folder: Path, warehouse: Warehouse, config: ReportConfig, now: datetime,
                   new_polls: list[str], cache: dict) -> dict | None:
    """Rebuild the rankings section of the report in `folder` (bundle.json, at the bundle's own
    cutoff) and re-render its report.md and dashboard.html. Every other bundle field stays."""
    path = folder / "bundle.json"
    if not path.exists() or not config.rankings_source:
        return None
    bundle = json.loads(path.read_text(encoding="utf-8"))
    asof = bundle["meta"]["asof_utc"]
    if asof not in cache:
        cache[asof] = build_rankings(warehouse, pd.Timestamp(asof), config.rankings_source, config.rankings_top)
    bundle["rankings"] = cache[asof]
    meta = bundle["meta"]
    notes = [n for n in meta.get("notes") or [] if not n.startswith(NOTE_PREFIX)]
    notes.append(f"{NOTE_PREFIX} {now:%Y-%m-%d %H:%M} UTC ({', '.join(new_polls)}); predictions and model "
                 f"ratings are from the weekly run of {meta.get('created_local') or meta.get('created_utc')}.")
    meta["notes"] = notes
    write_bundle(bundle, path)
    (folder / "report.md").write_text(render_markdown(bundle), encoding="utf-8")
    (folder / "dashboard.html").write_text(render_dashboard(bundle), encoding="utf-8")
    return cache[asof]


def publish_reports(data_dir: Path, key: str, dest_root: Path, refresh) -> Path:
    """Refresh dest_root/<key>/ in place, and latest/ when latest/KEY names this report. A
    report missing there gets the data dir's three files (not metrics/)."""
    dest = dest_root / key
    if (dest / "bundle.json").exists():
        refresh(dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        for name in REPORT_FILES:
            if (data_dir / "reports" / key / name).exists():
                shutil.copy2(data_dir / "reports" / key / name, dest / name)
    kfile = dest_root / "latest" / "KEY"
    if kfile.exists() and kfile.read_text(encoding="utf-8").strip() == key:
        refresh(dest_root / "latest")
    return dest


def archive_polls(data_dir: Path, season: int, dest: Path, now: datetime) -> Path:
    """data/raw/rankings/<season>.json.gz and data/tables/rankings/<season>.parquet in a dated
    tar; extract it over a restored snapshot (later files replace earlier ones)."""
    store = Store(data_dir)
    members = [p for p in (store.raw_path(DATASET, str(season)), store.table_path(DATASET, str(season))) if p.exists()]
    dest.mkdir(parents=True, exist_ok=True)
    name = f"polls_{season}_{now:%Y%m%dT%H%M%SZ}.tar"
    tmp = dest / (name + ".tmp")
    with tarfile.open(tmp, "w") as tar:
        for p in members:
            tar.add(p, arcname=f"data/{p.relative_to(data_dir).as_posix()}")
    tmp.replace(dest / name)
    (dest / "LATEST").write_text(name + "\n", encoding="utf-8")
    return dest / name


def summary(res: PollUpdate, top: int = 10) -> str:
    lines = [f"Poll update, season {res.season}" + (f", report {res.key}" if res.key else "")]
    if not res.new_polls:
        lines.append("No new poll since the report: nothing changed or published.")
        return "\n".join(lines + res.notes)
    lines.append(f"New: {', '.join(res.new_polls)}.")
    rk = res.rankings
    if rk:
        for p in rk["polls"]:
            c = rk["comparisons"][p["key"]]
            rho = "" if c["spearman"] is None else f", rank correlation {c['spearman']:.2f}"
            lines.append(f"- {poll_label(p['season_type'], p['week'], p['name'])}: same top 25 as the model "
                         f"{c['overlap']}/25{rho}")
        cfp = [t for t in rk["teams"] if t["polls"].get("cfp")]
        if cfp:
            cfp.sort(key=lambda t: t["polls"]["cfp"])
            lines.append(f"CFP top {min(top, len(cfp))} (model rank): " + "; ".join(
                f"{t['polls']['cfp']} {t['team']} ({t['rank'] or '-'})" for t in cfp[:top]))
    if res.warehouse is not None:
        w = res.warehouse
        lines.append(f"R2: {w.version}" + (f", {w.uploaded_files} files uploaded" if w.changed else ", unchanged"))
    if res.warehouse_error:
        lines.append(f"R2 publish failed: {res.warehouse_error}")
    if res.published:
        lines.append(f"Report files: {res.published}")
    if res.archive:
        lines.append(f"Polls archive: {res.archive}")
    return "\n".join(lines + res.notes)


def run(*, data_dir: Path | None = None, work: Path | None = None, season: int | None = None,
        client=None, store: ws.Store | None = None, config: ReportConfig | None = None,
        publish_dir: Path | None = None, archive_dir: Path | None = None, warehouse: bool = True,
        now: datetime | None = None) -> PollUpdate:
    now = now or datetime.now(timezone.utc)
    season = season or current_season(now)
    config = config or load_config()
    res = PollUpdate(season)
    need_store = warehouse or data_dir is None
    if need_store and store is None:
        store = ws.open_store(ws.r2_settings())

    log("step 1/4 data")
    key = None
    if data_dir is None:
        data_dir, key = fetch_data_dir(Path(work or Path(tempfile.gettempdir()) / "cfb_poll_update"), store)
    data_dir = Path(data_dir)
    key = key or ws.latest_report_key(data_dir)
    res.key = key
    dstore = Store(data_dir)
    before = poll_weeks(dstore, season)

    log(f"step 2/4 pull /rankings {season}")
    failures = ingest_run(client or CFBDClient(), dstore, [season], [DATASET], force=True, now=now)
    if failures:
        raise SystemExit("rankings pull failed: " + "; ".join(failures))
    added = sorted(poll_weeks(dstore, season) - before, key=lambda t: (t[0] == "postseason", t[1], t[2]))
    shown = {name for _, name, _ in POLLS}
    res.new_polls = [poll_label(*t) for t in added if t[2] in shown]
    if not res.new_polls:
        log("  no new AP, Coaches or CFP week: done")
        return res
    log(f"  new: {', '.join(res.new_polls)}")

    log("step 3/4 rankings")
    wh, cache = Warehouse(data_dir), {}

    def refresh(folder: Path) -> dict | None:
        return refresh_report(folder, wh, config, now, res.new_polls, cache)

    if key:
        res.rankings = refresh(data_dir / "reports" / key)
        res.report_updated = res.rankings is not None
    if not res.report_updated:
        res.notes.append("No report rankings section to refresh (no report, or rankings are off in the config).")

    log("step 4/4 publish")
    if warehouse:
        try:
            res.warehouse = ws.publish(data_dir, store, key, now=now, quota=ws.cfbd_quota(), commit=ws.git_commit(),
                                       log=log)
        except Exception as exc:  # report files and the archive still go out
            res.warehouse_error = f"{type(exc).__name__}: {exc}"
    if publish_dir is not None and res.report_updated:
        res.published = publish_reports(data_dir, key, Path(publish_dir), refresh)
    if archive_dir is not None:
        res.archive = archive_polls(data_dir, season, Path(archive_dir), now)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cfb.poll_update", description=__doc__.split("\n")[0])
    ap.add_argument("--season", type=int, help="default: the current season")
    ap.add_argument("--data-dir", help="use this complete data dir instead of downloading the R2 warehouse")
    ap.add_argument("--work", help="folder for the downloaded warehouse (default: <temp>/cfb_poll_update)")
    ap.add_argument("--publish", help="reports folder to update, e.g. /mnt/project-files/reports")
    ap.add_argument("--polls-archive", help="folder for the dated polls tar, e.g. /mnt/project-files/cfbd-data/polls")
    ap.add_argument("--no-warehouse", action="store_true", help="do not publish to R2")
    ap.add_argument("--config", help="report config (default: dashboard/config.toml)")
    args = ap.parse_args(argv)
    try:
        res = run(data_dir=Path(args.data_dir) if args.data_dir else None, work=Path(args.work) if args.work else None,
                  season=args.season, config=load_config(args.config), publish_dir=args.publish,
                  archive_dir=args.polls_archive, warehouse=not args.no_warehouse)
    except ws.WarehouseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(summary(res))
    return 1 if res.warehouse_error else 0


if __name__ == "__main__":
    sys.exit(main())

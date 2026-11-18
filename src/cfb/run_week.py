"""Weekly run: refresh data, rebuild features, predict the week, score last week and the
season so far, write the report and the dashboard.

    python -m cfb.run_week                          # the current calendar week
    python -m cfb.run_week --season 2026 --week 12  # a given week
    python -m cfb.run_week --season 2025 --week 5 --no-refresh   # a past week, from local data
    python -m cfb.run_week --publish /mnt/project-files/reports

Steps, each logged:
  1. data      restore the snapshot tar into the data dir if it holds no raw data (--snapshot),
               and the stored-features tar if data/features is empty (--features-snapshot)
  2. refresh   pull the target season from the API with --force and the dated-copy archive
               (--asof-archive; refuses to run without one unless --allow-no-archive).
               Skipped for past seasons and with --no-refresh.
  3. core      rebuild data/core (always after a refresh)
  4. features  unit ratings and every feature group for the season (config run.build_features);
               other seasons' rows in data/features are kept, seasons missing there are built
  5. predict   every model in the config for the season through the target week; built-in
               baselines and producers write data/predictions/<model_id>/
  6. report    data/reports/<season>_<type>_<week>/: report.md, dashboard.html, bundle.json, summary.md,
               metrics/<model_id>.json (season to date, FBS vs FBS)
  7. publish   copy the report folder to --publish/<key>/ and the newest files to --publish/latest/

The model the report is about and its baselines come from dashboard/config.toml; switching
from a baseline to the model is a change of `primary` there (or --primary).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .core.asof import Warehouse
from .core.build import build_core
from .core.schemas import read_predictions, validate_metrics, validate_predictions
from .eval.metrics import evaluate
from .eval.splits import write_weekly
from .report.bundle import build_bundle, write_bundle
from .report.config import DEFAULT_CONFIG, ReportConfig, load_config
from .report.html import render_dashboard
from .report.markdown import render_markdown
from .report.summary import render_summary
from .report.slate import Slate, known_results, resolve_slate

PROJECT_FILES = Path("/mnt/project-files")
DEFAULT_ARCHIVE = PROJECT_FILES / "cfbd-data" / "asof"
DEFAULT_SNAPSHOT_DIR = PROJECT_FILES / "cfbd-data"


@dataclass
class RunResult:
    slate: Slate
    out_dir: Path
    files: dict[str, Path]
    notes: list[str] = field(default_factory=list)
    published: Path | None = None


def log(msg: str) -> None:
    print(f"[run_week {datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


# ---- steps --------------------------------------------------------------------------------

def default_snapshot() -> Path | None:
    latest = DEFAULT_SNAPSHOT_DIR / "LATEST"
    if latest.exists():
        p = DEFAULT_SNAPSHOT_DIR / latest.read_text().strip()
        return p if p.exists() else None
    return None


def ensure_data(data_dir: Path, snapshot: Path | None) -> bool:
    """Restore the snapshot when the data dir has no raw data. Returns True if restored."""
    if (data_dir / "raw").exists() and any((data_dir / "raw").iterdir()):
        return False
    if snapshot is None or not Path(snapshot).exists():
        raise SystemExit(f"{data_dir}/raw is empty and no snapshot was found; pass --snapshot PATH")
    log(f"restoring {snapshot} (the tar holds data/; extracting next to {data_dir})")
    target = data_dir.resolve().parent
    with tarfile.open(snapshot) as tar:
        members = [m for m in tar.getmembers() if m.name == "data" or m.name.startswith("data/")]
        try:
            tar.extractall(target, members=members, filter="data")
        except TypeError:  # Python without extraction filters
            tar.extractall(target, members=members)
    if data_dir.resolve() != (target / "data"):
        raise SystemExit(f"the snapshot restores to {target / 'data'}; use --data-dir {target / 'data'}")
    return True


def refresh(data_dir: Path, season: int, archive: Path | None, now: datetime) -> list[str]:
    """Pull the season from the API. Returns failure descriptions."""
    from .client import CFBDClient
    from .ingest import BY_NAME, run as ingest_run
    from .store import Store

    client = CFBDClient()
    failures = ingest_run(client, Store(data_dir), [season], list(BY_NAME), force=True, now=now,
                          asof_archive=archive)
    log(f"refresh: {client.calls} API calls, {len(failures)} failures")
    return failures


def default_features_snapshot() -> Path | None:
    latest = DEFAULT_SNAPSHOT_DIR / "FEATURES_LATEST"
    if latest.exists():
        p = DEFAULT_SNAPSHOT_DIR / latest.read_text().strip()
        return p if p.exists() else None
    return None


def ensure_features(data_dir: Path, snapshot: Path | None) -> bool:
    """Restore the stored-features tar (data/features/ for every season) when the data dir
    has no feature files. The model trains on past seasons' stored features, and building
    them from scratch takes about 40 minutes. Returns True if restored."""
    feat = data_dir / "features"
    if feat.exists() and any(feat.glob("*.parquet")):
        return False
    if snapshot is None or not Path(snapshot).exists():
        return False
    log(f"restoring stored features from {snapshot}")
    target = data_dir.resolve().parent
    with tarfile.open(snapshot) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("data/features")]
        try:
            tar.extractall(target, members=members, filter="data")
        except TypeError:  # Python without extraction filters
            tar.extractall(target, members=members)
    return True


def missing_seasons(warehouse: Warehouse, group_name: str, through: int) -> list[int]:
    """Seasons up to `through` with scheduled games but no rows in the stored feature file."""
    sched = warehouse.game_index
    want = set(sched.loc[sched["season"] <= through, "season"].astype(int))
    path = warehouse.root / "features" / f"{group_name}.parquet"
    if not path.exists():
        return sorted(want)
    ids = pd.read_parquet(path, columns=["game_id"])["game_id"]
    have = set(sched.loc[sched["game_id"].isin(ids), "season"].astype(int))
    return sorted(want - have)


def build_features_for(warehouse: Warehouse, season: int) -> None:
    """Unit ratings, then every feature group for `season`. Rows of other seasons in the
    stored files are kept (the model's training rows); seasons missing from a file are
    built too."""
    from .features.base import discover_groups
    from .features.build import build_derived, build_features

    build_derived(warehouse)
    groups = discover_groups()
    plan: dict[tuple[int, ...], list] = {}
    for g in groups:
        seasons = tuple(sorted(set(missing_seasons(warehouse, g.name, season)) | {season}))
        plan.setdefault(seasons, []).append(g)
    for seasons, gs in plan.items():
        if len(seasons) > 1:
            log(f"features: {', '.join(g.name for g in gs)} also need seasons {seasons[0]}-{seasons[-2]} "
                "(missing from data/features); this takes a while")
        build_features(warehouse, gs, list(seasons))


def produce_predictions(warehouse: Warehouse, slate: Slate, config: ReportConfig, data_dir: Path,
                        write: bool = True) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Predictions of every configured model for the season through the target week."""
    sched = warehouse.schedule([slate.season])
    games = sched[sched["cutoff_utc"] <= slate.last_cutoff].reset_index(drop=True)
    out, notes = {}, []
    for model_id in config.models:
        producer = config.producer(model_id)
        t0 = time.time()
        try:
            if producer is not None:
                frame = producer(warehouse, games)
                if frame is None or frame.empty:
                    frame = pd.DataFrame()
                else:
                    validate_predictions(frame[[c for c in frame.columns if c not in ("season", "season_type", "week")]])
                    if write:
                        write_weekly(frame, data_dir, replace=False)
                source = "produced"
            else:
                frame = read_predictions(data_dir, model_id)
                frame = frame[frame["game_id"].isin(games["game_id"])]
                source = "read from data/predictions"
        except Exception as exc:  # a baseline failing should not stop the report
            if model_id == config.primary:
                raise
            notes.append(f"{model_id} could not be produced: {exc}")
            log(f"predict {model_id}: FAILED {exc}")
            continue
        n_week = int(frame["game_id"].isin(slate.games["game_id"]).sum()) if len(frame) else 0
        log(f"predict {model_id}: {len(frame)} predictions ({n_week} this week), {source}, {time.time() - t0:.1f}s")
        if not len(frame):
            notes.append(f"No predictions for {model_id}.")
            if model_id == config.primary:
                raise SystemExit(f"the primary model {model_id} has no predictions for {slate.label}")
        out[model_id] = frame
    return out, notes


def apply_qb_out(warehouse: Warehouse, slate: Slate, config: ReportConfig,
                 frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Manual QB-out override ([qb_out] in the config) on the target week's games of the
    primary model. Stored predictions stay the model's own."""
    from .model import qb_out as QO
    view = warehouse.view(slate.last_cutoff)
    adj = QO.resolve(view, config.qb_out)
    week = frame["game_id"].isin(slate.games["game_id"])
    sched = warehouse.schedule([slate.season])
    frame = pd.concat([frame[~week], QO.apply(frame[week], sched, adj)]).loc[frame.index]
    parts = [f"{a['starter_name']} out, {a['replacement_name']} starts ({a['points']:+.1f} pts)"
             for a in adj.values() if "starter_name" in a]
    log(f"qb_out: {'; '.join(parts) or 'no change'}")
    return frame, ("Manual QB-out override applied to " + config.primary + ": " + "; ".join(parts) + "."
                   if parts else "Manual QB-out override set, but no expected starter was found.")


def lines_time(data_dir: Path, season: int) -> pd.Timestamp | None:
    paths = list((data_dir / "raw" / "lines").glob(f"{season}_*.json.gz"))
    if not paths:
        return None
    return pd.Timestamp(max(p.stat().st_mtime for p in paths), unit="s", tz="UTC")


def write_report(warehouse: Warehouse, slate: Slate, predictions: dict, config: ReportConfig, data_dir: Path,
                 out_dir: Path, notes: list[str], lines_asof, created: pd.Timestamp) -> dict[str, Path]:
    bundle = build_bundle(warehouse, slate, predictions, config, data_dir=data_dir, created_utc=created,
                          lines_asof=lines_asof, run_notes=notes)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "bundle": write_bundle(bundle, out_dir / "bundle.json"),
        "report": out_dir / "report.md",
        "dashboard": out_dir / "dashboard.html",
        "summary": out_dir / "summary.md",
    }
    files["report"].write_text(render_markdown(bundle), encoding="utf-8")
    files["summary"].write_text(render_summary(bundle), encoding="utf-8")
    files["dashboard"].write_text(render_dashboard(bundle), encoding="utf-8")
    # Season-to-date metrics documents (cfb.core.schemas contract). They go with the report,
    # not to data/eval/, which holds the 2018-2025 validation documents.
    known = known_results(warehouse.outcomes(), slate.asof)
    for model_id, frame in predictions.items():
        season_preds = frame[frame["game_id"].isin(known["game_id"])]
        if season_preds.empty:
            continue
        doc = validate_metrics(evaluate(season_preds, known, model_id, created_utc=created,
                                        scope_label=f"fbs_vs_fbs; {slate.season} through the {slate.label} cutoff"))
        path = out_dir / "metrics" / f"{model_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        files[f"metrics_{model_id}"] = path
    return files


def publish(out_dir: Path, dest_root: Path, key: str) -> Path:
    dest = dest_root / key
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(out_dir, dest)
    latest = dest_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for name in ("report.md", "dashboard.html", "bundle.json", "summary.md"):
        if (out_dir / name).exists():
            shutil.copy2(out_dir / name, latest / name)
    (latest / "KEY").write_text(key + "\n", encoding="utf-8")
    return dest


# ---- driver ---------------------------------------------------------------------------------

def run(data_dir: str | Path = "data", config: ReportConfig | None = None, season: int | None = None,
        season_type: str | None = None, week: int | None = None, now=None, do_refresh: bool | None = None,
        archive: Path | None = None, allow_no_archive: bool = False, snapshot: Path | None = None,
        build_features: bool | None = None, out_dir: Path | None = None, publish_dir: Path | None = None,
        write_predictions: bool = True, features_snapshot: Path | None = None) -> RunResult:
    data_dir = Path(data_dir)
    config = config or load_config()
    now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    created = pd.Timestamp.now(tz="UTC")
    notes: list[str] = []

    log("step 1/7 data")
    ensure_data(data_dir, snapshot)
    ensure_features(data_dir, features_snapshot)
    core_missing = not (data_dir / "core" / "game_index.parquet").exists()
    if core_missing:
        build_core(data_dir)
    warehouse = Warehouse(data_dir)
    from .ingest import current_season
    live_season = current_season(now_ts.to_pydatetime())

    def target(wh: Warehouse) -> Slate:
        # `now` picks the week when none is given, and the postseason window of a live season.
        use_now = season is None or week is None or season >= live_season
        return resolve_slate(wh, season, season_type, week, now=now_ts if use_now else None)

    slate = target(warehouse)
    log(f"target: {slate.label} ({len(slate.games)} games), report as of {slate.asof}")

    live = slate.season >= live_season
    if do_refresh is None:
        do_refresh = live
    lines_asof = None
    if do_refresh:
        log("step 2/7 refresh")
        if archive is None and not allow_no_archive:
            raise SystemExit("refresh needs the dated-copy archive (--asof-archive DIR), because CFBD's weekly "
                             "SP+/FPI/SRS/CORE values cannot be pulled again later; pass --allow-no-archive to skip it")
        failures = refresh(data_dir, slate.season, archive, now_ts.to_pydatetime())
        if failures:
            notes.append(f"Data refresh had {len(failures)} failed requests; the report uses what was pulled.")
        lines_asof = now_ts
        log("step 3/7 core")
        build_core(data_dir)
        warehouse = Warehouse(data_dir)
        slate = target(warehouse)
    else:
        log("step 2/7 refresh: skipped")
        log("step 3/7 core: " + ("built" if core_missing else "kept"))
    if live:
        lines_asof = lines_asof or lines_time(data_dir, slate.season)

    feats = config.build_features if build_features is None else build_features
    if feats:
        log("step 4/7 features")
        build_features_for(warehouse, slate.season)
        warehouse = Warehouse(data_dir)
    else:
        log("step 4/7 features: skipped")

    log("step 5/7 predict")
    predictions, pnotes = produce_predictions(warehouse, slate, config, data_dir, write=write_predictions)
    notes += pnotes
    if config.qb_out and config.primary in predictions:
        predictions[config.primary], qnote = apply_qb_out(warehouse, slate, config, predictions[config.primary])
        notes.append(qnote)
    if config.primary.startswith("baseline_"):
        notes.insert(0, f"The report's model is the {config.primary} baseline until the trained model is "
                        "switched on in dashboard/config.toml.")

    log("step 6/7 report")
    out = Path(out_dir) if out_dir is not None else data_dir / "reports" / slate.key
    files = write_report(warehouse, slate, predictions, config, data_dir, out, notes, lines_asof, created)
    for k, v in files.items():
        if not k.startswith("metrics_"):
            log(f"  {k}: {v}")
    published = None
    if publish_dir is not None:
        log("step 7/7 publish")
        published = publish(out, Path(publish_dir), slate.key)
        log(f"  published to {published} and {Path(publish_dir) / 'latest'}")
    else:
        log("step 7/7 publish: skipped (no --publish)")
    return RunResult(slate, out, files, notes, published)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cfb.run_week", description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--config", default=None, help=f"run configuration (default {DEFAULT_CONFIG})")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--season-type", choices=("regular", "postseason"))
    ap.add_argument("--now", help="pretend the run happens at this UTC time (ISO); picks the week when none is given")
    ap.add_argument("--primary", help="model_id the report is about (overrides the config)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--refresh", dest="refresh", action="store_true", default=None,
                   help="pull the season from the API (default for the current season)")
    g.add_argument("--no-refresh", dest="refresh", action="store_false")
    ap.add_argument("--asof-archive", default=os.environ.get("CFB_ASOF_ARCHIVE")
                    or (str(DEFAULT_ARCHIVE) if DEFAULT_ARCHIVE.exists() else None),
                    help="dated rating copies folder (default $CFB_ASOF_ARCHIVE or the project files archive)")
    ap.add_argument("--allow-no-archive", action="store_true")
    ap.add_argument("--snapshot", default=None, help="tar to restore when the data dir is empty "
                    "(default: the LATEST snapshot in project files)")
    ap.add_argument("--features-snapshot", default=None, help="stored-features tar to restore when data/features "
                    "is empty (default: FEATURES_LATEST in project files)")
    f = ap.add_mutually_exclusive_group()
    f.add_argument("--features", dest="features", action="store_true", default=None)
    f.add_argument("--no-features", dest="features", action="store_false")
    ap.add_argument("--out", default=None, help="report folder (default data/reports/<season>_<type>_<week>)")
    ap.add_argument("--publish", default=None, help="copy the report here, e.g. /mnt/project-files/reports")
    ap.add_argument("--no-write-predictions", action="store_true", help="do not write data/predictions files")
    args = ap.parse_args(argv)

    if args.week is not None and args.season is None:
        ap.error("--week needs --season")
    config = load_config(args.config, primary=args.primary)
    snapshot = Path(args.snapshot) if args.snapshot else default_snapshot()
    feat_snapshot = Path(args.features_snapshot) if args.features_snapshot else default_features_snapshot()
    result = run(
        data_dir=args.data_dir, config=config, season=args.season, season_type=args.season_type, week=args.week,
        now=args.now, do_refresh=args.refresh, archive=Path(args.asof_archive) if args.asof_archive else None,
        allow_no_archive=args.allow_no_archive, snapshot=snapshot, build_features=args.features,
        features_snapshot=feat_snapshot,
        out_dir=Path(args.out) if args.out else None, publish_dir=Path(args.publish) if args.publish else None,
        write_predictions=not args.no_write_predictions,
    )
    for n in result.notes:
        log(f"note: {n}")
    print(result.files["report"])
    print("---- summary ----")
    print(result.files["summary"].read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

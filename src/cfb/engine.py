"""Local engine: the PC pulls the data from CFBD, builds the features, runs the week, writes the
report and backs everything up to R2, with no cloud session involved.

    python -m cfb.engine status                      # data, last run, schedule (JSON)
    python -m cfb.engine bootstrap [--source auto|r2|cfbd] [--force]
    python -m cfb.engine refresh [--season N]        # pull the season from CFBD (--force, dated copies)
    python -m cfb.engine build [--season N]          # core tables, unit ratings, the season's features
    python -m cfb.engine week [--season N --week N]  # predict and write the report (cfb.run_week)
    python -m cfb.engine all                         # the weekly update: bootstrap if needed, refresh,
                                                     # build, week, back up (one run_week pass)
    python -m cfb.engine light                       # daily: games, lines, polls, weather; core; week
    python -m cfb.engine check [--reference PATH]    # rerun the cloud's report week on the same data and
                                                     # compare every prediction
    python -m cfb.engine backup                      # publish the data dir to R2 (slim and full channels)
    python -m cfb.engine schedule show|install|remove   # Windows Task Scheduler (cfb.schedule)

Packaged app: `"CFB Predictor.exe" engine <command>` (console) or `"CFB Engine.exe" engine
<command>` (no window; what Task Scheduler runs).

Folders (all per edition, cfb.edition; main shown):
  data dir        %LOCALAPPDATA%\\cfb\\data        cfb.settings.data_dir()
  dated copies    %LOCALAPPDATA%\\cfb\\asof        cfb.settings.asof_archive()
  engine folder   %LOCALAPPDATA%\\cfb             $CFB_ENGINE_DIR: logs\\, engine-status.json (last run),
                                                 runs.jsonl (every run), engine.lock, reference\\,
                                                 engine.json (preferences: {"backup": true})

Bootstrap (first run): the R2 full channel (cfb.warehouse_sync, raw/ included) is downloaded and
becomes the data dir; its newest report is kept as the reference for `check`. Without R2 keys,
`--source cfbd` pulls everything from CFBD instead (about 2,100 calls, about an hour, plus about 40
minutes of feature building).

Exit codes: 0 done (or nothing to do), 1 failed, 2 done with a warning (failed requests, backup
failed), 3 another engine run is in progress.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import edition as _edition
from . import settings

LIGHT_DATASETS = ("games", "lines", "rankings", "weather", "media")
KEEP_LOGS = 60
HEARTBEAT_S = 20.0
STALE_S = 180.0
CHECK_TOL = {"p": 1.5e-4, "pred_margin": 0.015, "pred_total": 0.015}  # bundle rounding: 4 and 2 decimals
STEP_RE = re.compile(r"\[run_week [\d:]+\] step (\d)/8 (\w+)")

EXIT_OK, EXIT_FAILED, EXIT_WARNING, EXIT_BUSY = 0, 1, 2, 3


class EngineError(RuntimeError):
    pass


class EngineBusy(EngineError):
    pass


# ---- folders and preferences ------------------------------------------------------------------

def engine_dir() -> Path:
    env = os.environ.get("CFB_ENGINE_DIR")
    return Path(env) if env else settings.user_data_root()


def data_dir() -> Path:
    return settings.data_dir()


def archive_dir() -> Path:
    return settings.asof_archive() or engine_dir() / "asof"


def logs_dir() -> Path:
    return engine_dir() / "logs"


def status_path() -> Path:
    return engine_dir() / "engine-status.json"


def history_path() -> Path:
    return engine_dir() / "runs.jsonl"


def reference_dir() -> Path:
    return engine_dir() / "reference"


def prefs_path() -> Path:
    return engine_dir() / "engine.json"


DEFAULT_PREFS = {"backup": True}


def load_prefs() -> dict:
    try:
        doc = json.loads(prefs_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = {}
    return {**DEFAULT_PREFS, **(doc if isinstance(doc, dict) else {})}


def save_prefs(**changes) -> dict:
    prefs = {**load_prefs(), **changes}
    prefs_path().parent.mkdir(parents=True, exist_ok=True)
    _write_json(prefs_path(), prefs)
    return prefs


def _write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(t: datetime | None) -> str | None:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


# ---- run status ---------------------------------------------------------------------------------

def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":  # os.kill(pid, 0) would terminate the process on Windows
        import ctypes

        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        try:
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        finally:
            k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_status() -> dict | None:
    """The last run's status. A "running" status whose process is gone, or whose heartbeat stopped,
    is reported as "interrupted"."""
    try:
        doc = json.loads(status_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if doc.get("state") == "running":
        beat = doc.get("heartbeat_utc") or doc.get("started_utc")
        try:
            age = (_now() - datetime.strptime(beat, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            age = STALE_S + 1
        if not pid_alive(doc.get("pid")) or age > STALE_S:
            doc = {**doc, "state": "interrupted",
                   "error": "the run stopped before it finished (PC shut down, app closed or the process was killed)"}
    return doc


def history(limit: int = 20) -> list[dict]:
    try:
        lines = history_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
        if len(out) >= limit:
            break
    return out


class Run:
    """One engine run: its status file, heartbeat and step list."""

    def __init__(self, command: str, trigger: str, log_path: Path | None):
        self.mu = threading.Lock()
        self.doc = {
            "command": command, "trigger": trigger, "state": "running", "edition": _edition.current().name,
            "pid": os.getpid(), "started_utc": _iso(_now()), "heartbeat_utc": _iso(_now()), "finished_utc": None,
            "log": str(log_path) if log_path else None, "data_dir": str(data_dir()), "current": None,
            "steps": [], "notes": [], "warnings": [], "error": None,
        }
        self._stop = threading.Event()
        self._beat = None

    def save(self) -> None:
        with self.mu:
            doc = json.loads(json.dumps(self.doc, default=str))
        _write_json(status_path(), doc)

    def set(self, **kw) -> None:
        with self.mu:
            self.doc.update(kw)
        self.save()

    def note(self, text: str) -> None:
        with self.mu:
            self.doc["notes"].append(text)

    def warn(self, text: str) -> None:
        with self.mu:
            self.doc["warnings"].append(text)
        print(f"[engine] warning: {text}", flush=True)

    def current(self, name: str | None) -> None:
        with self.mu:
            self.doc["current"] = name
        self.save()

    @contextmanager
    def step(self, name: str):
        rec = {"name": name, "state": "running", "secs": None, "note": None}
        with self.mu:
            self.doc["steps"].append(rec)
            self.doc["current"] = name
        self.save()
        log(f"step {name}")
        t0 = time.monotonic()
        try:
            yield rec
        except BaseException:
            rec["state"] = "failed"
            raise
        else:
            if rec["state"] == "running":
                rec["state"] = "done"
        finally:
            rec["secs"] = round(time.monotonic() - t0, 1)
            self.save()

    def start_heartbeat(self) -> None:
        def beat():
            while not self._stop.wait(HEARTBEAT_S):
                with self.mu:
                    self.doc["heartbeat_utc"] = _iso(_now())
                try:
                    self.save()
                except OSError:
                    pass
        self._beat = threading.Thread(target=beat, name="cfb-engine-heartbeat", daemon=True)
        self._beat.start()

    def finish(self, state: str, error: str | None = None) -> dict:
        self._stop.set()
        with self.mu:
            self.doc.update(state=state, error=error, finished_utc=_iso(_now()), current=None)
            self.doc["heartbeat_utc"] = self.doc["finished_utc"]
        self.save()
        with self.mu:
            line = json.dumps({k: v for k, v in self.doc.items() if k != "current"}, default=str)
        history_path().parent.mkdir(parents=True, exist_ok=True)
        with open(history_path(), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return self.doc


# ---- logging --------------------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[engine {datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


class _Tee(io.TextIOBase):
    """Writes to the console (when there is one) and the log file; calls `on_line` per line."""

    def __init__(self, console, fh, on_line: Callable[[str], None] | None = None):
        self.console, self.fh, self.on_line, self._buf = console, fh, on_line, ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        for target in (self.console, self.fh):
            if target is None:
                continue
            try:
                target.write(s)
                target.flush()
            except Exception:  # a console that cannot show a character must not stop the run
                pass
        if self.on_line is not None:
            self._buf += s
            *lines, self._buf = self._buf.split("\n")
            for line in lines:
                try:
                    self.on_line(line)
                except Exception:
                    pass
        return len(s)

    def flush(self) -> None:
        for target in (self.console, self.fh):
            try:
                if target is not None:
                    target.flush()
            except Exception:
                pass


def _prune_logs(keep: int = KEEP_LOGS) -> None:
    logs = sorted(logs_dir().glob("engine-*.log"), key=lambda p: p.name)
    for p in logs[:-keep]:
        try:
            p.unlink()
        except OSError:
            pass


@contextmanager
def _logging(command: str, on_line: Callable[[str], None] | None = None):
    logs_dir().mkdir(parents=True, exist_ok=True)
    path = logs_dir() / f"engine-{_now():%Y%m%dT%H%M%SZ}-{command}.log"
    fh = open(path, "a", encoding="utf-8")
    out, err = sys.stdout, sys.stderr
    sys.stdout = _Tee(out, fh, on_line)
    sys.stderr = _Tee(err, fh)
    try:
        yield path
    finally:
        sys.stdout, sys.stderr = out, err
        fh.close()
        _prune_logs()


@contextmanager
def _locked():
    from .warehouse_sync import SyncBusy, _Lock

    try:
        with _Lock(engine_dir() / "engine.lock",
                   busy="another engine run is in progress (the scheduled task or the app); wait for it to finish"):
            yield
    except SyncBusy as exc:
        raise EngineBusy(str(exc)) from exc


# ---- data state ---------------------------------------------------------------------------------

def is_bootstrapped(d: Path | None = None) -> bool:
    d = Path(d or data_dir())
    raw, tables = d / "raw", d / "tables"
    return raw.is_dir() and tables.is_dir() and any(raw.iterdir()) and any(tables.iterdir())


def bootstrap_info(d: Path | None = None) -> dict | None:
    p = Path(d or data_dir()) / "bootstrap.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def live_season(now: datetime | None = None) -> int:
    from .ingest import current_season

    return current_season(now or _now())


def save_reference(d: Path, source: str) -> Path | None:
    """Copy the data dir's newest report bundle to reference/<key>/ for `check`."""
    from .ui.data import list_reports

    reports = list_reports(d)
    if not reports:
        return None
    key, folder = reports[0]
    dst = reference_dir() / key
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(folder / "bundle.json", dst / "bundle.json")
    _write_json(dst / "reference.json", {"key": key, "source": source, "saved_utc": _iso(_now())})
    return dst / "bundle.json"


# ---- steps --------------------------------------------------------------------------------------

def bootstrap(run: Run | None = None, *, source: str = "auto", force: bool = False) -> str:
    """Fill an empty data dir. Returns what was done: "present", "r2" or "cfbd"."""
    d = data_dir()
    if is_bootstrapped(d) and not force:
        log(f"data present in {d}; bootstrap skipped")
        return "present"
    from . import warehouse_sync as ws

    if source in ("auto", "r2"):
        r2 = ws.r2_settings()
        if r2.configured:
            return _bootstrap_r2(d, ws, r2, run)
        if source == "r2":
            raise EngineError("no R2 settings: save r2_account_id, r2_access_key_id and r2_secret_access_key "
                              "on the Settings page")
        raise EngineError(f"no data in {d} yet. Save the R2 keys on the Settings page and run Bootstrap, "
                          "or run a full pull from CFBD (Bootstrap with source cfbd, about two hours)")
    if source == "cfbd":
        return _bootstrap_cfbd(d, run)
    raise EngineError(f"unknown bootstrap source {source!r}")


def _move_aside(d: Path) -> None:
    if not d.exists():
        return
    if not any(d.iterdir()):
        d.rmdir()
        return
    old = d.with_name(f"{d.name}.old-{_now():%Y%m%dT%H%M%SZ}")
    os.replace(d, old)
    log(f"kept the previous data dir as {old}")


def _bootstrap_r2(d: Path, ws, r2, run: Run | None) -> str:
    from .ingest import sync_asof

    base = d.parent / f".bootstrap-{d.name}"
    log(f"bootstrap: downloading the full channel of bucket {r2.bucket} ({r2.backend})")
    res = ws.sync(base, ws.open_store(r2), channel="full", keep_versions=1, log=lambda m: log("  " + m))
    _move_aside(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    os.replace(res.data_dir, d)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    (d / "manifest.json").unlink()
    _write_json(d / "bootstrap.json", {"source": "r2", "bucket": r2.bucket, "version": manifest.get("version"),
                                       "key": manifest.get("key"), "published_utc": manifest.get("created_utc"),
                                       "bootstrapped_utc": _iso(_now()), "files": len(manifest.get("files", {})),
                                       "bytes": manifest.get("total_bytes")})
    shutil.rmtree(base, ignore_errors=True)
    n = sync_asof(d, archive_dir())
    log(f"bootstrap: {len(manifest.get('files', {}))} files ({(manifest.get('total_bytes') or 0) / 1e6:.0f} MB) "
        f"from {manifest.get('version')}; {n} dated copies saved to {archive_dir()}")
    ref = save_reference(d, f"r2 {manifest.get('version')}")
    if ref is not None:
        log(f"bootstrap: reference report for `check`: {ref.parent.name}")
    if run is not None:
        run.set(bootstrap=bootstrap_info(d))
    return "r2"


def _bootstrap_cfbd(d: Path, run: Run | None) -> str:
    from .client import CFBDClient
    from .core.asof import Warehouse
    from .core.build import build_core
    from .features.base import discover_groups
    from .features.build import build_derived, build_features
    from .ingest import BY_NAME, run as ingest_run
    from .store import Store

    now = _now()
    end = live_season(now)
    client = CFBDClient()
    log(f"bootstrap: full pull 2014-{end} from CFBD into {d} (about 2,100 calls; the player success "
        "endpoint takes about 12 s per call)")
    failures = ingest_run(client, Store(d), list(range(2014, end + 1)), list(BY_NAME), force=False, now=now,
                          asof_archive=archive_dir())
    log(f"bootstrap: {client.calls} API calls, {len(failures)} failures")
    if failures:
        for f in failures[:20]:
            log(f"  failed: {f}")
        if run is not None:
            run.warn(f"full pull had {len(failures)} failed requests; run bootstrap --source cfbd again to retry them")
    build_core(d)
    wh = Warehouse(d)
    build_derived(wh)
    build_features(wh, discover_groups())
    _write_json(d / "bootstrap.json", {"source": "cfbd", "api_calls": client.calls, "failures": len(failures),
                                       "bootstrapped_utc": _iso(_now())})
    if run is not None:
        run.set(bootstrap=bootstrap_info(d), api_calls=client.calls)
    return "cfbd"


def refresh(run: Run, season: int | None = None, names: tuple[str, ...] | None = None) -> list[str]:
    from . import run_week as RW

    season = season or live_season()
    stats: dict = {}
    failures = RW.refresh(data_dir(), season, archive_dir(), _now(), names=list(names) if names else None,
                          stats=stats)
    run.set(api_calls=run.doc.get("api_calls", 0) + stats.get("api_calls", 0), season=season)
    if failures:
        run.warn(f"{len(failures)} failed requests in the refresh; the run used what was pulled")
        run.set(failures=failures[:50])
    return failures


def build(run: Run, season: int | None = None) -> None:
    from . import run_week as RW
    from .core.asof import Warehouse
    from .core.build import build_core

    season = season or live_season()
    build_core(data_dir())
    RW.build_features_for(Warehouse(data_dir()), season)
    run.set(season=season)


def week(run: Run, *, season=None, season_type=None, week_no=None, do_refresh=None, build_features=None,
         config_path=None, out_dir=None, write_predictions=True):
    from . import run_week as RW
    from .report.config import load_config

    config = load_config(config_path) if config_path else load_config()
    res = RW.run(data_dir=data_dir(), config=config, season=season, season_type=season_type, week=week_no,
                 do_refresh=do_refresh, archive=archive_dir(), build_features=build_features, out_dir=out_dir,
                 write_predictions=write_predictions, publish_warehouse_to_r2=False)
    run.set(report_key=res.slate.key, report_dir=str(res.out_dir), report_label=res.slate.label,
            report_games=len(res.slate.games), api_calls=run.doc.get("api_calls", 0) + res.api_calls)
    if res.refresh_failures:
        run.warn(f"{len(res.refresh_failures)} failed requests in the refresh; the report used what was pulled")
        run.set(failures=res.refresh_failures[:50])
    for n in res.notes:
        run.note(n)
    return res


OFF_SEASON_MONTHS = (2, 3, 4, 5, 6, 7)  # no FBS games from February to July


def off_season(now: datetime | None = None) -> bool:
    """February to July: the scheduled runs have nothing to report and skip. Decided by the date,
    not the data, so the first run in August still pulls the new season's calendar."""
    return (now or _now()).month in OFF_SEASON_MONTHS


def _commit() -> str | None:
    try:
        from . import _build
        return getattr(_build, "COMMIT", None)
    except ImportError:
        from .warehouse_sync import git_commit
        return git_commit()


def backup(run: Run, key: str | None = None) -> dict:
    """Publish the data dir to R2: the slim channel (what apps and cloud threads sync) and the full
    channel (raw/ included, a complete copy to restore from)."""
    from . import warehouse_sync as ws

    r2 = ws.r2_settings()
    if not r2.configured:
        out = {"state": "skipped", "note": "no R2 settings"}
        run.set(backup=out)
        return out
    out: dict = {"bucket": r2.bucket}
    try:
        store, quota, commit = ws.open_store(r2), ws.cfbd_quota(), _commit()
        for channel in ("slim", "full"):
            res = ws.publish(data_dir(), store, key, quota=quota, commit=commit, channel=channel,
                             log=lambda m: log("  " + m))
            out[channel] = {"version": res.version, "changed": res.changed, "uploaded_files": res.uploaded_files,
                            "uploaded_mb": round(res.uploaded_bytes / 1e6, 1)}
        out["state"] = "done"
    except Exception as exc:  # the run's own results are on disk; the backup can be retried
        msg = f"{exc.__class__.__name__}: {exc}"
        if "AccessDenied" in msg or "403" in msg:
            msg += " (the R2 key may be read-only: save a key pair with Object Read & Write to back up)"
        out.update(state="failed", error=msg)
        run.warn(f"backup to R2 failed: {msg}")
    run.set(backup=out)
    return out


def quota(run: Run) -> None:
    from .warehouse_sync import cfbd_quota

    q = cfbd_quota()
    if q:
        run.set(cfbd_quota=q)


def compare_bundles(ref: dict, new: dict) -> dict:
    """Every model's p_home_win, and the primary's margin and total, game by game."""
    new_games = {g["game_id"]: g for g in new.get("games", [])}
    worst = {"p": 0.0, "pred_margin": 0.0, "pred_total": 0.0}
    n_values, missing, over = 0, [], []
    for g in ref.get("games", []):
        h = new_games.get(g["game_id"])
        if h is None:
            missing.append(g["game_id"])
            continue
        pairs = [("p", m, v, (h.get("p") or {}).get(m)) for m, v in (g.get("p") or {}).items()]
        pairs += [(k, k, g.get(k), h.get(k)) for k in ("pred_margin", "pred_total")]
        for kind, name, a, b in pairs:
            if a is None and b is None:
                continue
            n_values += 1
            if a is None or b is None:
                over.append({"game_id": g["game_id"], "value": name, "reference": a, "local": b})
                continue
            diff = abs(float(a) - float(b))
            worst[kind] = max(worst[kind], diff)
            if diff > CHECK_TOL[kind]:
                over.append({"game_id": g["game_id"], "value": name, "reference": a, "local": b})
    return {"games": len(ref.get("games", [])), "values": n_values, "max_diff": worst, "missing_games": missing,
            "mismatches": over[:50], "n_mismatches": len(over), "ok": not missing and not over}


def latest_reference() -> Path | None:
    root = reference_dir()
    if not root.is_dir():
        return None
    from .ui.data import report_sort_key

    keys = sorted((p for p in root.iterdir() if (p / "bundle.json").exists()),
                  key=lambda p: report_sort_key(p.name), reverse=True)
    return keys[0] / "bundle.json" if keys else None


def check(run: Run, reference: Path | None = None, config_path=None) -> dict:
    """Rerun the reference report's week on the local data (no refresh, stored features) and compare."""
    ref_path = Path(reference) if reference else latest_reference()
    if ref_path is None or not ref_path.exists():
        raise EngineError("no reference report: bootstrap from R2 saves one (or pass --reference bundle.json)")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    meta = ref["meta"]
    m = re.match(r"^(\d{4})_([a-z]+)_(\d+)$", ref_path.parent.name)
    season = int(meta.get("season") or m.group(1))
    season_type = meta.get("season_type") or m.group(2)
    week_no = int(meta.get("week") or m.group(3))
    boot = bootstrap_info()
    since = [h for h in history(200) if boot and (h.get("api_calls") or 0) > 0
             and (h.get("finished_utc") or "") > (boot.get("bootstrapped_utc") or "")]
    if since:
        run.warn("the data has been refreshed since the bootstrap, so differences from the reference are expected; "
                 "bootstrap again (--force) for an exact check")
    out_dir = engine_dir() / "check" / ref_path.parent.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    res = week(run, season=season, season_type=season_type, week_no=week_no, do_refresh=False, build_features=False,
               config_path=config_path, out_dir=out_dir, write_predictions=False)
    new = json.loads(Path(res.files["bundle"]).read_text(encoding="utf-8"))
    result = compare_bundles(ref, new)
    result["reference"] = str(ref_path)
    run.set(check=result)
    md = result["max_diff"]
    log(f"check {ref_path.parent.name}: {result['games']} games, {result['values']} values; max difference "
        f"p {md['p']:.6f}, margin {md['pred_margin']:.3f}, total {md['pred_total']:.3f}; "
        f"{'MATCH' if result['ok'] else str(result['n_mismatches']) + ' mismatches'}")
    if not result["ok"]:
        raise EngineError(f"the local predictions differ from the reference ({result['n_mismatches']} values, "
                          f"{len(result['missing_games'])} games missing)")
    return result


# ---- commands -------------------------------------------------------------------------------------

def _execute(command: str, trigger: str, body: Callable[[Run], None]) -> int:
    """Run `body` under the engine lock with logging and a status file. Returns the exit code."""
    try:
        with _locked():
            run_holder: dict = {}

            def on_line(line: str) -> None:
                m = STEP_RE.search(line)
                if m and "run" in run_holder:
                    run_holder["run"].current(m.group(2))

            with _logging(command, on_line) as log_path:
                run = Run(command, trigger, log_path)
                run_holder["run"] = run
                run.save()
                run.start_heartbeat()
                ed = _edition.current()
                log(f"{ed.app_name} engine: {command} (trigger {trigger}), data {data_dir()}, "
                    f"dated copies {archive_dir()}")
                try:
                    body(run)
                except Exception as exc:
                    traceback.print_exc()
                    run.finish("failed", f"{exc.__class__.__name__}: {exc}")
                    log(f"{command} FAILED: {exc}")
                    return EXIT_FAILED
                except SystemExit as exc:  # run_week raises SystemExit with a message for user errors
                    msg = str(exc.code) if exc.code not in (None, 0) else "stopped"
                    run.finish("failed", msg)
                    log(f"{command} FAILED: {msg}")
                    return EXIT_FAILED
                state = run.doc.get("state")
                if state == "running":
                    state = "warning" if run.doc["warnings"] else "ok"
                run.finish(state)
                log(f"{command} finished: {state}")
                return EXIT_WARNING if state == "warning" else EXIT_OK
    except EngineBusy as exc:
        print(f"[engine] {exc}", file=sys.stderr, flush=True)
        return EXIT_BUSY


def _maybe_backup(run: Run, key: str | None, enabled: bool | None) -> None:
    enabled = load_prefs().get("backup", True) if enabled is None else enabled
    if not enabled:
        run.set(backup={"state": "off"})
        return
    with run.step("backup") as rec:
        out = backup(run, key)
        rec["state"] = {"done": "done", "skipped": "skipped"}.get(out["state"], "failed")
        rec["note"] = out.get("error") or out.get("note")


def cmd_all(run: Run, args) -> None:
    with run.step("bootstrap") as rec:
        did = bootstrap(run, source=args.source)
        rec["state"] = "skipped" if did == "present" else "done"
    if args.season is None and args.week is None and off_season():
        run.set(state="skipped")
        run.note("nothing to do in the off-season (February to July); pass --season and --week to run a week")
        log("off-season: nothing to do")
        return
    with run.step("weekly run") as rec:
        res = week(run, season=args.season, season_type=args.season_type, week_no=args.week,
                   do_refresh=False if args.no_refresh else None, config_path=args.config)
        rec["note"] = res.slate.label
    _maybe_backup(run, res.slate.key, args.backup)
    quota(run)


def cmd_light(run: Run, args) -> None:
    if not is_bootstrapped():
        raise EngineError("no data yet: run the weekly update (or Bootstrap) first")
    if off_season():
        run.set(state="skipped")
        run.note("nothing to do in the off-season (February to July)")
        return
    from .core.build import build_core

    with run.step("refresh (light)"):
        refresh(run, names=LIGHT_DATASETS)
    with run.step("core"):
        build_core(data_dir())
    with run.step("weekly run") as rec:
        res = week(run, do_refresh=False, config_path=args.config)
        rec["note"] = res.slate.label
    _maybe_backup(run, res.slate.key, args.backup)
    quota(run)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cfb.engine", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--trigger", default=None, help="who started the run: manual, app, scheduled")
        return p

    add("status", "data, last run and schedule as JSON")
    b = add("bootstrap", "fill an empty data dir from R2 (or a full CFBD pull)")
    b.add_argument("--source", choices=("auto", "r2", "cfbd"), default="auto")
    b.add_argument("--force", action="store_true", help="bootstrap again; the current data dir is kept as data.old-*")
    r = add("refresh", "pull the season from CFBD (--force, dated copies)")
    r.add_argument("--season", type=int)
    bu = add("build", "core tables, unit ratings and the season's features")
    bu.add_argument("--season", type=int)
    for name, text in (("week", "predict the week and write the report"),
                       ("all", "weekly update: bootstrap if needed, refresh, build, week, back up"),
                       ("light", "daily: games, lines, polls, weather; core; the week's report")):
        p = add(name, text)
        p.add_argument("--config", default=None, help=f"run configuration (default {settings.config_path()})")
        p.add_argument("--season", type=int)
        p.add_argument("--week", type=int)
        p.add_argument("--season-type", choices=("regular", "postseason"))
        g = p.add_mutually_exclusive_group()
        g.add_argument("--backup", dest="backup", action="store_true", default=None, help="back up to R2 (default: preference)")
        g.add_argument("--no-backup", dest="backup", action="store_false")
        if name == "all":
            p.add_argument("--no-refresh", action="store_true", help="do not pull from CFBD")
            p.add_argument("--source", choices=("auto", "r2", "cfbd"), default="auto", help="bootstrap source")
        if name == "week":
            p.add_argument("--refresh", action="store_true", help="pull the season first (like all)")
            f = p.add_mutually_exclusive_group()
            f.add_argument("--features", dest="features", action="store_true", default=None)
            f.add_argument("--no-features", dest="features", action="store_false")
    c = add("check", "rerun the reference week on the local data and compare every prediction")
    c.add_argument("--reference", default=None, help="bundle.json (default: the one saved by the bootstrap)")
    c.add_argument("--config", default=None)
    add("backup", "publish the data dir to R2 (slim and full channels)")
    sch = sub.add_parser("schedule", help="Windows Task Scheduler tasks (show, install, remove, xml)")
    sch.add_argument("rest", nargs=argparse.REMAINDER)
    pr = sub.add_parser("prefs", help="show or change preferences")
    pr.add_argument("--backup", choices=("on", "off"))
    args = ap.parse_args(argv)
    trigger = getattr(args, "trigger", None) or "manual"

    if args.cmd == "schedule":
        from .schedule import main as schedule_main
        return schedule_main(args.rest)
    if args.cmd == "prefs":
        prefs = save_prefs(backup=args.backup == "on") if args.backup else load_prefs()
        print(json.dumps(prefs))
        return 0
    if args.cmd == "status":
        print(json.dumps(status(), indent=1, default=str))
        return 0

    bodies = {
        "bootstrap": lambda run: bootstrap(run, source=args.source, force=args.force),
        "refresh": lambda run: refresh(run, args.season),
        "build": lambda run: build(run, args.season),
        "week": lambda run: (week(run, season=args.season, season_type=args.season_type, week_no=args.week,
                                  do_refresh=bool(args.refresh), build_features=args.features,
                                  config_path=args.config),
                             _maybe_backup(run, None, args.backup) if args.backup else None),
        "all": lambda run: cmd_all(run, args),
        "light": lambda run: cmd_light(run, args),
        "check": lambda run: check(run, args.reference, args.config),
        "backup": lambda run: backup(run),
    }
    return _execute(args.cmd, trigger, bodies[args.cmd])


def status() -> dict:
    from . import warehouse_sync as ws

    ed = _edition.current()
    try:
        r2 = ws.r2_settings()
        r2_doc = {"configured": r2.configured, "bucket": r2.bucket, "backend": r2.backend}
    except Exception as exc:
        r2_doc = {"configured": False, "error": str(exc)}
    out = {"edition": ed.name, "app": ed.app_name, "data_dir": str(data_dir()), "bootstrapped": is_bootstrapped(),
           "bootstrap": bootstrap_info(), "asof_archive": str(archive_dir()), "engine_dir": str(engine_dir()),
           "cfbd_key": settings.secret_source("cfbd_api_key"), "r2": r2_doc, "prefs": load_prefs(),
           "last_run": read_status()}
    try:
        from . import schedule as S
        if S.available():
            out["schedule"] = [S.show(k, info=False) for k in S.KINDS]
    except Exception as exc:
        out["schedule_error"] = str(exc)
    return out


if __name__ == "__main__":
    sys.exit(main())

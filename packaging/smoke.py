"""Smoke test of the packaged app (CI): selftest (with the key store and Task Scheduler checks),
the local engine end to end on the synthetic store (bootstrap from a folder bucket, the weekly
run, backup, check) with the windowless "CFB Engine.exe", then start the server on a synthetic
report and check it answers.
Usage: python packaging/smoke.py "dist/CFB Predictor/CFB Predictor.exe" [--no-system]"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_ui import make_bundle  # noqa: E402  (synthetic bundle, no real data)

PORT = 8765


def get(url: str) -> int:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status


ENGINE_CONFIG = ('[models]\nprimary = "baseline_elo"\nbaselines = ["baseline_close", "baseline_home"]\n'
                 '[run]\nbuild_features = false\nmarket_models = ["baseline_close"]\n')


def engine_smoke(exe: str, tmp: Path) -> int:
    """The cloud side builds a synthetic data dir with a report and publishes the full channel to a
    folder bucket; the packaged engine bootstraps from it, runs the week, backs up and checks."""
    from cfb import warehouse_sync as ws
    from cfb.core.synthetic import SEASONS, synthetic_warehouse
    from cfb.report.config import load_config
    from cfb.run_week import run

    root = tmp / "engine"
    cloud = synthetic_warehouse(root / "cloud" / "data").root
    conf = root / "config.toml"
    conf.write_text(ENGINE_CONFIG, encoding="utf-8")
    season = max(SEASONS)
    run(cloud, load_config(conf), season=season, week=4, do_refresh=False)
    ws.publish(cloud, ws.LocalStore(root / "bucket"), channel="full", log=lambda m: None)
    env = {**os.environ, "CFB_DATA_DIR": str(root / "pc" / "data"), "CFB_ENGINE_DIR": str(root / "pc"),
           "CFB_ASOF_ARCHIVE": str(root / "pc" / "asof"), "CFB_WAREHOUSE_LOCAL": str(root / "bucket"),
           "CFB_WAREHOUSE_DIR": str(root / "pc" / "warehouse"), "CFB_PROJECT_FILES": str(root / "no-mount")}
    quiet = str(Path(exe).with_name("CFB Engine.exe"))
    steps = [
        (quiet, ["engine", "all", "--season", str(season), "--week", "4", "--no-refresh", "--config", str(conf),
                 "--trigger", "ci"]),
        (exe, ["engine", "check", "--config", str(conf)]),
        (exe, ["engine", "status"]),
    ]
    for prog, args in steps:
        t0 = time.time()
        r = subprocess.run([prog, *args], env=env, capture_output=True, text=True, timeout=900)
        status = json.loads((root / "pc" / "engine-status.json").read_text(encoding="utf-8"))
        print(f"{Path(prog).name} {' '.join(args[:2])}: exit {r.returncode} in {time.time() - t0:.0f} s, "
              f"state {status.get('state')}")
        if r.stdout.strip():
            print(r.stdout[-3000:])
        if r.returncode != 0:
            print(r.stderr[-3000:])
            log = status.get("log")
            if log and Path(log).exists():
                print(Path(log).read_text(encoding="utf-8", errors="replace")[-6000:])
            print("engine smoke test failed", file=sys.stderr)
            return 1
    status = json.loads((root / "pc" / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ok = (status["state"] == "ok" and status["report_key"] == f"{season}_regular_04"
          and status["backup"]["state"] == "done" and (root / "bucket" / "full" / "manifest.json").exists()
          and (root / "pc" / "data" / "reports" / f"{season}_regular_04" / "bundle.json").exists())
    print(f"engine smoke test {'ok' if ok else 'failed'}: {status['steps']}")
    return 0 if ok else 1


def main() -> int:
    exe = sys.argv[1]
    tmp = Path(tempfile.mkdtemp())
    rep = tmp / "data" / "reports" / "2026_regular_12"
    rep.mkdir(parents=True)
    (rep / "bundle.json").write_text(json.dumps(make_bundle()), encoding="utf-8")
    env = {**os.environ, "CFB_DATA_DIR": str(tmp / "data"), "CFB_WAREHOUSE_DIR": str(tmp / "warehouse"),
           "CFB_PROJECT_FILES": str(tmp / "no-mount"), "NO_PROXY": "127.0.0.1,localhost"}

    system = [] if "--no-system" in sys.argv else ["--system"]
    st = subprocess.run([exe, "selftest", *system], env=env, capture_output=True, text=True, timeout=300)
    print(st.stdout, st.stderr)
    if st.returncode != 0 or "selftest ok" not in st.stdout:
        return 1
    if engine_smoke(exe, tmp) != 0:
        return 1

    log = tmp / "server.log"
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen([exe, "--no-browser", "--port", str(PORT)], env=env, stdout=fh, stderr=subprocess.STDOUT)
    ok = False
    try:
        for _ in range(120):
            try:
                ok = get(f"http://127.0.0.1:{PORT}/_stcore/health") == 200
            except OSError:
                ok = False
            if ok:
                break
            time.sleep(1)
        ok = ok and get(f"http://127.0.0.1:{PORT}/") == 200
        time.sleep(3)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    text = log.read_text(encoding="utf-8", errors="replace")
    print(text)
    if not ok or "Traceback" in text:
        print("smoke test failed", file=sys.stderr)
        return 1
    print("smoke test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

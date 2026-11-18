"""Smoke test of the packaged app (CI): selftest, then start the server on a synthetic report
and check it answers. Usage: python packaging/smoke.py "dist/CFB Predictor/CFB Predictor.exe" """
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


def main() -> int:
    exe = sys.argv[1]
    tmp = Path(tempfile.mkdtemp())
    rep = tmp / "data" / "reports" / "2026_regular_12"
    rep.mkdir(parents=True)
    (rep / "bundle.json").write_text(json.dumps(make_bundle()), encoding="utf-8")
    env = {**os.environ, "CFB_DATA_DIR": str(tmp / "data"), "CFB_WAREHOUSE_DIR": str(tmp / "warehouse"),
           "CFB_PROJECT_FILES": str(tmp / "no-mount"), "NO_PROXY": "127.0.0.1,localhost"}

    st = subprocess.run([exe, "selftest"], env=env, capture_output=True, text=True, timeout=300)
    print(st.stdout, st.stderr)
    if st.returncode != 0 or "selftest ok" not in st.stdout:
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

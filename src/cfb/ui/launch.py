"""Start the local app: a Streamlit server on 127.0.0.1 that opens in the default browser.

    cfb-ui [--port 8501] [--no-browser]

The packaged .exe calls main() too. The server listens on localhost only.
"""
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path


def free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return s.getsockname()[1]
    raise RuntimeError("no free port")


def app_path() -> Path:
    """app.py as a real file: Streamlit runs a script path. In the packaged app the script is
    bundled as data under the app folder (packaging/cfb_app.spec)."""
    from .. import settings
    root = settings.bundle_root()
    return root / "cfb" / "ui" / "app.py" if root is not None else Path(__file__).with_name("app.py")


def streamlit_argv(port: int, browser: bool) -> list[str]:
    return [
        "streamlit", "run", str(app_path()),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        f"--server.headless={'false' if browser else 'true'}",
        "--browser.gatherUsageStats=false",
        # The packaged app has no source tree to watch.
        "--server.fileWatcherType=none",
        "--global.developmentMode=false",
        # No "Deploy" button or developer menu: this is an app, not a Streamlit dev session.
        "--client.toolbarMode=viewer",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cfb-ui", description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("Streamlit is not installed: pip install -e '.[ui]'", file=sys.stderr)
        return 1
    port = free_port(args.port)
    sys.argv = streamlit_argv(port, not args.no_browser)
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())

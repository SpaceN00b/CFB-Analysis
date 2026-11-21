"""Start local engine runs from the app and follow them (cfb.engine).

A run is its own process, the same command Task Scheduler starts, so it keeps going when the
user changes page or closes the app, and a crash cannot take the app down. Progress comes
from the engine's status file and log, so a scheduled run shows up the same way as one
started here. No Streamlit here, so it is testable without the UI.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from .. import engine as E
from .. import settings

NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW
NEW_GROUP = 0x00000200  # CREATE_NEW_PROCESS_GROUP: Ctrl+C in the app's console does not stop the run
COMMANDS = ("all", "light", "bootstrap", "check", "backup")


def command_line(command: str, *extra: str) -> list[str]:
    """The engine command line for this install: "CFB Engine.exe" (no window) next to the
    packaged app, else the app exe, else `python -m cfb.engine` from a source checkout."""
    if settings.frozen():
        exe = Path(sys.executable)
        quiet = exe.with_name("CFB Engine.exe")
        return [str(quiet if quiet.exists() else exe), "engine", command, *extra]
    return [sys.executable, "-m", "cfb.engine", command, *extra]


class EngineJob:
    def __init__(self):
        self._mu = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._seen_finish: str | None = None
        st = E.read_status()
        self._seen_finish = st.get("finished_utc") if st else None

    def start(self, command: str, *extra: str) -> bool:
        """Start a run. False (nothing started) when one is already running."""
        if command not in COMMANDS:
            raise ValueError(f"unknown engine command {command!r}")
        with self._mu:
            if self.running():
                return False
            E.logs_dir().mkdir(parents=True, exist_ok=True)
            out = open(E.logs_dir() / "app-launch.log", "a", encoding="utf-8")
            kw: dict = {"stdout": out, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
            if sys.platform == "win32":
                kw["creationflags"] = NO_WINDOW | NEW_GROUP
            else:
                kw["start_new_session"] = True
            try:
                self._proc = subprocess.Popen(command_line(command, *extra, "--trigger", "app"), **kw)
            finally:
                out.close()
            return True

    def running(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        st = E.read_status()
        return bool(st) and st.get("state") == "running"

    def status(self) -> dict | None:
        return E.read_status()

    def exit_code(self) -> int | None:
        return self._proc.poll() if self._proc is not None else None

    def take_fresh(self) -> bool:
        """True once after a run (started here or by Task Scheduler) finished, so the app can
        clear its caches and show the new report."""
        st = E.read_status()
        fin = st.get("finished_utc") if st else None
        with self._mu:
            if fin and fin != self._seen_finish:
                self._seen_finish = fin
                return True
            return False


def tail(path: str | Path | None, lines: int = 15) -> list[str]:
    if not path:
        return []
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 16384))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


JOB = EngineJob()  # one per app process

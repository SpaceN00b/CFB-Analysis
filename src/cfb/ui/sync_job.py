"""The Data page's sync, run in a background thread owned by the app process.

Streamlit stops a page's script when the user opens another page, but a call already inside
`warehouse_sync.sync` keeps running in the abandoned script thread, holding the lock, with no
page left to show it. Running the sync here instead keeps it going across page changes, lets
every page see that it is running, and lets the Data page pick the progress up again.
No Streamlit here, so it is testable without the UI.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

MAX_LINES = 200


@dataclass(frozen=True)
class Snapshot:
    state: str  # idle | running | done | failed
    lines: list[str] = field(default_factory=list)
    error: BaseException | None = None
    result: Any = None
    started: float | None = None
    finished: float | None = None

    @property
    def running(self) -> bool:
        return self.state == "running"


class SyncJob:
    def __init__(self):
        self._mu = threading.Lock()
        self._state, self._lines, self._error, self._result = "idle", [], None, None
        self._started = self._finished = None
        self._fresh = False
        self._thread: threading.Thread | None = None

    def start(self, run: Callable[[Callable[[str], None]], Any]) -> bool:
        """Start `run(log)` in a thread. False (and nothing started) when a sync is already running."""
        with self._mu:
            if self._state == "running":
                return False
            self._state, self._lines, self._error, self._result = "running", [], None, None
            self._started, self._finished = time.time(), None
            self._thread = threading.Thread(target=self._run, args=(run,), name="cfb-sync", daemon=True)
            self._thread.start()
            return True

    def _log(self, msg: str) -> None:
        with self._mu:
            self._lines.append(str(msg))
            del self._lines[:-MAX_LINES]

    def _run(self, run) -> None:
        try:
            result = run(self._log)
        except BaseException as exc:  # reported on the Data page, never raised into the app
            with self._mu:
                self._state, self._error, self._finished = "failed", exc, time.time()
        else:
            with self._mu:
                self._state, self._result, self._finished = "done", result, time.time()
                self._fresh = bool(getattr(result, "changed", True))

    def snapshot(self) -> Snapshot:
        with self._mu:
            return Snapshot(self._state, list(self._lines), self._error, self._result,
                            self._started, self._finished)

    @property
    def running(self) -> bool:
        with self._mu:
            return self._state == "running"

    def take_fresh(self) -> bool:
        """True once after a sync that switched to new data, so the app can clear its caches."""
        with self._mu:
            fresh, self._fresh = self._fresh, False
            return fresh

    def wait(self, timeout: float | None = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)


JOB = SyncJob()  # one per app process: shared by all pages, reruns and browser tabs

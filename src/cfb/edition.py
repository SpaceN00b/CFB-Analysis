"""Which edition of the app this is: `main` or `experimental`.

The two editions install and run side by side on one PC and share nothing, so an experimental
run can never write main's data, keys, schedule or bucket:

| | main | experimental |
| --- | --- | --- |
| App name | CFB Predictor | CFB Predictor Experimental |
| Data, logs (`%LOCALAPPDATA%\\<folder>`), user config (`%APPDATA%\\<folder>`) | cfb | cfb-exp |
| Keyring service (Windows Credential Manager) | cfb | cfb-exp |
| R2 bucket | cfb-warehouse | cfb-warehouse-exp |
| Scheduled tasks | "CFB Predictor weekly update" | "CFB Predictor Experimental weekly update", 30 min later |
| Release tags | build-<n> | exp-build-<n> |

The edition is set when the app is packaged (the Windows app workflow writes EDITION into
cfb/_build.py). $CFB_EDITION overrides it, for tests and source checkouts. Default: main.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

MAIN = "main"
EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class Edition:
    name: str
    app_name: str
    folder: str  # per-user data and config folder name
    keyring_service: str
    bucket: str
    tag_prefix: str  # GitHub Release tags: <tag_prefix>-<run number>
    schedule_offset_min: int  # added to the default schedule times, so editions never overlap

    @property
    def is_main(self) -> bool:
        return self.name == MAIN

    def task_name(self, kind: str) -> str:
        """Task Scheduler name for `weekly` or `daily`."""
        return f"{self.app_name} {'weekly update' if kind == 'weekly' else 'daily refresh'}"


EDITIONS = {
    MAIN: Edition(MAIN, "CFB Predictor", "cfb", "cfb", "cfb-warehouse", "build", 0),
    EXPERIMENTAL: Edition(EXPERIMENTAL, "CFB Predictor Experimental", "cfb-exp", "cfb-exp",
                          "cfb-warehouse-exp", "exp-build", 30),
}
MAIN_BUCKET = EDITIONS[MAIN].bucket


def _built() -> str | None:
    try:
        from . import _build  # written by the Windows app workflow
    except ImportError:
        return None
    return getattr(_build, "EDITION", None)


def current() -> Edition:
    name = (os.environ.get("CFB_EDITION") or _built() or MAIN).strip().lower()
    if name not in EDITIONS:
        raise ValueError(f"unknown edition {name!r}; known: {', '.join(EDITIONS)}")
    return EDITIONS[name]

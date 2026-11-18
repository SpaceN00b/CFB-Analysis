"""Update check for the packaged app: is there a newer GitHub Release than this build?

CI writes cfb/_build.py (BUILD = run number, COMMIT = git sha) into the packaged app and
tags each release `build-<BUILD>`. A source checkout has no _build.py and never offers updates.
"""
from __future__ import annotations

import re

import requests

RELEASES_URL = "https://api.github.com/repos/SpaceN00b/CFB-Analysis/releases/latest"
TAG = re.compile(r"^build-(\d+)$")


def current_build() -> int | None:
    try:
        from .. import _build  # written by the Windows app workflow
    except ImportError:
        return None
    return int(_build.BUILD)


def latest_release(timeout: float = 5.0, session=None) -> dict | None:
    """{"build", "tag", "url", "installer_url"} of the latest release, or None (offline, none yet)."""
    try:
        resp = (session or requests).get(RELEASES_URL, timeout=timeout,
                                         headers={"Accept": "application/vnd.github+json"})
        if resp.status_code != 200:
            return None
        doc = resp.json()
    except (requests.RequestException, ValueError):
        return None
    m = TAG.match(doc.get("tag_name") or "")
    if not m:
        return None
    installer = next((a.get("browser_download_url") for a in doc.get("assets", [])
                      if str(a.get("name", "")).lower().endswith(".exe")), None)
    return {"build": int(m.group(1)), "tag": doc["tag_name"], "url": doc.get("html_url"), "installer_url": installer}


def update_available(session=None) -> dict | None:
    """The latest release when it is newer than this build, else None."""
    cur = current_build()
    if cur is None:
        return None
    rel = latest_release(session=session)
    return rel if rel and rel["build"] > cur else None

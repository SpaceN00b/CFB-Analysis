"""Update check for the packaged app: is there a newer GitHub Release than this build?

CI writes cfb/_build.py (BUILD = run number, COMMIT = git sha, EDITION) into the packaged app
and tags each release `build-<BUILD>` (main) or `exp-build-<BUILD>` (experimental, published as
a pre-release, so GitHub's "latest" stays main's). Each edition only offers its own releases.
A source checkout has no _build.py and never offers updates.
"""
from __future__ import annotations

import re

import requests

from .. import edition

RELEASES_URL = "https://api.github.com/repos/SpaceN00b/CFB-Analysis/releases/latest"
RELEASES_LIST_URL = "https://api.github.com/repos/SpaceN00b/CFB-Analysis/releases?per_page=30"
TAG = re.compile(r"^build-(\d+)$")


def tag_re(ed: edition.Edition | None = None) -> re.Pattern:
    return re.compile(rf"^{re.escape((ed or edition.current()).tag_prefix)}-(\d+)$")


def current_build() -> int | None:
    try:
        from .. import _build  # written by the Windows app workflow
    except ImportError:
        return None
    return int(_build.BUILD)


def latest_release(timeout: float = 5.0, session=None) -> dict | None:
    """{"build", "tag", "url", "installer_url"} of this edition's newest release, or None
    (offline, none yet)."""
    ed = edition.current()
    url = RELEASES_URL if ed.is_main else RELEASES_LIST_URL
    try:
        resp = (session or requests).get(url, timeout=timeout,
                                         headers={"Accept": "application/vnd.github+json"})
        if resp.status_code != 200:
            return None
        doc = resp.json()
    except (requests.RequestException, ValueError):
        return None
    pat = tag_re(ed)
    if isinstance(doc, list):  # the list endpoint: newest of this edition, drafts skipped
        mine = [d for d in doc if not d.get("draft") and pat.match(d.get("tag_name") or "")]
        if not mine:
            return None
        doc = max(mine, key=lambda d: int(pat.match(d["tag_name"]).group(1)))
    m = pat.match(doc.get("tag_name") or "")
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

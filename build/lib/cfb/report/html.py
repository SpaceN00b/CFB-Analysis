"""The dashboard: one static HTML page with the bundle embedded. No server, no network
access when opened."""
from __future__ import annotations

import html
import json
from importlib import resources


def _template() -> str:
    return resources.files("cfb.report").joinpath("templates/dashboard.html").read_text(encoding="utf-8")


def render_dashboard(bundle: dict) -> str:
    data = json.dumps(bundle, separators=(",", ":"), allow_nan=False)
    # "<" can only occur inside JSON strings; escaping it keeps "</script>" and "<!--" in team
    # names or reasons from ending the script element early.
    data = data.replace("<", "\\u003c")
    title = html.escape(f"CFB {bundle['meta']['label']}")
    return _template().replace("__TITLE__", title, 1).replace("__BUNDLE__", data, 1)


__all__ = ["render_dashboard"]

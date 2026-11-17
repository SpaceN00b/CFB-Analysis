"""Weekly report and dashboard (W9).

    slate = resolve_slate(warehouse, now=...)                # or season=, week=
    bundle = build_bundle(warehouse, slate, predictions, config, data_dir="data")
    render_markdown(bundle), render_dashboard(bundle)

`python -m cfb.run_week` runs the whole weekly pipeline; see docs/report.md.
"""
from .bundle import build_bundle, model_label, write_bundle
from .config import ReportConfig, load_config
from .html import render_dashboard
from .markdown import render_markdown
from .slate import Slate, resolve_slate

__all__ = [
    "ReportConfig", "Slate", "build_bundle", "load_config", "model_label", "render_dashboard", "render_markdown",
    "resolve_slate", "write_bundle",
]

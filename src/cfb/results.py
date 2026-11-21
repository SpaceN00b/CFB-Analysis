"""A week's published picks next to the final scores, printed as Markdown.

    python -m cfb.results --season 2026 --week 12
    python -m cfb.results --season 2026 --week 12 --out week12_results.md --json week12_results.json

The picks come from the stored report (data/reports/<key>/bundle.json), never recomputed;
the code is in cfb.report.results.
"""
import sys

from .report.results import main

if __name__ == "__main__":
    sys.exit(main())

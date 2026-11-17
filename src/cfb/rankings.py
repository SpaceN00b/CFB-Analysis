"""Model top N vs the AP, Coaches and CFP top 25, printed as Markdown.

    python -m cfb.rankings                          # the current calendar week
    python -m cfb.rankings --season 2026 --week 12
    python -m cfb.rankings --season 2026 --week 12 --source elo --top 25

The code is in cfb.report.rankings; this module is the command-line entry point.
"""
import sys

from .report.rankings import main

if __name__ == "__main__":
    sys.exit(main())

"""Team strength features (W3, prefix str_).

`unit_ratings(view)` replaces the W1 placeholder for the `unit_ratings` table
(opponent-adjusted, garbage time removed, with a preseason prior). The feature groups
(power rating, preseason prior, Elo, schedule strength and record, luck, special teams,
recent form, polls) are listed in GROUPS. Method and parameters: docs/strength.md.
"""
from __future__ import annotations

from .groups import GROUPS
from .units import unit_ratings

__all__ = ["GROUPS", "unit_ratings"]

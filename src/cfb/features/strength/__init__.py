"""Team strength features (W3, prefix str_).

`unit_ratings(view)` replaces the W1 placeholder for the `unit_ratings` table
(opponent-adjusted, garbage time removed, with a preseason prior). Method and
parameters: docs/strength.md. The str_ feature groups follow in GROUPS.
"""
from __future__ import annotations

from .units import unit_ratings

GROUPS: list = []

__all__ = ["GROUPS", "unit_ratings"]

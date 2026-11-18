"""Team strength features (W3, prefix str_).

`unit_ratings(view)` replaces the W1 placeholder for the `unit_ratings` table
(opponent-adjusted, garbage time removed, with a preseason prior). The feature groups
(power rating, preseason prior, Elo, schedule strength and record, luck, special teams,
recent form, polls) are listed in GROUPS, followed by the model_v2 groups (player-based
power rating, its prior, program residual; groups_v2.py). Method and parameters: docs/strength.md.
"""
from __future__ import annotations

from .groups import GROUPS as _V1_GROUPS
from .groups_v2 import GROUPS_V2
from .playtext import PassAccuracy
from .units import unit_ratings

# str_pass_ci (detailed play text, 2025 week 9 on) is not used by any model: docs/player-maps.md.
GROUPS = _V1_GROUPS + GROUPS_V2 + [PassAccuracy()]

__all__ = ["GROUPS", "unit_ratings"]

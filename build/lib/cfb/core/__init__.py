"""Foundations shared by every feature, evaluation and model module.

    game_index, team_game, pbp   core tables built from the raw store (cfb.core.build)
    Warehouse, AsOfView          the only data access feature code may use (cfb.core.asof)
    leak                         scramble-after-cutoff leak test (cfb.core.leak)
    unit_ratings                 placeholder unit ratings in the final schema
    schemas                      prediction and metrics file contracts
    synthetic                    small fake CFBD store for tests without the snapshot
"""
from .asof import AsOfView, BannedTableError, MissingTableError, UnknownTableError, Warehouse
from .timeutil import RESULT_LAG, as_utc

__all__ = [
    "AsOfView",
    "BannedTableError",
    "MissingTableError",
    "UnknownTableError",
    "Warehouse",
    "RESULT_LAG",
    "as_utc",
]

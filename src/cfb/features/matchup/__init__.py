"""Matchup and play style features (W4), prefix mat_.

Groups:
    matchup_style     style profile, unit edges from unit_ratings, variance inputs
    matchup_history   how the team has done against styles like this opponent's

Definitions and the choice of shrinkage: docs/matchup.md.
"""
from .history import MatchupHistory
from .style import MatchupStyle

GROUPS = [MatchupStyle(), MatchupHistory()]

__all__ = ["GROUPS", "MatchupHistory", "MatchupStyle"]

"""Matchup and play style features (W4), prefix mat_.

Groups:
    matchup_style     style profile, unit edges from unit_ratings, variance inputs
    matchup_history   how the team has done against styles like this opponent's
    matchup_run_dir   run direction against the opponent's run defense by direction (play text)
    matchup_pass_zone pass depth and direction against the opponent's pass defense there (play text)

Definitions and the choice of shrinkage: docs/matchup.md.
"""
from .history import MatchupHistory
from .playtext import PassZone, RunDirection
from .style import MatchupStyle

GROUPS = [MatchupStyle(), MatchupHistory(), RunDirection(), PassZone()]

__all__ = ["GROUPS", "MatchupHistory", "MatchupStyle", "PassZone", "RunDirection"]

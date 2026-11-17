"""Context features (W6, prefix ctx_): site, rest, travel, venue, kickoff time, crowd,
season phase, portal departures, quarterback continuity and weather. See docs/context.md."""
from .crowd import Crowd
from .home_road import HomeRoad, LeagueHomeField
from .portal import Portal
from .qb import QBContinuity
from .schedule import ScheduleContext, SeasonPhase
from .travel import Travel, Venue
from .weather import Weather

GROUPS = [
    ScheduleContext(), SeasonPhase(), Travel(), Venue(), HomeRoad(), LeagueHomeField(), Crowd(), Portal(),
    QBContinuity(), Weather(),
]

__all__ = [
    "GROUPS", "Crowd", "HomeRoad", "LeagueHomeField", "Portal", "QBContinuity", "ScheduleContext", "SeasonPhase",
    "Travel", "Venue", "Weather",
]

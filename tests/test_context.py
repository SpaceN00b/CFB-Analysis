"""Context feature groups (W6): contract, values on hand-checked cases, leak tests."""
import numpy as np
import pandas as pd
import pytest

from cfb.core import Warehouse
from cfb.core.synthetic import synthetic_warehouse
from cfb.features.base import assert_group_no_leak, discover_groups, run_group, validate_group
from cfb.features.context import (
    GROUPS, Crowd, HomeRoad, LeagueHomeField, Portal, QBContinuity, ScheduleContext, SeasonPhase, Travel, Venue,
    Weather, home_road,
)
from cfb.features.context._util import haversine_km, utc_offset_hours
from cfb.features.context.crowd import reference_season


@pytest.fixture(scope="module")
def extended(tmp_path_factory) -> Warehouse:
    """The synthetic warehouse plus the tables the context groups read that it lacks."""
    wh = synthetic_warehouse(tmp_path_factory.mktemp("ctx"))
    gi = wh.game_index
    played = gi[gi["completed"]]
    rng = np.random.default_rng(3)
    rows = []
    for g in played.itertuples():
        for side, team in (("home", g.home_team), ("away", g.away_team)):
            # Starter: athlete <team>-1, except that team 1 switches to <team>-2 from 2022 week 3.
            starter = 2 if (g.home_id if side == "home" else g.away_id) == 1 and (g.season, g.week) >= (2022, 3) \
                and g.season_type == "regular" else 1
            for k, att in ((starter, 30), (3 - starter, 4)):
                rows.append({"gameId": g.game_id, "team": team, "conference": "", "homeAway": side,
                             "category": "passing", "statType": "C/ATT", "athleteId": f"{team}-{k}",
                             "athleteName": f"{team} {k}", "stat": f"{att // 2}/{att}", "season": g.season,
                             "season_type": g.season_type, "week": g.week})
            rows.append({"gameId": g.game_id, "team": team, "conference": "", "homeAway": side,
                         "category": "passing", "statType": "C/ATT", "athleteId": "-1", "athleteName": " Team",
                         "stat": "0/1", "season": g.season, "season_type": g.season_type, "week": g.week})
    wh.put("games_players", pd.DataFrame(rows))
    wh.put("weather", pd.DataFrame({
        "id": gi["game_id"], "season": gi["season"], "gameIndoors": gi["neutral"],
        "temperature": rng.uniform(30, 95, len(gi)), "windSpeed": rng.uniform(0, 20, len(gi)),
        "precipitation": rng.uniform(0, 0.2, len(gi)), "snowfall": 0.0, "humidity": rng.uniform(20, 90, len(gi)),
    }))
    post = gi[gi["season_type"] == "postseason"]
    entries = []
    for g in post.itertuples():
        for j in range(3):
            when = g.start_time_utc - pd.Timedelta(days=5 + j)
            entries.append({"season": g.season + 1, "firstName": "A", "lastName": str(j), "position": "WR",
                            "origin": g.home_team, "destination": None,
                            "transferDate": when.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                            "rating": None, "stars": float(2 + j), "eligibility": "Immediate"})
    wh.put("transfer_portal", pd.DataFrame(entries))
    return wh


def test_groups_are_registered_and_valid():
    names = {g.name for g in discover_groups()}
    for group in GROUPS:
        validate_group(group)
        assert group.name in names
        assert all(c.name.startswith("ctx_") for c in group.columns)


@pytest.mark.parametrize("group", GROUPS, ids=lambda g: g.name)
def test_context_groups_do_not_leak(group, extended):
    assert_group_no_leak(group, extended)


def test_home_road_does_not_leak_when_fitted(extended, monkeypatch):
    monkeypatch.setattr(home_road, "MIN_GAMES_FOR_FIT", 10)
    out = run_group(HomeRoad(), extended)
    assert out["ctx_home_resid"].notna().any()
    hfa = run_group(LeagueHomeField(), extended)
    assert hfa["ctx_league_hfa"].notna().any()
    assert_group_no_leak(HomeRoad(), extended)
    assert_group_no_leak(LeagueHomeField(), extended)


def test_schedule_values(extended):
    games = extended.schedule()
    out = run_group(ScheduleContext(), extended).merge(
        games[["game_id", "season", "season_type", "start_time_utc", "home_id", "neutral"]], on="game_id")
    home = out[out["team_id"] == out["home_id"]]
    assert (home["ctx_site"] == np.where(home["neutral"], 0, 1)).all()
    # Rest recomputed directly from the schedule.
    t = 1
    mine = out[out["team_id"] == t].sort_values("start_time_utc")
    for _, season in mine.groupby("season"):
        gaps = season["start_time_utc"].diff().dt.total_seconds() / 86400
        assert np.allclose(season["ctx_rest_days"].iloc[1:], gaps.iloc[1:])
        assert np.isnan(season["ctx_rest_days"].iloc[0])
        assert list(season["ctx_game_number"]) == list(range(1, len(season) + 1))
    post = out[out["season_type"] == "postseason"]
    assert post["ctx_days_since_regular"].notna().all()
    assert out[out["season_type"] == "regular"]["ctx_days_since_regular"].isna().all()
    phase = run_group(SeasonPhase(), extended)
    assert phase["ctx_season_week"].min() == 1


def test_travel_values(extended):
    out = run_group(Travel(), extended).merge(extended.schedule()[["game_id", "home_id", "neutral"]], on="game_id")
    on_campus_home = out[(out["team_id"] == out["home_id"]) & ~out["neutral"] & (out["team_id"] < 100)]
    assert (on_campus_home["ctx_travel_km"].abs() < 1e-6).all()
    assert (on_campus_home["ctx_at_own_stadium"] == 1).all()
    assert (on_campus_home["ctx_tz_shift_h"] == 0).all()
    bowl = out[out["neutral"] & (out["team_id"] < 100)]
    # The synthetic bowl is in Eastern time, every stadium in Central.
    assert (bowl["ctx_tz_shift_h"] == 1).all() and (bowl["ctx_dome_change"] == 1).all()
    venue = run_group(Venue(), extended)
    assert venue["ctx_kickoff_local_hour"].between(0, 24).all()


def test_haversine_and_offsets():
    # Ann Arbor to Columbus is about 260 km.
    assert 250 < haversine_km(42.2658, -83.7487, 40.0017, -83.0197) < 270
    when = pd.Series(pd.to_datetime(["2025-07-01T12:00:00Z", "2025-12-01T12:00:00Z"], utc=True))
    off = utc_offset_hours(pd.Series(["America/Denver", "America/Denver"]), when)
    assert list(off) == [-6.0, -7.0]
    assert np.isnan(utc_offset_hours(pd.Series(["Not/AZone"]), when.iloc[:1])[0])


def test_crowd(extended):
    assert reference_season(2021) == 2019 and reference_season(2025) == 2024
    out = run_group(Crowd(), extended).merge(extended.schedule()[["game_id", "season"]], on="game_id")
    assert out[out["season"] == 2021]["ctx_crowd_prior_att"].isna().all()  # 2019 is not in the data
    later = out[(out["season"] == 2023) & (out["team_id"] < 100)]
    assert later["ctx_crowd_prior_att"].notna().mean() > 0.7  # a team may have had no home game
    assert later["ctx_crowd_prior_att"].dropna().between(30, 50).all()
    assert later["ctx_crowd_prior_fill"].dropna().between(0.5, 1.5).all()


def test_qb_continuity(extended):
    out = run_group(QBContinuity(), extended).merge(
        extended.schedule()[["game_id", "season", "season_type", "week"]], on="game_id")
    t1 = out[(out["team_id"] == 1) & (out["season"] == 2022) & (out["season_type"] == "regular")].sort_values("week")
    first = t1[t1["week"] == 1].iloc[0]
    assert np.isnan(first["ctx_qb_leader_share"]) and np.isnan(first["ctx_qb_games_seen"])
    # Week 4: weeks 1-2 led by athlete 1, week 3 by athlete 2, so the last-game leader changed.
    w4 = t1[t1["week"] == 4].iloc[0]
    assert w4["ctx_qb_changed"] == 1 and w4["ctx_qb_leader_games"] == 1 and w4["ctx_qb_games_seen"] == 3
    assert w4["ctx_qb_leader_share"] == pytest.approx((30 + 4 + 4) / (3 * 34))
    assert w4["ctx_qb_passers_10"] == 2
    w5 = t1[t1["week"] == 5].iloc[0]
    assert w5["ctx_qb_changed"] == 0
    other = out[(out["team_id"] == 2) & (out["season"] == 2022) & (out["week"] == 4) & (out["season_type"] == "regular")]
    assert (other["ctx_qb_changed"] == 0).all() and np.allclose(other["ctx_qb_leader_share"], 30 / 34)


def test_portal(extended):
    out = run_group(Portal(), extended).merge(
        extended.schedule()[["game_id", "season_type", "home_id"]], on="game_id")
    bowl_home = out[(out["season_type"] == "postseason") & (out["team_id"] == out["home_id"])]
    assert (bowl_home["ctx_portal_out_since_last"] == 3).all()
    assert (bowl_home["ctx_portal_out_rated_since_last"] == 2).all()
    assert (bowl_home["ctx_portal_out_since_regular"] == 3).all()
    assert out[out["season_type"] == "regular"]["ctx_portal_out_since_regular"].isna().all()


def test_weather(extended):
    out = run_group(Weather(), extended)
    assert out["ctx_wx_temp_f"].notna().all() and set(out["ctx_wx_indoors"].unique()) <= {0.0, 1.0}


def test_context_groups_on_an_unscheduled_matchup(extended):
    """An on-demand matchup is in no table; groups must describe it from `games` alone."""
    sched = extended.schedule()
    slate = sched[(sched["season"] == 2023) & (sched["week"] == 4) & (sched["season_type"] == "regular")]
    game = slate.iloc[[0]].copy()
    game["game_id"] = 1
    game["home_id"], game["away_id"] = slate.iloc[0]["away_id"], slate.iloc[0]["home_id"]
    for group in GROUPS:
        out = run_group(group, extended, game)
        assert len(out) == (2 if group.level == "team" else 1), group.name


def test_groups_without_their_table_return_empty_values(synthetic):
    """The plain synthetic store has no portal, box score or weather table; the driver must
    still run, and the values are unknown rather than zero."""
    for group in (Portal(), QBContinuity(), Weather()):
        out = run_group(group, synthetic)
        assert len(out) and out[group.column_names].isna().all().all(), group.name


# ---- real data (skipped in CI) --------------------------------------------------------

@pytest.mark.parametrize("group", GROUPS, ids=lambda g: g.name)
def test_context_groups_do_not_leak_on_real_data(group, snapshot):
    assert_group_no_leak(group, snapshot, n=3)

"""AsOfView: what a view may and may not show."""
import numpy as np
import pandas as pd
import pytest

from cfb.core import BannedTableError, MissingTableError, UnknownTableError, Warehouse
from cfb.core.game_index import RESULT_COLUMNS
from cfb.core.policies import GAME, KINDS, POLICIES, SEASON_FINAL, TablePolicy
from cfb.core.timeutil import RESULT_LAG, as_utc

GAME_TABLES = ["game_index", "team_game", "pbp", "games", "drives", "plays", "games_teams",
               "stats_game_advanced", "ppa_games"]


def _mid_cutoff(wh, season=2022, week=3):
    gi = wh.game_index
    return gi[(gi["season"] == season) & (gi["week"] == week) & (gi["season_type"] == "regular")]["cutoff_utc"].iloc[0]


@pytest.mark.parametrize("name", GAME_TABLES)
def test_result_tables_stop_at_the_cutoff(synthetic, name):
    cutoff = _mid_cutoff(synthetic)
    view = synthetic.view(cutoff)
    frame = view.table(name)
    col = synthetic.policy(name).game_col
    start = synthetic.game_index.set_index("game_id")["start_time_utc"]
    seen = start.reindex(pd.unique(frame[col]))
    assert len(frame) and seen.notna().all()
    assert (seen + RESULT_LAG <= cutoff).all()
    visible = set(start[start + RESULT_LAG <= cutoff].index)
    with_rows = set(pd.unique(synthetic.prepared(name).frame[col]))
    assert set(pd.unique(frame[col])) == visible & with_rows  # nothing visible is missing


def test_view_grows_with_the_cutoff(synthetic):
    cutoffs = synthetic.cutoffs()
    sizes = [len(synthetic.view(c).table("team_game")) for c in cutoffs]
    keys = [synthetic.view(c).data_key for c in cutoffs]
    assert sizes == sorted(sizes) and keys == sorted(keys)
    assert sizes[0] == 0 and keys[0] == 0
    assert 2 * keys[-1] == sizes[-1]


def test_game_in_progress_is_hidden(synthetic):
    """The second bowl kicks off 26 hours after the first, so 24 hours before it the
    first has started but not finished."""
    gi = synthetic.game_index
    bowls = gi[(gi["season"] == 2021) & (gi["season_type"] == "postseason")].sort_values("start_time_utc")
    first, second = bowls.iloc[0], bowls.iloc[1]
    assert first["start_time_utc"] < second["cutoff_utc"] < first["start_time_utc"] + RESULT_LAG
    view = synthetic.view(second["cutoff_utc"])
    assert first["game_id"] not in set(view.table("team_game")["game_id"])
    later = synthetic.view(first["start_time_utc"] + RESULT_LAG)
    assert first["game_id"] in set(later.table("team_game")["game_id"])


def test_pregame_tables_cover_the_slate_but_not_later_games(synthetic):
    cutoff = _mid_cutoff(synthetic)
    view = synthetic.view(cutoff)
    gi = synthetic.game_index
    slate = set(gi[gi["cutoff_utc"] == cutoff]["game_id"])
    later = set(gi[gi["cutoff_utc"] > cutoff]["game_id"])
    for name, col in (("lines", "id"), ("game_pregame", "game_id")):
        ids = set(view.table(name)[col])
        assert slate <= ids and not (ids & later), name
    lines = view.table("lines")
    assert "homeScore" not in lines.columns and "awayScore" not in lines.columns
    assert not slate & set(view.table("games")["id"])  # the slate's results are not visible


def test_season_tables_are_banned_but_prior_seasons_are_served(synthetic):
    view = synthetic.view(_mid_cutoff(synthetic, 2022))
    for name, policy in POLICIES.items():
        if policy.kind == SEASON_FINAL:
            with pytest.raises(BannedTableError):
                view.table(name)
    assert set(view.prior_seasons("ratings_sp")["season"]) == {2021}
    assert set(view.prior_seasons("records")["season"]) == {2021}
    with pytest.raises(ValueError):
        view.prior_seasons("team_game")


def test_a_season_is_final_only_after_its_last_calendar_week(synthetic):
    weeks = synthetic.meta["weeks"]
    end_2021 = weeks[weeks["season"] == 2021]["end_utc"].max()
    during_bowls = synthetic.view(end_2021 - pd.Timedelta(days=3))
    assert during_bowls.season == 2021 and during_bowls.prior_seasons("ratings_sp").empty
    after = synthetic.view(end_2021 + pd.Timedelta(minutes=1))
    assert set(after.prior_seasons("ratings_sp")["season"]) == {2021}


def test_preseason_tables_include_the_cutoff_season(synthetic):
    opener = synthetic.game_index.query("season == 2022")["cutoff_utc"].min()
    view = synthetic.view(opener)
    assert view.season == 2022
    assert set(view.table("talent")["season"]) == {2021, 2022}
    assert set(view.table("teams", seasons=2022)["season"]) == {2022}
    summer = synthetic.view(opener - pd.Timedelta(days=40))
    assert summer.season == 2021 and set(summer.table("talent")["season"]) == {2021}
    planned = synthetic.view(opener - pd.Timedelta(days=40), season=2022)
    assert set(planned.table("talent")["season"]) == {2021, 2022}
    assert len(planned.table("team_game")) == len(summer.table("team_game"))  # results are unaffected


def test_unknown_and_missing_tables(synthetic):
    view = synthetic.view(_mid_cutoff(synthetic))
    with pytest.raises(UnknownTableError):
        view.table("something_new")
    with pytest.raises(MissingTableError):
        view.table("weather")
    assert view.has("team_game") and not view.has("weather") and not view.has("something_new")


def test_columns_and_seasons_arguments(synthetic):
    view = synthetic.view(synthetic.cutoffs()[-1])
    frame = view.table("team_game", columns=["game_id", "team_id", "points"], seasons=[2022])
    assert list(frame.columns) == ["game_id", "team_id", "points"]
    assert len(frame) == len(view.table("team_game", seasons=2022))
    full = view.table("team_game")
    assert set(full["season"]) == {2021, 2022, 2023}
    assert full["start_time_utc"].is_monotonic_increasing


def test_returned_frames_cannot_change_the_warehouse(synthetic):
    view = synthetic.view(synthetic.cutoffs()[-1])
    before = view.table("team_game")["points"].sum()
    frame = view.table("team_game")
    frame["points"] = -1.0
    frame.loc[frame.index[0], "margin"] = 999.0
    assert view.table("team_game")["points"].sum() == before
    assert (view.table("team_game")["margin"] != 999.0).all()


def test_schedule_has_no_results_and_future_is_opt_in(synthetic):
    cutoff = _mid_cutoff(synthetic, 2023, 2)
    view = synthetic.view(cutoff)
    sched = view.schedule()
    assert not set(RESULT_COLUMNS) & set(sched.columns)
    assert (sched["cutoff_utc"] <= cutoff).all() and (sched["cutoff_utc"] == cutoff).any()
    future = view.schedule(future=True)
    extra = future[future["cutoff_utc"] > cutoff]
    assert len(extra) and (extra["season"] == 2023).all() and (extra["season_type"] == "regular").all()


def test_future_schedule_leaves_out_title_games():
    def game(gid, season_type, week, start, conf):
        ts = as_utc(start)
        return {"game_id": gid, "season": 2024, "season_type": season_type, "week": week, "start_time_utc": ts,
                "cutoff_utc": ts - pd.Timedelta(days=2), "neutral": False, "conference_game": conf, "venue_id": 1,
                "home_id": 1, "away_id": 2, "home_fbs": True, "away_fbs": True, "completed": False,
                "home_points": np.nan, "away_points": np.nan, "home_team": "A", "away_team": "B", "start_time_tbd": False}
    gi = pd.DataFrame([
        game(1, "regular", 2, "2024-09-07T16:00Z", True),
        game(2, "regular", 13, "2024-11-30T17:00Z", True),
        game(3, "regular", 14, "2024-12-01T01:00Z", True),   # Saturday night Nov 30 in the US
        game(4, "regular", 15, "2024-12-07T21:00Z", True),   # conference title game
        game(5, "regular", 16, "2024-12-14T20:00Z", False),  # non-conference December game
        game(6, "postseason", 1, "2024-12-28T20:00Z", False),
    ])
    weeks = pd.DataFrame({"season": [2024], "season_type": ["regular"], "week": [1],
                          "start_utc": [as_utc("2024-08-20T07:00Z")], "end_utc": [as_utc("2025-01-20T07:00Z")]})
    wh = Warehouse(None, frames={"game_index": gi, "weeks": weeks})
    view = wh.view("2024-09-01T00:00Z")
    assert list(view.schedule()["game_id"]) == []
    assert list(view.schedule(future=True)["game_id"]) == [1, 2, 3, 5]


def test_unit_ratings_are_dated(synthetic):
    cutoff = _mid_cutoff(synthetic)
    view = synthetic.view(cutoff)
    assert view.table("unit_ratings")["cutoff_utc"].max() == cutoff
    latest = view.latest("unit_ratings", ["team_id", "unit"])
    assert (latest["cutoff_utc"] == cutoff).all() and not latest.duplicated(["team_id", "unit"]).any()


# ---- policies for tables that exist only in the full snapshot --------------------------

def _with(synthetic, **frames) -> Warehouse:
    wh = Warehouse(synthetic.root)
    for name, frame in frames.items():
        wh.put(name, frame)
    return wh


def test_weekly_tables_pre_and_post(synthetic):
    rankings = pd.DataFrame({"season": 2022, "seasonType": ["regular"] * 3 + ["postseason"], "week": [1, 2, 3, 1],
                             "poll": "AP Top 25", "teamId": 1, "rank": [5, 4, 3, 1]})
    elo = pd.DataFrame({"season": 2022, "week": [1, 2, 3], "team": "Team A", "elo": [1500, 1510, 1520]})
    wh = _with(synthetic, rankings=rankings, ratings_elo_weekly=elo)
    view = wh.view(_mid_cutoff(synthetic, 2022, 3))  # start of week 3
    assert list(view.table("rankings")["week"]) == [1, 2, 3]  # the week 3 poll is out before week 3
    assert list(view.table("ratings_elo_weekly")["week"]) == [1, 2]  # week 3 Elo includes week 3 games
    weeks = synthetic.meta["weeks"]
    post_end = weeks[(weeks["season"] == 2022) & (weeks["season_type"] == "postseason")]["end_utc"].iloc[0]
    assert len(wh.view(post_end).table("rankings")) == 3  # the final poll waits for the last bowl
    assert len(wh.view(post_end + pd.Timedelta(seconds=1)).table("rankings")) == 4


def test_dated_tables_and_portal_destination(synthetic):
    cutoff = _mid_cutoff(synthetic, 2022, 3)
    portal = pd.DataFrame({
        "season": [2022, 2023, 2023],
        "firstName": ["A", "B", "C"], "origin": ["Team A", "Team B", "Team C"],
        "destination": ["Team D", "Team E", "Team F"],
        "transferDate": [(cutoff - pd.Timedelta(days=200)).isoformat(), (cutoff - pd.Timedelta(days=1)).isoformat(),
                         (cutoff + pd.Timedelta(days=1)).isoformat()],
    })
    asof = pd.DataFrame({"season": 2022, "team": "Team A", "rating": [1.0, 2.0],
                         "asof_utc": [(cutoff - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                      (cutoff + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")]})
    wh = _with(synthetic, transfer_portal=portal, ratings_sp_asof=asof)
    view = wh.view(cutoff)
    seen = view.table("transfer_portal")
    assert list(seen["firstName"]) == ["A", "B"]
    assert seen["destination"].iloc[0] == "Team D" and pd.isna(seen["destination"].iloc[1])
    assert list(view.table("ratings_sp_asof")["rating"]) == [1.0]


def test_coach_tenures_hide_what_came_later(synthetic):
    cutoff = _mid_cutoff(synthetic, 2022, 3)
    tenures = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "coach.id": [10, 11, 12, 13],
        "team.id": [1, 1, 2, 2],
        "startYear": [2019, 2022, 2022, 2023],
        "endYear": [2021, 2022, np.nan, np.nan],
        "hireDate": [None, (cutoff - pd.Timedelta(days=250)).isoformat(), (cutoff + pd.Timedelta(days=20)).isoformat(), None],
        "isInterim": [False, False, True, False],
        "record.wins": [20, 3, 1, 0],
        "active": [False, False, True, True],
    })
    view = _with(synthetic, coach_tenures=tenures).view(cutoff)
    seen = view.table("coach_tenures")
    assert list(seen["id"]) == [1, 2]  # the interim hired later and the 2023 hire are not known yet
    assert "record.wins" not in seen.columns and "active" not in seen.columns
    assert seen["endYear"].iloc[0] == 2021 and pd.isna(seen["endYear"].iloc[1])  # a 2022 exit has not happened


def test_every_policy_is_well_formed():
    assert all(p.kind in KINDS for p in POLICIES.values())
    assert POLICIES["team_game"].kind == GAME
    with pytest.raises(ValueError):
        TablePolicy(GAME)
    with pytest.raises(ValueError):
        TablePolicy("sometimes")
    assert TablePolicy(GAME, game_col="id", hide=("record.*", "x")).hidden(["record.wins", "x", "y"]) == ["record.wins", "x"]


def test_put_requires_a_policy(synthetic):
    wh = Warehouse(synthetic.root)
    with pytest.raises(UnknownTableError):
        wh.put("made_up", pd.DataFrame({"a": [1]}))


# ---- against the real snapshot (skipped in CI) ---------------------------------------

def test_snapshot_views(snapshot):
    cutoff = as_utc("2025-10-06T07:00:00Z")  # start of 2025 week 7
    view = snapshot.view(cutoff)
    assert view.season == 2025
    tg = view.table("team_game")
    assert (tg["start_time_utc"] + RESULT_LAG <= cutoff).all() and len(tg) == 2 * view.data_key
    assert view.table("pbp", seasons=2025)["game_id"].isin(tg["game_id"]).all()
    assert view.table("talent")["season"].max() == 2025
    assert view.prior_seasons("ratings_sp")["season"].max() == 2024
    with pytest.raises(BannedTableError):
        view.table("stats_season_advanced")
    slate = view.schedule()
    assert slate["cutoff_utc"].max() == cutoff and not set(RESULT_COLUMNS) & set(slate.columns)


def test_snapshot_every_table_on_disk_has_a_policy(snapshot):
    on_disk = {p.name for p in (snapshot.root / "tables").iterdir() if p.is_dir()}
    assert not on_disk - set(POLICIES), f"tables without a policy: {sorted(on_disk - set(POLICIES))}"
    view = snapshot.view(as_utc("2025-10-06T07:00:00Z"))
    today = snapshot.view(snapshot.cutoffs()[-1])  # dated rating copies start on 2026-11-17
    for name in sorted(on_disk - {"plays"}):
        policy = POLICIES[name]
        if policy.kind == SEASON_FINAL:
            frame = view.prior_seasons(name)
        elif name == "depth_charts":  # imported snapshots, visible from their download time
            from cfb.store import Store
            last = as_utc(Store(snapshot.root).load(name)["asof_utc"].max())
            frame = snapshot.view(last).table(name)
        else:
            frame = (today if name.endswith("_asof") else view).table(name)
        assert len(frame), name


def test_snapshot_new_tables_respect_the_cutoff(snapshot):
    cutoff = as_utc("2025-10-06T07:00:00Z")  # start of 2025 week 7
    view = snapshot.view(cutoff)
    for name in ("games_players", "stats_game_havoc", "punting_games", "weather", "media", "rankings",
                 "ratings_elo_weekly", "transfer_portal", "coach_tenures", "recruiting_teams", "ratings_elo_preseason"):
        if not view.has(name):
            pytest.skip(f"snapshot has no {name}; restore the 2026-11-17 snapshot or later")
    played = set(view.table("team_game")["game_id"])
    for name in ("games_players", "stats_game_havoc", "punting_games"):
        assert set(view.table(name)["gameId"]) <= played, name
    slate_or_past = set(view.schedule()["game_id"])
    assert set(view.table("weather")["id"]) <= slate_or_past and set(view.table("media")["id"]) <= slate_or_past
    elo = view.table("ratings_elo_weekly")
    assert elo[elo["season"] == 2025]["week"].max() == 6  # week 7 Elo includes week 7 results
    polls = view.table("rankings")
    assert polls[polls["season"] == 2025]["week"].max() == 7 and set(polls[polls["season"] == 2025]["seasonType"]) == {"regular"}
    portal = view.table("transfer_portal")
    assert pd.to_datetime(portal["transferDate"], utc=True, format="ISO8601").max() <= cutoff
    assert portal[portal["season"] > 2025]["destination"].isna().all()
    tenures = view.table("coach_tenures")
    assert not [c for c in tenures.columns if c.startswith("record.")] and "active" not in tenures.columns
    hired = pd.to_datetime(tenures["hireDate"], utc=True, errors="coerce", format="ISO8601")
    assert (hired.dropna() <= cutoff).all() and (tenures["endYear"].dropna().astype(float) < 2025).all()
    assert view.table("recruiting_teams")["season"].max() == 2025
    assert view.table("ratings_sp_asof").empty and view.prior_seasons("ratings_core")["season"].max() == 2024

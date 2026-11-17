"""The leak test must pass honest code and catch planted leaks."""
import numpy as np
import pandas as pd
import pytest

from cfb.core.leak import LeakError, assert_no_leak, check_leak, diff_frames, lint_source, pick_cutoffs, scramble_after
from cfb.core.timeutil import RESULT_LAG
from cfb.core.unit_ratings import UNIT_RATINGS_COLUMNS, placeholder_unit_ratings
from cfb.features.base import FeatureColumn, FeatureGroup, assert_group_no_leak, check_group_leak

COLUMNS = [
    FeatureColumn("ctx_games_played", "Games the team has played this season", ("team_game",), 2014),
    FeatureColumn("ctx_points_avg", "Points per game this season", ("team_game",), 2014),
]


def _season_to_date(view, games, team_game):
    sides = pd.concat([
        games[["game_id", "home_id"]].rename(columns={"home_id": "team_id"}),
        games[["game_id", "away_id"]].rename(columns={"away_id": "team_id"}),
    ])
    agg = team_game.groupby("team_id")["points"].agg(ctx_games_played="count", ctx_points_avg="mean").reset_index()
    return sides.merge(agg, on="team_id", how="left")


class Honest(FeatureGroup):
    """Season-to-date scoring, read through the view at each cutoff."""
    name, prefix, level, columns = "honest", "ctx_", "team", COLUMNS

    def build(self, view, games):
        return _season_to_date(view, games, view.table("team_game", seasons=view.season))


class HonestBatch(FeatureGroup):
    """The same numbers computed in one pass, respecting each game's own cutoff."""
    name, prefix, level, mode, columns = "honest_batch", "ctx_", "team", "batch", COLUMNS

    def build(self, view, games):
        tg = view.table("team_game", columns=["team_id", "season", "start_time_utc", "points"]).dropna(subset=["points"])
        rows = []
        for g in games.itertuples():
            for team in (g.home_id, g.away_id):
                past = tg[(tg["team_id"] == team) & (tg["season"] == g.season)
                          & (tg["start_time_utc"] + RESULT_LAG <= g.cutoff_utc)]
                rows.append({"game_id": g.game_id, "team_id": team, "ctx_games_played": len(past),
                             "ctx_points_avg": past["points"].mean() if len(past) else np.nan})
        return pd.DataFrame(rows)


class LeakyBatch(FeatureGroup):
    """Planted leak: the season average includes games played after the one being predicted."""
    name, prefix, level, mode, columns = "leaky_batch", "ctx_", "team", "batch", COLUMNS

    def build(self, view, games):
        tg = view.table("team_game").dropna(subset=["points"])
        agg = tg.groupby(["team_id", "season"])["points"].agg(ctx_games_played="count", ctx_points_avg="mean").reset_index()
        sides = pd.concat([
            games[["game_id", "season", "home_id"]].rename(columns={"home_id": "team_id"}),
            games[["game_id", "season", "away_id"]].rename(columns={"away_id": "team_id"}),
        ])
        return sides.merge(agg, on=["team_id", "season"], how="left").drop(columns="season")


class Remembers(FeatureGroup):
    """Planted leak: keeps the first table it sees and reuses it at every later call."""
    name, prefix, level, columns = "remembers", "ctx_", "team", COLUMNS

    def __init__(self):
        self.cache = {}

    def build(self, view, games):
        if view.season not in self.cache:
            self.cache[view.season] = view.table("team_game", seasons=view.season)
        return _season_to_date(view, games, self.cache[view.season])


class PeeksAtSlateLines(FeatureGroup):
    """Reads the final score that the raw lines table carries. AsOfView hides those columns."""
    name, prefix, level = "peeks", "mkt_", "game"
    columns = [FeatureColumn("mkt_home_score", "Home score from the lines table", ("lines",), 2014)]

    def build(self, view, games):
        lines = view.table("lines")
        return pd.DataFrame({"game_id": games["game_id"], "mkt_home_score": games["game_id"].map(
            lines.set_index("id")["homeScore"])})


def test_scramble_changes_only_rows_after_the_cutoff(synthetic):
    cutoff = pick_cutoffs(synthetic, 3)[1]
    scrambled = scramble_after(synthetic, cutoff)
    real, fake = synthetic.prepared("team_game"), scrambled.prepared("team_game")
    k = int(np.searchsorted(real.avail, cutoff.value, side="right"))
    assert 0 < k < len(real.frame)
    pd.testing.assert_frame_equal(real.frame.iloc[:k], fake.frame.iloc[:k])
    keys = ["game_id", "team_id", "opp_id", "is_home", "season", "week", "start_time_utc"]
    pd.testing.assert_frame_equal(real.frame[keys], fake.frame[keys])
    tail_real, tail_fake = real.frame.iloc[k:], fake.frame.iloc[k:]
    played = tail_real["points"].notna()
    assert played.any() and (tail_real.loc[played, "points"] != tail_fake.loc[played, "points"]).all()
    assert scrambled.scrambled_rows["team_game"] == len(real.frame) - k
    # schedule facts and dtypes survive
    assert (real.frame.dtypes == fake.frame.dtypes).all()
    pd.testing.assert_frame_equal(synthetic.schedule(), scrambled.schedule())
    # the view at the cutoff is identical on both
    pd.testing.assert_frame_equal(synthetic.view(cutoff).table("team_game"), scrambled.view(cutoff).table("team_game"))


def test_scramble_reaches_every_kind_of_table(synthetic):
    cutoff = pick_cutoffs(synthetic, 3)[1]
    scrambled = scramble_after(synthetic, cutoff)
    for name in ("game_index", "pbp", "lines", "talent", "ratings_sp", "unit_ratings", "games"):
        real, fake = synthetic.prepared(name), scrambled.prepared(name)
        assert scrambled.scrambled_rows[name] > 0, name
        assert not real.frame.equals(fake.frame), name
    gi = scrambled.prepared("game_index").frame
    pd.testing.assert_frame_equal(gi[["game_id", "home_id", "away_id", "cutoff_utc"]],
                                  synthetic.prepared("game_index").frame[["game_id", "home_id", "away_id", "cutoff_utc"]])


def test_honest_groups_pass(synthetic):
    assert check_group_leak(Honest(), synthetic) == []
    assert check_group_leak(HonestBatch(), synthetic) == []
    assert_group_no_leak(Honest(), synthetic, n=5)


def test_planted_batch_leak_is_caught(synthetic):
    problems = check_group_leak(LeakyBatch(), synthetic)
    assert problems and any("ctx_points_avg" in p for p in problems)
    with pytest.raises(LeakError, match="leaky_batch"):
        assert_group_no_leak(LeakyBatch(), synthetic)


def test_state_kept_between_calls_is_caught(synthetic):
    with pytest.raises(LeakError, match="remembers"):
        assert_group_no_leak(Remembers(), synthetic)


def test_result_columns_of_pregame_tables_are_unreachable(synthetic):
    with pytest.raises(KeyError):
        check_group_leak(PeeksAtSlateLines(), synthetic)


def test_placeholder_unit_ratings_pass_the_leak_test(synthetic):
    build = lambda wh, cutoff: placeholder_unit_ratings(wh.view(cutoff))
    assert check_leak(build, synthetic, pick_cutoffs(synthetic, 4), ["team_id", "cutoff_utc", "unit"]) == []
    leaky = lambda wh, cutoff: placeholder_unit_ratings(wh.view(synthetic.cutoffs()[-1])).assign(
        cutoff_utc=cutoff)[UNIT_RATINGS_COLUMNS]
    with pytest.raises(LeakError):
        assert_no_leak(leaky, synthetic, pick_cutoffs(synthetic, 2), ["team_id", "cutoff_utc", "unit"])


def test_diff_frames_reports_changes():
    a = pd.DataFrame({"k": [1, 2], "x": [1.0, np.nan], "s": ["a", None]})
    assert diff_frames(a, a.copy(), ["k"]) == []
    assert diff_frames(a, a.iloc[::-1], ["k"]) == []  # row order does not matter
    b = a.assign(x=[1.0, 5.0])
    assert "x" in diff_frames(a, b, ["k"])[0]
    assert diff_frames(a, a.iloc[:1], ["k"])[0].startswith("row keys differ")
    assert diff_frames(a, a.rename(columns={"x": "y"}), ["k"])[0].startswith("columns differ")


@pytest.mark.parametrize("source, flagged", [
    ("from cfb.store import Store", True),
    ("from ...store import Store", True),
    ("from ..core.asof import Warehouse", True),
    ("import pandas as pd\npd.read_parquet('data/tables/games/2025_regular.parquet')", True),
    ("f = open('data/x.json')", True),
    ("import requests", True),
    ("def build(view, games):\n    return view._wh.game_index", True),
    ("import numpy as np\nimport pandas as pd\nfrom ..base import FeatureGroup\n"
     "def build(view, games):\n    return view.table('team_game')", False),
])
def test_source_lint(source, flagged):
    assert bool(lint_source(source)) is flagged


# ---- against the real snapshot (skipped in CI) ---------------------------------------

def test_snapshot_leak_test_catches_a_planted_leak(snapshot):
    cutoffs = [c for c in snapshot.cutoffs() if c.year == 2024][5:7]
    assert check_group_leak(Honest(), snapshot, cutoffs) == []
    problems = check_group_leak(LeakyBatch(), snapshot, cutoffs[:1])
    assert problems and any("ctx_points_avg" in p for p in problems)


def test_snapshot_placeholder_unit_ratings_do_not_leak(snapshot):
    build = lambda wh, cutoff: placeholder_unit_ratings(wh.view(cutoff))
    cutoffs = [c for c in snapshot.cutoffs() if c.year == 2023][8:10]
    assert check_leak(build, snapshot, cutoffs, ["team_id", "cutoff_utc", "unit"]) == []

"""Feature group interface, runner, catalog and the build driver."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cfb.features as features_pkg
from cfb.core.asof import MissingTableError
from cfb.core.leak import lint_paths
from cfb.core.unit_ratings import (
    UNIT_RATINGS_COLUMNS, UNITS, build_unit_ratings, placeholder_unit_ratings, resolve_builder, validate_unit_ratings,
)
from cfb.features import build as driver
from cfb.features.base import (
    FeatureColumn, FeatureError, FeatureGroup, catalog_table, check_group_leak, discover_groups, expected_keys,
    run_group, validate_group, validate_output, write_catalog,
)


class Rest(FeatureGroup):
    name, prefix, level = "rest", "ctx_", "team"
    columns = [FeatureColumn("ctx_rest_days", "Days since the team's previous game", ("schedule",), 2014)]

    def build(self, view, games):
        sched = view.schedule()
        long = pd.concat([
            sched[["game_id", "home_id", "start_time_utc"]].rename(columns={"home_id": "team_id"}),
            sched[["game_id", "away_id", "start_time_utc"]].rename(columns={"away_id": "team_id"}),
        ]).sort_values("start_time_utc")
        long["ctx_rest_days"] = long.groupby("team_id")["start_time_utc"].diff().dt.total_seconds() / 86400
        return long[long["game_id"].isin(games["game_id"])][["game_id", "team_id", "ctx_rest_days"]]


class EloGap(FeatureGroup):
    name, prefix, level = "elo_gap", "mkt_", "game"
    columns = [FeatureColumn("mkt_elo_gap", "Home minus away pregame Elo", ("game_pregame",), 2014)]

    def build(self, view, games):
        elo = view.table("game_pregame")
        out = games[["game_id"]].merge(elo, on="game_id", how="inner")
        out["mkt_elo_gap"] = out["home_pregame_elo"] - out["away_pregame_elo"]
        return out.dropna(subset=["mkt_elo_gap"])[["game_id", "mkt_elo_gap"]]  # FCS games have no row


def test_run_group_returns_one_row_per_key(synthetic):
    games = synthetic.schedule()
    out = run_group(Rest(), synthetic)
    assert list(out.columns) == ["game_id", "team_id", "ctx_rest_days"]
    assert len(out) == 2 * len(games) and not out.duplicated(["game_id", "team_id"]).any()
    first = out.merge(games[["game_id", "season", "season_type", "week"]], on="game_id")
    opener = (first["season"] == 2021) & (first["season_type"] == "regular") & (first["week"] == 1)
    assert first[opener]["ctx_rest_days"].isna().all()
    assert (out["ctx_rest_days"].dropna() > 0).all()

    gap = run_group(EloGap(), synthetic)
    assert len(gap) == len(games)  # rows the group left out come back empty
    fcs = games[~(games["home_fbs"] & games["away_fbs"])]["game_id"]
    assert gap[gap["game_id"].isin(fcs)]["mkt_elo_gap"].isna().all()
    assert gap[~gap["game_id"].isin(fcs)]["mkt_elo_gap"].notna().all()


def test_run_group_passes_schedule_only(synthetic):
    seen = {}

    class Spy(Rest):
        def build(self, view, games):
            seen.setdefault("columns", set()).update(games.columns)
            assert (games["cutoff_utc"] == view.cutoff).all()
            return super().build(view, games)

    run_group(Spy(), synthetic, synthetic.schedule([2022]))
    assert not {"completed", "home_points", "away_points"} & seen["columns"]
    assert {"game_id", "home_id", "away_id", "cutoff_utc", "start_time_utc", "neutral"} <= seen["columns"]


def test_expected_keys_order(synthetic):
    games = synthetic.schedule().head(3)
    keys = expected_keys(Rest(), games)
    assert list(keys["game_id"]) == [g for g in games["game_id"] for _ in range(2)]
    assert list(keys["team_id"][:2]) == [games["home_id"].iloc[0], games["away_id"].iloc[0]]


def _group(**overrides):
    attrs = {"name": "x", "prefix": "ctx_", "level": "team",
             "columns": [FeatureColumn("ctx_a", "A thing", ("team_game",), 2014)]}
    attrs.update(overrides)
    return type("G", (FeatureGroup,), attrs)()


@pytest.mark.parametrize("overrides, message", [
    ({"name": "Bad Name"}, "snake case"),
    ({"prefix": "foo_"}, "prefix"),
    ({"level": "player"}, "level"),
    ({"mode": "sometimes"}, "mode"),
    ({"columns": []}, "no columns"),
    ({"columns": [FeatureColumn("str_a", "A", ("team_game",), 2014)]}, "must start with"),
    ({"columns": [FeatureColumn("ctx_a", " ", ("team_game",), 2014)]}, "no description"),
    ({"columns": [FeatureColumn("ctx_a", "A", (), 2014)]}, "no sources"),
    ({"columns": [FeatureColumn("ctx_a", "A", ("mystery_table",), 2014)]}, "unknown sources"),
    ({"columns": [FeatureColumn("ctx_a", "A", ("team_game",), 2014)] * 2}, "listed twice"),
])
def test_validate_group_rejects(overrides, message):
    with pytest.raises(FeatureError, match=message):
        validate_group(_group(**overrides))


def test_validate_output_rejects(synthetic):
    group = _group()
    games = synthetic.schedule().head(2)
    good = expected_keys(group, games).assign(ctx_a=1.0)
    assert len(validate_output(group, good, games)) == 4
    assert validate_output(group, good.head(1), games)["ctx_a"].isna().sum() == 3
    cases = {
        "key columns": good.drop(columns="team_id"),
        "missing from the catalog": good.assign(ctx_b=1.0),
        "missing from the output": good.drop(columns="ctx_a"),
        "duplicate rows": pd.concat([good, good]),
        "numeric or boolean": good.assign(ctx_a="high"),
        "not among the requested games": good.assign(game_id=good["game_id"] + 10_000),
    }
    for message, frame in cases.items():
        with pytest.raises(FeatureError, match=message):
            validate_output(group, frame, games)
    with pytest.raises(FeatureError, match="DataFrame"):
        validate_output(group, None, games)


def test_catalog(tmp_path):
    groups = [Rest(), EloGap()]
    table = catalog_table(groups)
    assert list(table["column"]) == ["ctx_rest_days", "mkt_elo_gap"]
    assert set(table.columns) == {"group", "level", "column", "description", "sources", "first_season"}
    write_catalog(groups, tmp_path / "catalog.json")
    doc = json.loads((tmp_path / "catalog.json").read_text())
    assert [g["name"] for g in doc["groups"]] == ["rest", "elo_gap"]
    assert doc["groups"][0]["columns"][0] == {"name": "ctx_rest_days", "description": "Days since the team's previous game",
                                               "sources": ["schedule"], "first_season": 2014}


# ---- unit_ratings ------------------------------------------------------------------------

def test_placeholder_unit_ratings_schema(synthetic):
    cutoff = synthetic.game_index.query("season == 2022 and week == 4")["cutoff_utc"].iloc[0]
    view = synthetic.view(cutoff)
    ratings = validate_unit_ratings(placeholder_unit_ratings(view))
    assert list(ratings.columns) == UNIT_RATINGS_COLUMNS
    assert set(ratings["unit"]) == set(UNITS) and (ratings["cutoff_utc"] == cutoff).all()
    teams = view.table("teams", seasons=2022)["team_id"]
    assert len(ratings) == len(UNITS) * teams.nunique()
    one = ratings[ratings["team_id"] == 1].set_index("unit")
    tg = view.table("team_game", seasons=2022)
    tg = tg[tg["team_id"] == 1]
    assert one.loc["off_rush", "n_plays"] == tg["pbp_off_rush_plays"].sum()
    assert one.loc["off_all", "epa_adj"] == pytest.approx(tg["pbp_off_all_epa_sum"].sum() / tg["pbp_off_all_plays"].sum())
    assert one.loc["def_pass", "success_adj"] == pytest.approx(
        tg["pbp_def_pass_successes"].sum() / tg["pbp_def_pass_plays"].sum())
    assert one.loc["off_all", "n_plays"] == one.loc["off_rush", "n_plays"] + one.loc["off_pass", "n_plays"]


def test_unit_ratings_are_empty_before_a_team_has_played(synthetic):
    opener = synthetic.game_index.query("season == 2022")["cutoff_utc"].min()
    ratings = placeholder_unit_ratings(synthetic.view(opener))
    assert (ratings["n_plays"] == 0).all()
    assert ratings[["epa_adj", "success_adj", "explosive_adj"]].isna().all().all()


def test_unit_ratings_cover_every_cutoff(synthetic):
    ratings = synthetic.view(synthetic.cutoffs()[-1]).table("unit_ratings")
    assert set(ratings["cutoff_utc"]) == set(synthetic.cutoffs())
    assert resolve_builder()[1] in ("placeholder", "cfb.features.strength.unit_ratings")
    custom = lambda view: placeholder_unit_ratings(view).assign(epa_adj=0.0)
    out = build_unit_ratings(synthetic, synthetic.cutoffs()[:2], builder=custom)
    assert (out["epa_adj"] == 0.0).all() and out["cutoff_utc"].nunique() == 2


def test_validate_unit_ratings_rejects(synthetic):
    good = placeholder_unit_ratings(synthetic.view(synthetic.cutoffs()[5]))
    for message, frame in {
        "missing columns": good.drop(columns="n_plays"),
        "unknown units": good.assign(unit="special_teams"),
        "duplicate": pd.concat([good, good]),
        "timezone-aware": good.assign(cutoff_utc=good["cutoff_utc"].dt.tz_localize(None)),
        "non-negative": good.assign(n_plays=-1),
    }.items():
        with pytest.raises(ValueError, match=message):
            validate_unit_ratings(frame)


# ---- driver --------------------------------------------------------------------------------

def test_driver_builds_and_writes(synthetic, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(driver, "discover_groups", lambda: [Rest(), EloGap()])
    assert driver.main(["--data-dir", str(synthetic.root), "--seasons", "2022-2023", "--skip-unit-ratings"]) == 0
    out = synthetic.root / "features"
    rest = pd.read_parquet(out / "rest.parquet")
    games = synthetic.schedule([2022, 2023])
    assert len(rest) == 2 * len(games) and set(rest["game_id"]) == set(games["game_id"])
    assert len(pd.read_parquet(out / "elo_gap.parquet")) == len(games)
    assert [g["name"] for g in json.loads((out / "catalog.json").read_text())["groups"]] == ["rest", "elo_gap"]
    assert driver.main(["--data-dir", str(synthetic.root), "--list"]) == 0
    assert "ctx_rest_days" in capsys.readouterr().out
    assert driver.main(["--data-dir", str(synthetic.root), "--leak-test", "--skip-unit-ratings"]) == 0
    assert "rest: ok" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        driver.main(["--data-dir", str(synthetic.root), "--groups", "nope"])


def test_driver_rebuilds_missing_core_and_unit_ratings(tmp_path, capsys):
    from cfb.core.synthetic import write_synthetic_store

    write_synthetic_store(tmp_path)
    assert driver.main(["--data-dir", str(tmp_path)]) == 0
    assert (tmp_path / "core" / "game_index.parquet").exists()
    ratings = pd.read_parquet(tmp_path / "derived" / "unit_ratings.parquet")
    assert list(ratings.columns) == UNIT_RATINGS_COLUMNS and len(ratings)
    assert driver.main(["--tables"]) == 0
    assert "team_game" in capsys.readouterr().out
    assert driver.parse_seasons("2019,2021-2023") == [2019, 2021, 2022, 2023] and driver.parse_seasons("") is None


# ---- every registered group, as the feature threads add them ---------------------------------

def test_registered_groups_are_valid_and_unique():
    groups = discover_groups()
    assert len({g.name for g in groups}) == len(groups)


def test_registered_feature_sources_only_use_the_view():
    root = Path(features_pkg.__file__).parent
    areas = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    assert lint_paths(areas) == []


def test_registered_groups_pass_the_leak_test_on_synthetic_data(synthetic):
    """Runs for every group in cfb.features.*. A group that needs a table the synthetic
    store does not have is skipped here and must be leak-tested on the snapshot."""
    skipped = []
    for group in discover_groups():
        try:
            assert check_group_leak(group, synthetic) == [], group.name
        except MissingTableError:
            skipped.append(group.name)
    if skipped:
        pytest.skip(f"need tables outside the synthetic store: {skipped}")


def test_snapshot_group_run(snapshot):
    games = snapshot.schedule([2025])
    out = run_group(EloGap(), snapshot, games)
    fbs = games[games["home_fbs"] & games["away_fbs"]]
    assert len(out) == len(games)
    assert out[out["game_id"].isin(fbs["game_id"])]["mkt_elo_gap"].notna().all()
    assert np.isfinite(out["mkt_elo_gap"].dropna()).all()

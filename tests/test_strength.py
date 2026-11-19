"""Team strength (W3): unit ratings engine and the str_ feature groups."""
import numpy as np
import pandas as pd
import pytest

from cfb.core.leak import check_leak, lint_paths, pick_cutoffs
from cfb.core.unit_ratings import UNITS, resolve_builder, validate_unit_ratings
from cfb.features.strength import ridge, unit_ratings
from cfb.features.strength import units as U

KEYS = ["team_id", "cutoff_utc", "unit"]


# ---- ridge ---------------------------------------------------------------------------

def test_ridge_recovers_known_ratings():
    rng = np.random.default_rng(0)
    n = 12
    off, deff = rng.normal(0, 0.1, n), rng.normal(0, 0.1, n)
    off, deff = off - off.mean(), deff - deff.mean()
    o = rng.integers(0, n, 4000)
    d = (o + rng.integers(1, n, 4000)) % n
    h = rng.choice([-1.0, 0.0, 1.0], 4000)
    y = 0.05 + 0.02 * h + off[o] + deff[d] + rng.normal(0, 0.05, 4000)
    f = ridge.fit(n, o, d, h, y, np.full(4000, 60.0), lam=1.0)
    assert abs(f.mu - 0.05) < 0.01 and abs(f.hfa - 0.02) < 0.005
    assert np.corrcoef(f.off, off)[0, 1] > 0.99 and np.corrcoef(f.deff, deff)[0, 1] > 0.99


def test_ridge_returns_prior_without_data():
    off0, def0 = np.array([0.1, -0.1, 0.0]), np.array([0.0, 0.2, -0.2])
    f = ridge.fit(3, [], [], [], [], [], lam=100.0, off0=off0, def0=def0, mu0=0.04, lam_mu=10.0)
    assert f.mu == pytest.approx(0.04)
    assert np.allclose(f.off, off0) and np.allclose(f.deff, def0)
    empty = ridge.fit(3, [], [], [], [], [], lam=100.0)
    assert np.isnan(empty.mu) and np.isnan(empty.off).all()


def test_ridge_adjusts_for_opponents():
    # Team 0 scores 0.3 against a weak defense (2), team 1 scores 0.1 against a strong
    # one (3). Defenses 2 and 3 are measured against a common third offense (4).
    o = np.array([0, 1, 4, 4])
    d = np.array([2, 3, 2, 3])
    y = np.array([0.3, 0.1, 0.25, -0.05])
    f = ridge.fit(5, o, d, np.zeros(4), y, np.full(4, 1000.0), lam=1.0)
    assert f.off[1] > f.off[0] - 0.05  # raw gap 0.2 shrinks after adjustment
    assert f.deff[2] > f.deff[3]


# ---- unit_ratings --------------------------------------------------------------------

def test_driver_uses_w3_unit_ratings():
    assert resolve_builder()[1] == "cfb.features.strength.unit_ratings"


def test_unit_ratings_meet_the_contract(synthetic):
    for cutoff in synthetic.cutoffs()[::5]:
        out = validate_unit_ratings(unit_ratings(synthetic.view(cutoff)))
        assert set(out["unit"]) == set(UNITS)
        assert (out["cutoff_utc"] == cutoff).all()
        assert not out.duplicated(["team_id", "unit"]).any()


def test_unit_ratings_have_prior_values_before_the_first_game(synthetic):
    opener = synthetic.game_index.query("season == 2022")["cutoff_utc"].min()
    out = unit_ratings(synthetic.view(opener))
    assert (out["n_plays"] == 0).all()
    assert out[["epa_adj", "success_adj", "explosive_adj"]].notna().all().all()
    first = synthetic.game_index["cutoff_utc"].min()   # no earlier season: unknown
    assert unit_ratings(synthetic.view(first))["epa_adj"].isna().all()


def test_unit_ratings_count_non_garbage_plays(synthetic):
    cutoff = synthetic.cutoffs()[-1]
    view = synthetic.view(cutoff)
    out = unit_ratings(view).set_index(["team_id", "unit"])
    pbp = view.table("pbp", seasons=view.season)
    keep = pbp[pbp["is_scrimmage"] & ~pbp["garbage_time"] & pbp["is_rush"]]
    team = int(keep["offense_id"].iloc[0])
    assert out.loc[(team, "off_rush"), "n_plays"] == int((keep["offense_id"] == team).sum())
    assert out.loc[(team, "off_all"), "n_plays"] == out.loc[(team, "off_rush"), "n_plays"] + out.loc[(team, "off_pass"), "n_plays"]


def test_observations_drop_garbage_time(synthetic):
    view = synthetic.view(synthetic.cutoffs()[-1])
    pbp = view.table("pbp", columns=U.PBP_COLUMNS, seasons=view.season).copy()
    pbp["garbage_time"] = False
    pbp.loc[pbp.index[::3], "garbage_time"] = True
    obs = U.observations(pbp, view.schedule())
    assert obs["all_plays"].sum() == (pbp["is_scrimmage"] & ~pbp["garbage_time"]).sum()
    assert set(obs["h"]) <= {-1.0, 0.0, 1.0}


def test_unit_ratings_do_not_leak(synthetic):
    build = lambda wh, cutoff: unit_ratings(wh.view(cutoff))
    assert check_leak(build, synthetic, pick_cutoffs(synthetic, 4), KEYS) == []


def test_strength_sources_pass_the_lint():
    import cfb.features.strength as pkg
    from pathlib import Path
    assert lint_paths([Path(pkg.__file__).parent]) == []


# ---- against the real snapshot (skipped in CI) ---------------------------------------

def test_snapshot_unit_ratings_do_not_leak(snapshot):
    build = lambda wh, cutoff: unit_ratings(wh.view(cutoff))
    cutoffs = [c for c in snapshot.cutoffs() if c.year == 2023][3:5] + [c for c in snapshot.cutoffs() if c.year == 2024][-2:]
    assert check_leak(build, snapshot, cutoffs, KEYS) == []


def test_snapshot_unit_ratings_are_plausible(snapshot):
    view = snapshot.view([c for c in snapshot.cutoffs() if c.year == 2024 and c.month == 11][-1])
    out = validate_unit_ratings(unit_ratings(view))
    teams = view.table("teams", seasons=view.season)
    fbs = set(teams.loc[teams["fbs"], "team_id"])
    off = out[(out["unit"] == "off_all") & out["team_id"].isin(fbs)]
    assert off["epa_adj"].notna().all() and off["epa_adj"].std() > 0.05
    assert (off["n_plays"] > 400).mean() > 0.9


# ---- feature groups ------------------------------------------------------------------

from cfb.core import Warehouse  # noqa: E402
from cfb.features.base import assert_group_no_leak, run_group  # noqa: E402
from cfb.features.strength import GROUPS  # noqa: E402
from cfb.features.strength.groups import Polls, Power, Schedule  # noqa: E402


def _by_name(name):
    return next(g for g in GROUPS if g.name == name)


def test_groups_are_registered():
    names = {g.name for g in GROUPS}
    assert names == {"str_power", "str_prior", "str_elo", "str_schedule", "str_luck", "str_special_teams",
                     "str_form", "str_polls", "str_power2", "str_prior2", "str_program", "str_qb",
                     "str_pass_ci"}
    assert all(c.name.startswith("str_") for g in GROUPS for c in g.columns)


def test_power_rating_fades_from_prior(synthetic):
    out = run_group(Power(), synthetic).merge(synthetic.schedule()[["game_id", "season"]], on="game_id")
    s = out[out["season"] == 2022]
    assert np.allclose(s["str_pwr_net"], s["str_pwr_off"] - s["str_pwr_def"], equal_nan=True)
    fbs = s[s["str_pwr_net"].notna()]
    assert fbs["str_pwr_net"].std() > 0
    early, late = fbs[fbs["str_pwr_games"] == 0], fbs[fbs["str_pwr_games"] >= 3]
    assert (early["str_pwr_prior_share"] == 1).all()
    assert (late["str_pwr_prior_share"] < 0.6).all()
    assert (late["str_pwr_net_se"] < early["str_pwr_net_se"].min()).all()


def test_schedule_counts_and_sos(synthetic):
    out = run_group(Schedule(), synthetic)
    played = run_group(Power(), synthetic)["str_pwr_games"]
    fbs = out["str_wins"].notna()
    assert ((out["str_wins"] + out["str_losses"])[fbs] <= played[fbs] + 1e-9).all()
    assert out["str_sos_remaining"].notna().any() and out["str_sos_played"].notna().any()


def test_polls_lag_the_committee_ranking(synthetic, tmp_path):
    wh = Warehouse(synthetic.root)
    gi = wh.game_index
    season = 2022
    weeks = sorted(gi[(gi["season"] == season) & (gi["season_type"] == "regular")]["week"].unique())
    w = int(weeks[2])
    team = int(gi[(gi["season"] == season) & (gi["week"] == w)]["home_id"].iloc[0])
    rows = []
    for week in weeks[: 3]:
        rows.append({"season": season, "seasonType": "regular", "week": int(week), "poll": "AP Top 25",
                     "isFinal": None, "rank": int(week), "teamId": team, "school": "x", "conference": "x",
                     "firstPlaceVotes": 0.0, "points": 100.0 * int(week)})
        rows.append({"season": season, "seasonType": "regular", "week": int(week), "poll": "Playoff Committee Rankings",
                     "isFinal": None, "rank": 10 + int(week), "teamId": team, "school": "x", "conference": "x",
                     "firstPlaceVotes": 0.0, "points": None})
    wh.put("rankings", pd.DataFrame(rows))
    games = wh.schedule([season])
    slate = games[(games["week"] == w) & (games["season_type"] == "regular")]
    out = run_group(Polls(), wh, slate)
    row = out[out["team_id"] == team].iloc[0]
    assert row["str_poll_ap_rank"] == w and row["str_poll_ap_points"] == 100.0 * w
    assert row["str_poll_cfp_rank"] == 10 + int(weeks[1])          # last week's committee ranking
    other = out[out["team_id"] != team]
    assert (other["str_poll_ap_points"] == 0).all() and other["str_poll_ap_rank"].isna().all()
    assert (other["str_poll_ranked"] == 0).all() and row["str_poll_ranked"] == 1
    assert_group_no_leak(Polls(), wh, n=2)


# ---- groups on the real snapshot (skipped in CI) -------------------------------------

def test_snapshot_groups_do_not_leak(snapshot):
    cutoffs = [c for c in snapshot.cutoffs() if c.year == 2024][6:7] + [c for c in snapshot.cutoffs() if c.year == 2025][-1:]
    for group in GROUPS:
        assert_group_no_leak(group, snapshot, cutoffs)


def test_snapshot_power_is_plausible(snapshot):
    games = snapshot.schedule([2024])
    late = games[(games["season_type"] == "regular") & (games["week"] == 12)]
    out = run_group(Power(), snapshot, late)
    teams = snapshot.view(late["cutoff_utc"].iloc[0]).table("teams", seasons=2024)
    fbs = out[out["team_id"].isin(teams.loc[teams["fbs"], "team_id"])]
    assert fbs["str_pwr_net"].notna().all()
    assert 10 < fbs["str_pwr_net"].std() < 25
    assert (fbs["str_pwr_games"] >= 8).mean() > 0.9

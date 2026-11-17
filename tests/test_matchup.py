"""Matchup and play style features (W4)."""
import numpy as np
import pandas as pd
import pytest

from cfb.core.synthetic import synthetic_warehouse
from cfb.features.base import assert_group_no_leak, run_group, validate_group
from cfb.features.matchup import GROUPS, MatchupHistory, MatchupStyle
from cfb.features.matchup.history import HISTORY_SHRINK, game_residuals
from cfb.features.matchup.stats import _NEEDED, PRIOR_GAMES, blended_profiles, game_stats
from cfb.features.matchup.style import _EDGE_KINDS


def _havoc_frame(wh) -> pd.DataFrame:
    """A stats_game_havoc table for the synthetic games (keyed by team name, like CFBD)."""
    rng = np.random.default_rng(3)
    gi = wh.game_index
    gi = gi[gi["completed"]]
    rows = []
    for g in gi.itertuples():
        plays = rng.integers(55, 80, size=2)
        events = rng.integers(5, 20, size=2)
        for i, (team, opp) in enumerate(((g.home_team, g.away_team), (g.away_team, g.home_team))):
            rows.append({"gameId": g.game_id, "season": g.season, "seasonType": g.season_type, "week": g.week,
                         "team": team, "opponent": opp,
                         "offense.totalPlays": int(plays[i]), "offense.totalHavocEvents": float(events[i]),
                         "defense.totalPlays": int(plays[1 - i]), "defense.totalHavocEvents": float(events[1 - i])})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def wh(tmp_path_factory):
    warehouse = synthetic_warehouse(tmp_path_factory.mktemp("matchup"))
    warehouse.put("stats_game_havoc", _havoc_frame(warehouse))
    return warehouse


def _mid_season(wh, season=2022, week=4):
    games = wh.schedule([season])
    cutoff = games.query("season_type == 'regular' and week == @week")["cutoff_utc"].iloc[0]
    return cutoff, games[games["cutoff_utc"] == cutoff].reset_index(drop=True)


def test_groups_are_valid():
    for group in GROUPS:
        validate_group(group)
        assert group.prefix == "mat_" and group.level == "team"
    names = [c for g in GROUPS for c in g.column_names]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("group", GROUPS, ids=lambda g: g.name)
def test_no_leak_with_havoc(wh, group):
    assert_group_no_leak(group, wh, n=3)


def test_blend_counts_prior_season_as_games():
    stats = pd.DataFrame({
        "season": [2021, 2021, 2022], "team_id": [1, 1, 1], "game_id": [1, 2, 3],
        "off_rush": [30.0, 30.0, 10.0], "off_scrim": [60.0, 60.0, 50.0],
    })
    full = stats.reindex(columns=stats.columns.union(_NEEDED), fill_value=0.0)
    prof = blended_profiles(full, 2022, prior_games=4.0)
    # prior per game: 30 rushes of 60 plays; current: 10 of 50
    assert prof.loc[1, "off_run_rate"] == pytest.approx((10 + 4 * 30) / (50 + 4 * 60))
    assert prof.loc[1, "games"] == 1
    no_prior = blended_profiles(full, 2022, prior_games=0.0)
    assert no_prior.loc[1, "off_run_rate"] == pytest.approx(10 / 50)
    only_prior = blended_profiles(full[full["season"] == 2021], 2022, prior_games=4.0)
    assert only_prior.loc[1, "off_run_rate"] == pytest.approx(0.5) and only_prior.loc[1, "games"] == 0


def test_style_profile_matches_pbp(wh):
    cutoff, slate = _mid_season(wh)
    view = wh.view(cutoff)
    out = run_group(MatchupStyle(), wh, slate)
    team = int(slate["home_id"].iloc[0])
    pbp = view.table("pbp")
    live = pbp[pbp["is_scrimmage"] & ~pbp["garbage_time"] & (pbp["offense_id"] == team)]
    cur, prev = live[live["season"] == view.season], live[live["season"] == view.season - 1]
    tg = view.table("team_game")
    g_prev = ((tg["season"] == view.season - 1) & (tg["team_id"] == team)).sum()
    scale = PRIOR_GAMES / g_prev
    expect = (cur["is_rush"].sum() + scale * prev["is_rush"].sum()) / (len(cur) + scale * len(prev))
    row = out[(out["team_id"] == team)].iloc[0]
    assert row["mat_off_run_rate"] == pytest.approx(expect)
    assert row["mat_style_games"] == ((tg["season"] == view.season) & (tg["team_id"] == team)).sum()
    assert 0 < row["mat_off_expl_share"] < 1 and row["mat_off_plays"] > 30
    assert row["mat_def_havoc"] > 0 and row["mat_off_sec_per_play"] > 0


def test_edges_mirror_and_variance_is_shared(wh):
    _, slate = _mid_season(wh)
    out = run_group(MatchupStyle(), wh, slate)
    for game in slate.itertuples():
        h = out[(out["game_id"] == game.game_id) & (out["team_id"] == game.home_id)].iloc[0]
        a = out[(out["game_id"] == game.game_id) & (out["team_id"] == game.away_id)].iloc[0]
        for kind, *_ in _EDGE_KINDS:
            assert h[f"mat_off_edge_{kind}"] == pytest.approx(-a[f"mat_def_edge_{kind}"], nan_ok=True), kind
            assert a[f"mat_off_edge_{kind}"] == pytest.approx(-h[f"mat_def_edge_{kind}"], nan_ok=True), kind
        for col in ("mat_var_pace", "mat_var_sec_per_play", "mat_var_expl_share"):
            assert h[col] == pytest.approx(a[col])
    assert out["mat_off_edge_rush_epa"].notna().all()


def test_edges_follow_unit_ratings(wh):
    """A team whose rush offense is rated higher gets a larger rush edge against the same defense."""
    cutoff, slate = _mid_season(wh)
    view = wh.view(cutoff)
    ur = view.latest("unit_ratings", ["team_id", "unit"]).set_index(["team_id", "unit"])["epa_adj"]
    out = run_group(MatchupStyle(), wh, slate).merge(
        pd.concat([slate[["game_id", "home_id", "away_id"]].rename(columns={"home_id": "team_id", "away_id": "opp_id"}),
                   slate[["game_id", "away_id", "home_id"]].rename(columns={"away_id": "team_id", "home_id": "opp_id"})]),
        on=["game_id", "team_id"])
    raw = [ur.loc[(t, "off_rush")] + ur.loc[(o, "def_rush")] for t, o in zip(out["team_id"], out["opp_id"])]
    # the edge is the raw sum minus constant league means
    diff = out["mat_off_edge_rush_epa"].to_numpy() - np.array(raw)
    assert np.allclose(diff, diff[0])


def test_on_demand_matchup(wh):
    """A game in no table (an on-demand matchup) gets features from the slate alone."""
    cutoff, slate = _mid_season(wh)
    game = slate.iloc[[0]].copy()
    game["game_id"] = 999_000_001
    game[["home_id", "away_id"]] = [[1, 8]]
    for group in GROUPS:
        out = run_group(group, wh, game)
        assert len(out) == 2 and set(out["team_id"]) == {1, 8}
        assert out.drop(columns=["game_id", "team_id"]).notna().any(axis=None)


def test_history_counts_and_shrinkage(wh):
    cutoff, slate = _mid_season(wh, 2023, 3)
    view = wh.view(cutoff)
    out = run_group(MatchupHistory(), wh, slate)
    assert out["mat_hist_def_vs_run_heavy"].notna().all()
    stats = game_stats(view, [view.season - 1, view.season])
    res = game_residuals(view, stats)
    for row in out.itertuples():
        n = row.mat_hist_n_def_vs_run_heavy + row.mat_hist_n_def_vs_pass_heavy
        assert n <= (res["team_id"] == row.team_id).sum()
        # |shrunk mean| <= max |residual| * n / (n + k)
        mx = res.loc[res["team_id"] == row.team_id, "res_def"].abs().max()
        assert abs(row.mat_hist_def_vs_run_heavy) <= mx * row.mat_hist_n_def_vs_run_heavy / (
            row.mat_hist_n_def_vs_run_heavy + HISTORY_SHRINK) + 1e-12


def test_game_residuals_use_ratings_at_the_game_cutoff(wh):
    cutoff, _ = _mid_season(wh, 2023, 3)
    view = wh.view(cutoff)
    stats = game_stats(view, [view.season])
    res = game_residuals(view, stats).dropna(subset=["res_off"])
    row = res.iloc[-1]
    gcut = view.schedule().set_index("game_id").loc[row["game_id"], "cutoff_utc"]
    ur = view.table("unit_ratings")
    ur = ur[ur["cutoff_utc"] == gcut].set_index(["team_id", "unit"])
    teams = view.table("teams", seasons=view.season)
    fbs = teams.loc[teams["fbs"], "team_id"]
    rated = ur.xs("off_all", level="unit")
    lg = rated[(rated["n_plays"] > 0) & rated.index.isin(fbs)]["epa_adj"].mean()
    s = stats[(stats["game_id"] == row["game_id"]) & (stats["team_id"] == row["team_id"])].iloc[0]
    expect = s["off_epa"] / s["off_scrim"] - (ur.loc[(row["team_id"], "off_all"), "epa_adj"]
                                              + ur.loc[(row["opp_id"], "def_all"), "epa_adj"] - lg)
    assert row["res_off"] == pytest.approx(expect)


# ---- real data (skipped in CI) ---------------------------------------------------------

@pytest.mark.parametrize("group", GROUPS, ids=lambda g: g.name)
def test_no_leak_on_real_data(snapshot, group):
    assert_group_no_leak(group, snapshot, n=2)


def test_real_style_values_are_plausible(snapshot):
    games = snapshot.schedule([2025])
    cutoff = games.query("season_type == 'regular' and week == 10")["cutoff_utc"].iloc[0]
    slate = games[games["cutoff_utc"] == cutoff]
    out = run_group(MatchupStyle(), snapshot, slate)
    fbs = set(slate.loc[slate["home_fbs"], "home_id"]) | set(slate.loc[slate["away_fbs"], "away_id"])
    o = out[out["team_id"].isin(fbs)]
    assert o["mat_off_run_rate"].between(0.25, 0.95).all() and o["mat_off_run_rate"].std() > 0.03
    assert o["mat_off_plays"].between(50, 90).all()
    assert o["mat_off_sec_per_play"].between(15, 40).all()
    assert o["mat_def_havoc"].between(0.05, 0.35).all()
    assert o["mat_off_edge_rush_epa"].notna().mean() > 0.95

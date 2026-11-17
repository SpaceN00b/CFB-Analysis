# Matchup and play style features (W4)

Package `src/cfb/features/matchup/`, prefix `mat_`, level `team` (keys `game_id, team_id`;
the row describes the team against this opponent). Two groups, both `per_cutoff`:

| Group | Columns | What |
| --- | --- | --- |
| `matchup_style` | 48 | Style profile (27), unit and style edges in both directions (18), variance inputs (3) |
| `matchup_history` | 16 | How the team did against offenses or defenses of this opponent's style |

Full catalog: `python -m cfb.features.build --list`. Build time for all 660 cutoffs: about
3 min for `matchup_style` and 3.5 min for `matchup_history`.

## Definitions

All play-level values use the shared `pbp` flags (scrimmage, rush, pass, success, passing
down, garbage time). Garbage time is removed except for pace.

| Item | Definition |
| --- | --- |
| Run rate | Rushes over scrimmage plays. Sacks count as passes (pbp flags). |
| Neutral | First or second down, score within 14. |
| Down and distance buckets | 1st down; 2nd and 1-3 / 4-7 / 8+; 3rd or 4th and 1-3 / 4+; score within 14. |
| Pace | `mat_off_plays`: offensive scrimmage plays per game. `mat_pace_plays`: both teams' plays in the team's games. |
| Seconds per play | Drive elapsed time over drive plays (`drives`), drives with 0 to 900 seconds. |
| Explosive play | Rush of 12+ yards or pass of 16+ yards. Share = positive EPA on explosive plays over all positive EPA. |
| Standard and passing downs | Success rate on each (passing down per the pbp flag) and the share of plays on passing downs. |
| Havoc | `stats_game_havoc` events over plays, summed over games (play-weighted). The table is keyed by team name; names are matched to ids through the schedule. |
| Line yards, stuff rate | `adv_*` per-game rates weighted by the game's rush plays. |

**Prior-season blend.** Every profile rate is (current-season numerator + 4 x prior-season
numerator per game) over (the same for the denominator). In week 1 the profile is last
season's; the prior counts as 4 games and fades as games are played. `mat_style_games` is
the number of current-season games. 2014 week 1 has no profile (data starts in 2014).

**Unit edges.** From `view.latest("unit_ratings", ["team_id", "unit"])`, rows of the current
season with plays. Offense edge for unit u = (team off_u - league off_u) + (opponent def_u -
league def_u); league = mean over FBS teams. Defense edge = minus the same with the roles
swapped. Positive always favors the row's team. Kinds: rush and pass EPA, rush and pass
success, explosiveness (all plays), `mix_epa` (rush and pass EPA edges weighted by the
offense's run rate), `align` (run-rate lean times rush edge minus pass edge), and two
style edges from the profile, not from `unit_ratings`: havoc and line yards. No opponent
adjustment is done here: the edges improve when W3's ratings replace the placeholder, with
no code change. Until then week 1 edges are empty (placeholder has no plays).

By construction `mat_def_edge_*` of a team equals minus `mat_off_edge_*` of its opponent's
row. In a home-minus-away model the two families give identical differences, so W8 should
use one of them in that form.

**Variance inputs** (same value on both rows): mean offensive plays per game of the two
teams, mean seconds per play, mean explosive share.

**Style history.** For every past game against an FBS opponent in the current and previous
season, a residual in EPA per play (garbage time removed): actual minus expected, where
expected = team unit rating + opponent unit rating - league average, all from `unit_ratings`
(`off_all`, `def_all`) at that game's own cutoff. Positive favors the team. Opponents are
placed in thirds of that season's FBS teams: offenses by neutral run rate and by seconds per
play, defenses by (rush EPA allowed - league) - (pass EPA allowed - league) and by havoc
created. Each value is sum / (n + 150) over the games against that kind of opponent.
`mat_hist_n_*` hold n. The four `*_vs_opp_*` columns pick the class of this game's opponent.
A team without rated past games against FBS teams gets empty values.

## Shrinkage chosen on seasons through 2025

`python scripts/matchup_tune.py` (reads 2014 to 2025 only).

Prior-season weight: for each team-season 2015 to 2025 and each point after one or more
games, the blended rate predicts the rate over the rest of the season. Mean squared error
over 10 metrics, each scaled by its spread across teams:

| Prior games | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scaled MSE | 0.744 | 0.513 | 0.464 | 0.445 | **0.438** | 0.439 | 0.446 | 0.464 |

Pure tendencies prefer less prior (run rates and seconds per play are best at 2 to 3),
efficiency-like rates more (6 to 8). One value, 4, is used for all.

History shrinkage: split-half reliability of the class means (odd against even games per
team and class, two-season windows ending 2015 to 2025, about 4,100 groups per family).
Residuals of one team share a persistent rating error, so the covariance of a placebo
grouping (opponent id mod 3) is subtracted as the bias.

| Family | Side | True variance (EPA/play)^2 | SE | Implied shrink (games) |
| --- | --- | --- | --- | --- |
| Run rate | defense | -0.00026 | 0.00042 | none (no signal) |
| Tempo | defense | -0.00071 | 0.00043 | none |
| Front (run/pass strength) | offense | 0.00010 | 0.00042 | 552 |
| Havoc | offense | 0.00039 | 0.00041 | 144 |

No family has a team-by-style effect distinguishable from zero. Per-game noise variance is
0.056, so the upper end of the estimates (about 0.0012) still implies a shrink of 47 games
or more. 150 is used; with 8 games against a class the value is about 5% of the raw mean.

## Check against the margin after Elo and home field

`python scripts/matchup_check.py` (2014 to 2025, 9,085 completed FBS vs FBS games with
pregame Elo; base model 3.22 points home field, 4.24 points per 100 Elo, residual SD 16.7).
Single-feature slopes of the residual on the home-minus-away value, in points per standard
deviation, using the placeholder unit ratings. Selected rows (the script prints all 64):

| Column | Pts per SD | t |
| --- | --- | --- |
| `mat_off_edge_rush_success` | 1.59 | 8.9 |
| `mat_off_edge_pass_success` | 1.10 | 6.1 |
| `mat_off_edge_mix_epa` | 1.08 | 6.0 |
| `mat_off_edge_line_yards` | 1.02 | 5.8 |
| `mat_off_edge_rush_epa` | 1.00 | 5.6 |
| `mat_off_sd_success` | 0.86 | 4.9 |
| `mat_off_pd_rate` | -0.85 | -4.8 |
| `mat_off_edge_explosive` | -0.51 | -2.8 |
| `mat_off_edge_havoc` | 0.36 | 2.0 |
| `mat_off_edge_align` | 0.00 | 0.0 |
| `mat_off_run_rate_neutral` | 0.04 | 0.3 |
| `mat_hist_def_vs_opp_run_class` | -0.22 | -1.3 |
| `mat_hist_def_vs_opp_tempo_class` | -0.33 | -1.9 |
| `mat_hist_off_vs_opp_front_class` | 0.18 | 1.0 |
| `mat_hist_off_vs_opp_havoc_class` | 0.26 | 1.5 |

Variance inputs against the absolute residual: `mat_var_pace` +0.30 points per SD (t=2.8;
SD 3.6 plays, so about +0.8 points per 10 plays, near the plan's measured +1.0),
`mat_var_sec_per_play` -0.43 (t=-4.0, faster offenses, larger misses), `mat_var_expl_share`
-0.04 (t=-0.4).

Reading: the edges carry information beyond Elo (they overlap with W3's strength ratings;
the ablation is W8's). Pure style (run rates, pace) does not predict the margin by itself,
as expected. Style history shows nothing beyond noise: |t| at most 2.1 over 10 columns, in
line with the reliability estimate. W8 can likely drop `matchup_history`.

Coverage (share of FBS vs FBS games with both rows filled): profile columns 94% in 2014
(week 1 has no prior season) and 99 to 100% from 2015; havoc 68% in 2014, 99% or more from
2015 (the blend uses the games that have havoc rows); unit edges 86 to 95% (empty in week 1
with the placeholder); history 83% in 2014, 99 to 100% from 2015.

## Open issues

- Re-run `scripts/matchup_check.py` once W3's ratings are on main; edge values change.
- Seconds per play includes clock stoppages and end-of-half drives; it is a drive-level
  proxy for tempo, not snap-to-snap time.
- History residuals need a rated opponent at the game's cutoff; with the placeholder,
  week 1 games of each season have none.

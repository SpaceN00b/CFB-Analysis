# Coaching features (W5)

Package `src/cfb/features/coaching/`, prefix `coach_`, level `team` (keys `game_id, team_id`),
mode `per_cutoff`. Two groups, so W8 can drop one without the other.

| Group | Columns |
| --- | --- |
| `coach_tenure` | `coach_tenure_seasons`, `coach_first_year`, `coach_interim`, `coach_games_career`, `coach_games_school`, `coach_ats_career`, `coach_ats_career_n`, `coach_ats_school`, `coach_ats_school_n` |
| `coach_style` | `coach_fourth_go_oe`, `coach_fourth_n`, `coach_run_oe`, `coach_run_n`, `coach_flex_slope`, `coach_flex_games`, `coach_flex_shift` |

All columns are float64; flags are 1.0/0.0. A team without a known coach (FCS teams) gets
empty values. A known coach without history gets 0 for counts and rates over expected,
0.5 for ATS, and the pooled slope for flexibility. "Career" means since 2014: earlier
games are not in the data.

Reliability figures and constants below come from `python scripts/coaching_check.py`
(seasons 2014 to 2025 through one view on 2026-07-01, no 2026 data).

## Coach-to-game map (`mapping.py`)

Sources: `coach_tenures` (as the view shows it: a tenure once hired, future end years
blanked, records hidden) and `view.prior_seasons("coaches")` (games per coach per finished
team-season). CFBD had not attributed 2026 games to coaches on 2026-11-17, so games are
mapped by tenure dates, not by attribution.

- A tenure that starts on or after 25 August of its season (effective start, else hire
  date) is an in-season start and takes over from that moment. Otherwise the season-start
  tenure with the latest start year wins. An undated tenure that starts in the same season
  as an older candidate cannot be placed and does not take over in the live season.
- Finished seasons follow CFBD's per-season game counts: one coach takes all games; with
  several, they are ordered by when the date rule first assigns them and take their counts
  in kickoff order.
- `coach_interim`: tenure flagged interim (set almost only from 2022), or the coach took
  over during the season (before 2022 this is the only signal). A new permanent coach who
  coaches only the bowl is flagged too.
- `coach_tenure_seasons`: 1 in the first full season at the school, 0 while finishing a
  season as an in-season replacement. Back-to-back tenure rows of the same coach at the
  same school are joined (interim then permanent). `coach_first_year` = tenure 0 or 1.

Checks:

| Check | Result |
| --- | --- |
| FBS team-games with a coach, 2014-2025 | 100% of 19,458 |
| Mapped games equal CFBD's count per coach-team-season | 98.6% (99.5% within one game), 1,693 |
| Coach at each season's last cutoff (point in time) equals the final map | 98.2% to 99.9% by season |
| First-year effect on margin after pregame Elo and home field | -1.52 points (se 0.31, 9,085 games); plan doc -1.6 |

Known limits: a coach who was interim for weeks before being hired permanently shows up
only from the hire date while the season is live (2017 team 290: 5 games). Undated
mid-season takeovers (7 games in 2017) are fixed once the season's counts exist.

## Record against the spread (`coach_ats_*`)

Closing spreads of past games only: `lines` joined to games visible in `team_game`, so the
slate's lines are never read. Provider priority consensus, Bovada, DraftKings, ESPN Bet.
Pushes are dropped. Rate = 0.5 + sum(cover - 0.5) / (n + 200), career and at the school.

| Figure | Value |
| --- | --- |
| Team-games with a line | 18,510 |
| Coaches with >= 50 games | 160 |
| Split-half reliability (>= 25 games per half, 143 coaches) | 0.03 |
| True between-coach variance (method of moments) | -0.0002 (none detectable) |
| Correlation of career rate through S-1 with season S | -0.01 |

ATS skill is not distinguishable from noise. The 200 pseudo-games keep the column within
about ±0.05 of 0.5. Recommendation for W8: drop `coach_ats_*` unless the ablation says
otherwise.

## Fourth-down aggressiveness (`coach_fourth_go_oe`)

Fourth-down plays outside garbage time with a rush, pass, punt or field goal (fakes count
as going). Expected go rate per cell of distance (6 buckets) x yards to goal (7) x score
(9) x time (Q1-Q3, Q4, last 2 min of Q2, last 5 min of Q4, OT), pulled toward the
distance x field-position cell with 20 pseudo-plays, fitted on visible plays of the play's
own season and the season before. Coach value = season-weighted mean of (go - expected),
half-life 1 season, shrunk toward 0 with 75 pseudo-decisions.

| Figure | Value |
| --- | --- |
| Decisions | 136,331 (mean expected go rate 0.20) |
| Coaches with >= 200 decisions | 231 |
| Split-half reliability (>= 100 per half, 215 coaches) | 0.86 |
| True variance / per-play noise | 0.00135 / 0.092 -> prior about 68 decisions |
| Prediction of season S from history through S-1 (r) | 0.53 at half-life 1; 0.51 without decay |

## Run/pass mix above expected (`coach_run_oe`)

Scrimmage plays on first and second down, score within 14, outside garbage time (the
shared `pbp` flags; sacks count as passes). Expected run rate per cell of down x distance
(5) x yards to goal (6) x score (5) x period (Q1-Q4, last 5 min of Q4), same fitting rule
as above. Coach value = season-weighted mean of (rush - expected), half-life 1 season,
shrunk toward 0 with 600 pseudo-plays.

| Figure | Value |
| --- | --- |
| Plays | 763,111 (mean expected run rate 0.55) |
| Coaches with >= 2,000 plays | 160 |
| Split-half reliability (>= 1,000 per half, 139 coaches) | 0.97 |
| True variance / per-play noise | 0.0056 / 0.231 -> prior about 41 plays |
| Prediction of season S from history through S-1 (r) | 0.74 at half-life 1; 0.70 without decay |

The prior is set above the method-of-moments value because the predictive check is flat
between 150 and 1,200 and slightly best at 600; season-to-season drift (coordinator
changes) is what the half-life handles.

## Offensive flexibility (`coach_flex_*`)

For each past game: the team's neutral run rate over expected (as above) and the
opponent's defensive gap at that game's cutoff: rush EPA allowed minus pass EPA allowed
from `unit_ratings`, minus the league mean of that difference at the same cutoff (teams with
at least 60 plays in both units). Positive gap = weaker against the run than the pass.
Within each coach-season both are centred, so the slope measures movement away from the
coach's own tendency that season. Games are weighted 1 / (0.25 / n + 0.002). The coach
slope is shrunk toward the pooled slope of all coaches by its precision, with a
between-coach slope variance of 0.0006. `coach_flex_shift` = slope x this opponent's
current gap: the run-rate shift the coach's history implies for this game.

| Figure | Value |
| --- | --- |
| Team-games with neutral plays / with a rated opponent | 19,353 / 15,007 |
| Pooled within coach-season slope | +0.034 run share per EPA of gap (sd of gap 0.17) |
| Slope variance observed / sampling / true | 0.0087 / 0.0081 / 0.0006 |
| Split-half reliability of the raw slope (>= 30 rated games per half, 85 coaches) | 0.15 |
| Coaches with >= 60 rated games | 98 |

Coaches as a group do lean toward the run when the opponent is weaker against it, but
coaches barely differ from each other in how much: with the shrinkage, `coach_flex_slope`
stays close to the pooled value and `coach_flex_shift` is mostly the opponent's gap
times a common slope. Measured on the W1 placeholder ratings (unadjusted season to date);
rerun `scripts/coaching_check.py` after W3's adjusted ratings merge, and update
`FLEX_SLOPE_VAR` and `FLEX_GAME_VAR` in `metrics.py` if the figures move. The code needs no
other change.

## For W8: which columns are noisy

| Metric | Split-half reliability | Keep? |
| --- | --- | --- |
| `coach_first_year`, `coach_interim`, `coach_tenure_seasons` | not applicable (facts) | yes; first year is worth -1.5 points |
| `coach_ats_*` | 0.03 | drop unless the ablation disagrees |
| `coach_fourth_go_oe` | 0.86 | yes |
| `coach_run_oe` | 0.97 | yes |
| `coach_flex_slope`, `coach_flex_shift` | 0.15 | test in the ablation; mostly a common slope |

## Leak tests and run time

- Synthetic: both groups pass `assert_group_no_leak` with coach tenures and season counts
  added in `tests/test_coaching.py` (the synthetic store lacks them; without
  `coach_tenures` the groups return empty values).
- Snapshot: `python -m cfb.features.build --leak-test --groups coach_tenure,coach_style`
  passes, and `tests/test_coaching.py::test_snapshot_leak` passes at four chosen cutoffs
  (2016 week 9, 2022 week 11, 2026 week 11, a 2025 bowl).
- Full build at 660 cutoffs: see the PR.

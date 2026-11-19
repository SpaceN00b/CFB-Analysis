# Injury-adjusted test model (test bench, 2026-11-19)

Question (Josiah): rebuild the test model with the 68-team player maps / playbooks data, the PR 26
impact scores and the CFBDepth files, so it accounts for injuries and their effect on the team.

**Result:** model_v2 plus an injury term built from availability listings improves 2026 log loss
from 0.5453 to 0.5369 (-0.0084, SE 0.0028, 632 FBS games through week 11) and P4/Notre Dame games
from 0.5326 to 0.5192 (-0.0134, SE 0.0046). Nearly all of the gain is in games where a starting QB
is listed out (-0.041, SE 0.014, 106 games). Using only true pregame sources (conference reports and
the CFBDepth official-report entries) gives the same gain (-0.0085, SE 0.0022), so it does not come
from the hindsight-built injury windows. Before 2025 there are no listings, so 2018-2024 cannot test
it; 2025 conference games with a report listing move -0.0037 (SE 0.0040, n 276). The closing line
is still better (2026: 0.5155). Nothing in a model config changes.

Code: `src/cfb/lab/availability.py` (roles, listings, p_out, team columns), `scripts/injury_select.py`
(test), `scripts/injury_week.py` (this week's games), `scripts/injury_checks_prep.py` (per-player
check inputs). Tests: `tests/test_availability.py`.

```bash
python -m cfb.core.build
python scripts/impact_base.py               # model_v2 season-forward margins
python scripts/injury_select.py --build     # roles 2014-2026 (~1.5 min), then the test (~6 min)
python scripts/injury_week.py --week 12     # -> /mnt/project-files/reports/injury-test/week12_2026.{md,csv}
```

## Data

| Source | What | Coverage | Point in time |
| --- | --- | --- | --- |
| CFBDepth player-updates (D:\claude, 2026-11-18) | status, Last Update, narrative, official-report entries | all FBS but Nebraska, 1,449 rows | current status; report entries are per game |
| Per-player checks (Sonnet 5.5, 12 batches) | injury date, first listed out, games missed, expected return | all 1,449 rows | built after the fact from the narrative + box scores |
| Conference reports (HDI archive) | Initial ... Game Day status | 2025-2026 conference games, 62 teams, no Notre Dame | pregame |
| Box scores + /plays/stats | presence, dropbacks, carries, targets, tackles, EPA | 2014-2026 | per game |
| CFBDepth position exports (D:\claude) | snaps by alignment, games, PFF-style ratings | all FBS, season to 2026-11-18 | season snapshot |

Project rule (larger dataset first): the CFBDepth list wins where both cover a player and game; the
conference reports fill the rest. 97% of the 1,800 overlapping rows agree.

Per-player checks (`/mnt/project-files/injury-reports/checks_2026-11-19/`, combined file
`injury_checks_2026-11-19.csv`): injury date found for 559 players (147 preseason), first listed date
1,074, missed-game dates 1,223, expected return 614 (367 out for the season, 199 game-time). 29 web
searches in total (allowed only for P4/Notre Dame contributors whose text had no date); most found
only 2025 news. Confidence: high 376, medium 659, low 414.

## Method

Per team-game, from the team's earlier games of the same season (pregame):

- **Roles:** share of non-garbage dropbacks (QB), carries (RB), targets (WR/TE) and box-score tackles
  (DL/LB/DB) in the games the player appeared in, at least 2.
- **QB value over replacement** = (starter EPA/dropback - replacement EPA/dropback) x team dropbacks per
  game. Starter = most season dropbacks; replacement = next QB by dropbacks who is not listed out
  (else a league-average backup). EPA rates shrunk 120 dropbacks toward league means of starters /
  backups from earlier seasons. 2026 mean 4.2 points, top ~12.
- **p_out:** P(absent | listed status) measured on the other season's conference reports (Out 0.998,
  Questionable 0.57, Game Time Decision 0.54-0.60); not listed = 0 in the recommended version.
- **Injury points per team** = -(QB value x p_out) + share x p_out x PR 26 points per 100% share
  (RB -1.94, WR -3.24, TE -3.77). Game term = home minus away, added to the model_v2 margin and passed
  through model_v2's calibration slope.

## Results

Log loss (model_v2 -> test), paired bootstrap by week. `fixed_listed` is the recommended version.

| Version | 2026 FBS (n 632) | 2026 P4/ND (n 353) | 2026 QB1 listed (n 106) | 2025 conf-report games (n 276) | 2020-2025 |
| --- | --- | --- | --- | --- | --- |
| fixed_listed: listings only, fixed points | **-0.0084 (0.0028)** | **-0.0134 (0.0046)** | -0.0411 (0.0139) | -0.0037 (0.0040) | -0.0002 (0.0003) |
| fixed: listings, else box-score streak | -0.0087 (0.0020) | -0.0119 (0.0041) | -0.0469 (0.0096) | -0.0033 (0.0041) | +0.0015 (0.0011), 2/6 |
| fitted on 2018-2025 (QB only) | -0.0032 (0.0009) | -0.0042 (0.0016) | -0.0186 (0.0035) | -0.0016 (0.0009) | +0.0002 (0.0003) |
| fitted, QB + skill + defense + 2+ key out | -0.0030 (0.0012) | -0.0050 (0.0018) | -0.0151 (0.0041) | -0.0005 (0.0021) | +0.0002 (0.0004) |
| CFBDepth Impact column, fit on earlier 2026 weeks | -0.0011 (0.0042) | -0.0040 (0.0067) | | | |
| fixed_listed + OL / defensive starters out (CFBDepth snaps) | -0.0053 (0.0044) | -0.0101 (0.0075) | | | |

Accuracy 2026: 71.5% -> 71.8%; margin RMSE 15.84 -> 15.68.

**Source check (fixed points, 2026 FBS / P4):** streak only -0.0036 / -0.0058; pregame listings only
(conference reports + CFBDepth report entries) -0.0085 / -0.0138; + CFBDepth current status
-0.0065 / -0.0136; + check windows -0.0084 / -0.0134. The hindsight-built windows add nothing.

**Why the fitted versions are weaker:** before 2025 the only pregame signal is "missed the last game"
(55-75% chance of missing the next), so the fitted QB slope shrinks to -0.3 points per point of value.
In 2026, with listings, the in-sample slope is -1.3, close to the fixed 1.0.

**Team effects:**
- 2+ key offensive players out (2018-2025, realized absences): -0.49 points beyond the single-player
  terms; fitted pregame it adds nothing.
- Offensive line and defensive starters out (starter share = snaps per game / the team's top-N at the
  unit, from the CFBDepth exports; walk-forward fit on earlier 2026 weeks): in-sample -2.2 points per
  OL starter out and -0.7 per defensive starter, but out of sample it is worse than fixed_listed
  (+0.0031, SE 0.0040). Not added. The exports are season totals at the download, not point in time.
- Defensive tackle share (box scores) and the CFBDepth Impact column: no gain.

## Limits

- One test season with listings (2026) plus a weak 2025 check; no listings before 2025.
- Status probabilities come from conference reports (conference games); non-conference listings
  use the same rates.
- The CFBDepth list is a current-status snapshot; earlier 2026 games use its report entries, Last
  Update windows and the per-player check windows. Nebraska is not on the list (conference reports
  fill its conference games).
- The test fits model_v2 on 2015-2025 only (as in PR 26); production model_v2 refits at each 2026
  cutoff.

## Start-of-season depth chart

Josiah will add one later. Planned use: mark preseason starters so a starter hurt in camp (no 2026
snaps, so no role here) is known and his replacement's value can be set.

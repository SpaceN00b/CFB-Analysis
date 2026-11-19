# Player maps (test bench)

`cfb.lab.player_maps` is an experiment: no feature group, model or weekly run reads it. It uses the
detailed stat-crew play text that `cfb.core.playtext` parses (every game from 2025 week 9 on), so
it shows one recent season for one conference. The default is the 2026 SEC.

```bash
python -m cfb.core.build && python -m cfb.core.playtext   # core tables and play_detail / play_players
python -m cfb.lab.player_maps                              # about 80 s; --season, --conference, --fit-seasons, --out
```

Output in `data/lab/` (or `--out`): `player_maps_<season>_<conf>.json` (everything the viewer
shows), `.html` (the viewer, `src/cfb/lab/player_maps_viewer.html` with the JSON embedded),
`_defender_profiles.parquet` (every FBS defender profile with role probabilities) and
`_role_coef.csv` (role model weights).

Directions are the offense's: left, middle, right as the stat crew writes them. A defender who
makes plays on the offense's left lines up on the defense's right. The text has no hash marks or
formation strength, so field and boundary cannot be told apart. Stat crews chart more plays to
the right (2026: about 45% of targets with a side go left, 48% of located defender plays), so side
leans are measured against the league share, not 50%.

## Defense: approximate location

Each defender credit on a scrimmage play is placed on a 3-lane by 5-depth grid:

| credit | side | depth |
| --- | --- | --- |
| tackle on a run | run direction | gain (where the runner went down) |
| tackle on a catch | pass direction | total gain |
| pass breakup | pass direction | air yards |
| interception | none in the text | air yards |
| sack, hurry | none | backfield |

Per defender-season (at least 15 credits): run-tackle depth, share of tackles for loss and of
runs stopped 8+ yards downfield, share of catch tackles and their depth, pass-rush and coverage
rates, average air yards on coverage plays, middle share on runs and passes, side lean, shared
tackles and overall median depth. "Where he makes plays" is the median depth band and lane
against the FBS average. Where a player finishes plays is close to where linemen and linebackers
align and shallower than where safeties align.

A softmax regression on those 14 features predicts a role (interior DL, edge, off-ball LB,
cornerback, safety). Labels: CFBD roster position where it is specific (DT, NT, DE, EDGE, CB, S),
else Sportradar's position for the same team, jersey and last name; generic LB counts as off-ball
LB; generic DL and DB get only the 3-group label. It is trained on every non-SEC FBS defender-season
(2025 week 9 on and 2026) and tested on SEC 2026 defenders.

Result (built 2026-11-18, CFBD plays through 2026 week 11):

| test set | correct | most common role |
| --- | ---: | ---: |
| 5 roles, 169 SEC defenders with a specific listed role | 74.6% | 38.5% |
| 3 groups (DL / LB / DB), 271 SEC defenders | 90.4% | 39.9% |

Corners (19 of 20) and off-ball linebackers (62 of 65) are recognised; edge players are the weak
spot (11 of 28 correct, 13 called off-ball LB), since many stand-up ends and 3-4 outside
linebackers make plays where linebackers do. 63 of 271 SEC defenders get a role other than the
listed one; the most common are DL-listed players who play like off-ball linebackers (15) and
safeties who play like corners (6) or box linebackers (6). "Approximate alignment" in the viewer is
the model's role plus a side lean where there is a clear one.

## Offense: where players go and what is called for them

Targets by direction and air yards (behind the line, 1-9, 10-19, 20+), runs by direction and gain,
passers by direction and air yards. Play-type mix (run left/middle/right, screen or behind the
line, short, intermediate, deep) and situations (shotgun or under center, no huddle, down, red
zone) are shares of the player's own touches and targets. Style is a rule on those shares (for
example screens/swing at 35%+ behind the line, vertical at 25%+ deep or 14+ average depth). Players
with 15+ targets, 30+ rushes or 60+ passes. Sportradar's screen, play-action and RPO flags are
false on every 2026 play, so play action and RPOs cannot be identified.

## Running backs: where they struggle

RB and FB carries, garbage time left out, CFBD success. For each direction (and direction by
formation) the expected success is the FBS RB average in that direction plus each opponent's
adjustment in that direction (its success allowed that way, shrunk toward the FBS rate with 25
carries). A trouble spot is a split with at least 8 carries and z of -1.28 or lower (one-sided
10%). By game, the expectation uses the back's own success that way (shrunk with 15 carries) plus
the opponent adjustment; "struggled" = at least 4 carries and z of -1.28 or lower. With three
directions per game, a few of those flags are chance.

2026 SEC: 39 backs with 30+ carries, 5 with a trouble spot by direction (for example Jamal
Roberts, Missouri, left: 26% success on 42 carries against 41% expected), 17 of 335 back-games
flagged.

## Receivers and QBs: drops or the quarterback

No source marks drops: neither the play text nor Sportradar (no dropped, catchable or
incompletion-type field). The proxy is a clean incompletion: a pass to a named receiver that falls
incomplete with no breakup, interception or hurry credited. That covers drops, off-target throws
and missed reads. Hurried throws are left out because stat crews credit a hurry almost only when
the pass falls incomplete (2026: 3,886 of 4,230 hurried passes).

Model: logistic regression on 54,392 unhurried targets in games with an FBS team (2025 week 9 on and 2026, garbage time
left out) with a cell for each air-yard bin and direction, a 3rd/4th-down term, and ridge effects
for passer (logit SD 0.4), receiver (0.4) and defense (0.25), fit by backfitting. A receiver's
effect is relative to the other receivers his QBs threw to; a QB's is relative to the other QBs
his receivers caught from (backups, 2025 games, transfers). A flag needs z of 1.96 or more.
Play-type cells compare a receiver's clean incompletions on one depth and direction with his
teammates' on the same throw (shrunk toward the FBS rate with 10 throws); a flag needs a one-sided
binomial p below 0.01, 6+ targets and 20+ points above teammates.

2026 SEC readings: Texas A&M is the only team flagged as a QB issue (Marcel Reed 23.8% clean
incompletions on 235 unhurried targets vs 19.1% for an average QB with the same receivers and
throws, z 2.13). No receiver is flagged overall. Mississippi State has two play-type spots (Zion
Ragins short left, Sam West short middle). The other 14 teams show no clear issue.

## Depth chart labels (2026-11-19)

With the team depth charts imported (`cfb.depth_charts`, PR #32: all P4 teams and Notre Dame,
downloaded 2026-11-19), the lab takes each charted player's highest listing as his label:

- Defense roles: DT, NT and the ends of a three-man front (the chart also lists a stand-up edge
  such as JACK, BUCK or OLB) are interior; four-man-front ends and stand-up rushers are edges;
  NB, STAR, SPUR, MONEY and similar are a sixth role, nickel; LCB/RCB/FCB/BCB are corners; SS, FS
  and other DBs safeties; other LB labels off-ball linebackers. L and R on the label are kept as
  the chart side.
- Offense alignment: WR-X, WR-Z, slot (WR-SL, WR-H, WR-F, WR-A, WR-Y), TE, RB, QB. The playbooks
  lab (`cfb.lab.playbooks`) uses X, Z and Slot in place of WR for charted receivers.

The chart is one snapshot, so it labels 2026 profiles only; 2025 profiles and uncharted (Group of
Five) teams keep the CFBD / Sportradar labels.

Role model with chart labels, trained on 800 charted non-SEC defenders, tested on 257 charted
SEC defenders (`python -m cfb.lab.player_maps`, 2026 through week 11):

| | Result |
|---|---|
| 6 roles (adds nickel) | 74.7% correct (most common role 18.7%) |
| By role | corner 33/39, edge 17/34, interior 56/71, off-ball LB 46/48, nickel 8/23, safety 32/42 |
| Same players, nickels left out | chart-trained 78.6% vs CFBD-label model 66.7% (n 234) |
| CFBD position agrees with chart role | 81% of 158 SEC defenders with a specific CFBD position; chart edges listed LB by CFBD 10 of 31, chart nickels listed S 9 of 12 |
| Chart side in the play text | corners: chart-left players have 44.7% of their lateral plays to the offense's left, chart-right 52.1% (+7.4 pts, z 3.1, n 76/77); edges +7.6 pts (z 1.9, n 41/30); interior +3.1 (z 0.4, n 12/12) |

Chart L and R read as the defense's left and right. The lean is small because corners and ends
flip with the formation and the text has no hash or strength.

Targets by chart alignment, every charted team, 2026: X receivers 43% left / 38% right of
the offense, Z 31% / 49% (league 46% of lateral targets to the left); slot aDOT 8.8 with 22%
behind the line; X and Z aDOT 11.5 and 11.4; TE 55% at 1-9 air yards. Clean incompletions by
alignment are within 2 points of the model's expectation for every group, with 2 X receivers,
no Z or slot receiver and 1 RB flagged. SEC team readings are unchanged (Texas A&M is the only
QB flag).

## All 68 charted teams and injury status (2026-11-19)

`python -m cfb.lab.player_maps --conference P4` scopes the lab to ACC, Big 12, Big Ten, SEC and
Notre Dame (68 teams). The role model is tested on each group with that group held out (trained
on the other charted teams), and the results are pooled. 2026 through week 11:

| | Result |
|---|---|
| 6 roles | 75.0% of 1,057 held-out charted defenders (most common role 20.3%) |
| By held-out group | ACC 76.0% (275), Big 12 76.7% (253), Big Ten 72.9% (258), Notre Dame 71.4% (14), SEC 74.7% (257) |
| Same players, nickels left out | chart-trained 79.1% vs CFBD-label model 69.1% (n 967) |
| CFBD position agrees with chart role | 82% of 663 defenders with a specific CFBD position |
| 3 groups (DL / LB / DB) | 90.7% of 1,103 (most common 40%) |

Output: 1,103 defenders, 647 offense players, 167 RBs. QB-issue readings (clean incompletions
above expectation for the passer): Maryland, Ohio State, Syracuse, Texas A&M, Wisconsin; one
receiver flag, Nick Marsh (Indiana).

Injury status comes from the local CFBDepth player-updates list
(`data/raw/injuries/cfbdepth-player-updates.csv`, copy of the 2026-11-18 file; all FBS teams
except Nebraska), matched on team and normalized name for the latest season only. Rows gain
`injury_status`, `injury_date` and `injury_note`; 107 defenders and 96 offense players carry one.
The viewer shows a red tag for Out and amber for other statuses. This is display only; the role
model does not use it. The SEC-only build (`--conference SEC`) still runs with one held-out fold
(74.7%, n 257).

## Model features from the same text

Three feature groups, registered so `cfb.features.build` and the weekly run compute them, but in no
model (not in `MODEL_V2_GROUPS`). All read visible plays of the cutoff season and the one before,
so they are empty before 2025 week 9. Leak-tested at four cutoffs (`tests/test_playtext_features.py`).

| Group | Columns | What it measures |
|---|---|---|
| `str_pass_ci` (`cfb/features/strength/playtext.py`) | `str_pci_qb`, `str_pci_rec`, `str_pci_def`, `str_pci_n` | The clean-incompletion model above, fit on every FBS team: the last game's starting passer's effect, the target-weighted receiver effect, the defense's effect (positive = forces more), log of team targets |
| `matchup_run_dir` (`cfb/features/matchup/playtext.py`) | `mat_rdir_fit`, `mat_rdir_exp`, `mat_rdir_n` | Run direction (left, middle, right): the offense's tilt times this defense's success allowed by direction (`fit`), expected success over the FBS rate (`exp`), log of the smaller sample |
| `matchup_pass_zone` (same file) | `mat_pzone_fit`, `mat_pzone_exp`, `mat_pzone_n` | The same for passes, by depth (behind the line, 1-9, 10-19, 20+ air yards) and direction |

Shares shrink toward the FBS share with 30 plays and success rates toward the FBS rate with 40.
Garbage time is left out of the matchup groups; hurried throws out of `str_pass_ci`.

### Do they add to model_v2?

`python scripts/playtext_select.py [--lam 0.03]`. The 2018-2025 season-forward selection cannot
use them, so the test is a second stage on model_v2's own predictions (2025 from the pre-season
fit, 2026 refit at every cutoff): logistic regression with offset logit(p_v2), no intercept,
standardized home-minus-away differences, ridge per observation. Before each 2026 week it refits on
every earlier baseline-set game from 2025 week 9 on (321 from 2025), then scores the 630 baseline-set
2026 games (FBS vs FBS, spread, both Elos). Bootstrap SE over 12 weeks.

| Columns | lam 0.03 | lam 0.3 |
|---|---|---|
| model_v2 alone (log loss) | 0.5435 | 0.5435 |
| `str_pass_ci` (4) | +0.0016 (SE 0.0039) | -0.0003 (SE 0.0014) |
| `str_pci_qb` | -0.0003 (SE 0.0022) | -0.0004 (SE 0.0009) |
| `str_pci_qb`, `str_pci_rec` | +0.0002 (SE 0.0033) | -0.0006 (SE 0.0011) |
| `matchup_run_dir` (3) | +0.0028 (SE 0.0016) | +0.0010 (SE 0.0007) |
| `matchup_pass_zone` (3) | +0.0010 (SE 0.0012) | +0.0002 (SE 0.0005) |
| `mat_pzone_fit` | -0.0003 (SE 0.0010) | -0.0003 (SE 0.0004) |
| all 10 | +0.0060 (SE 0.0050) | +0.0010 (SE 0.0021) |

No version is more than 0.6 SE better than model_v2; the run-direction group is worse at both
penalties. In-sample, one column at a time over all 951 games, the largest effects are
`str_pci_qb` (-0.11 log-odds per SD, z -1.4, the expected sign: a passer with more clean
incompletions loses more) and `mat_pzone_fit` (+0.11, z +1.4). Verdict: not added to model_v2.
With about one season of detailed text this test can detect only large effects (an SE of about
0.001 in log loss); rerun it after the 2026 season, when 2025 week 9 to 2026 gives about twice the
training games.

## Limits

- Only players credited on a play appear; there are no snap counts or participation lists.
- One season of detailed text per player at most (2025 week 9 on); 2014 to 2025 week 8 text has no
  direction, depth or tacklers.
- Roster ids come from jersey and name matching (99.6% of named players in detailed text).
- The SEC was chosen as the test bench; other conferences run with `--conference`.

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

## Limits

- Only players credited on a play appear; there are no snap counts or participation lists.
- One season of detailed text per player at most (2025 week 9 on); 2014 to 2025 week 8 text has no
  direction, depth or tacklers.
- Roster ids come from jersey and name matching (99.6% of named players in detailed text).
- The SEC was chosen as the test bench; other conferences run with `--conference`.

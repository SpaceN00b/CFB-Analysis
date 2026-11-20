# Depth-chart player impact scores (test bench, 2026-11-19)

Question (Josiah): rebuild PR 26 with the updated depth charts and positions. Players with a real
role who are listed out go back on the chart where they fit best. Test on 2025, treating a player
who did not play in a regular-season game as out injured, with players in positions like their
2026 ones.

**Result:** pricing an absence as the starter's value over the player who fills his depth chart
slot beats PR 26's usage-share pricing for RB / WR / TE. On top of PR 34's QB term it improves log
loss by 0.0021 (SE 0.0007) on 2025 and 0.0002 (SE 0.0010) on 2026 listings; pooled 0.0012 (SE
0.0006), P4/Notre Dame 0.0019 (SE 0.0008). The share pricing hurt 2025 (+0.0022 vs model_v2, SE 0.0008);
the depth-chart pricing does not (+0.0002, SE 0.0003). The skill-position term is still small against
model_v2 (pooled -0.0005, SE 0.0004); QBs carry the gain, as in PR 34.

The literal version of the ask, an out starter keeps his slot for as long as he is out, is better on
2025 and worse on 2026. In 2026 it overprices QBs whose backups have played for weeks (Miami (OH),
Buffalo, Oregon), even with a discount for games already in the ratings. The version that keeps a
starter in his slot until his backup has more touches this season (`dc_total`) is the best pooled:
-0.0040 (SE 0.0024) vs model_v2 over 1,394 games and -0.0015 (SE 0.0010) vs PR 34; P4/Notre Dame -0.0097
(SE 0.0032) and -0.0020 (SE 0.0010).

The chart positions themselves change nothing measurable. They move only 11 (2025) and 16 (2026)
players to another group, and the version with CFBD roster positions scores the same within 0.0003.
Players who missed a whole season (same team, last-season starter) add nothing measurable either.
Nothing in a model config changes.

Code: `src/cfb/lab/depth_impact.py`, `scripts/depth_impact_select.py` (tests, backtest charts, this
week's chart and impact scores), `scripts/depth_impact_week.py` (this week's picks),
`scripts/depth_impact_slots.py` (cost by slot). Tests: `tests/test_depth_impact.py`.

```bash
python -m cfb.core.build
python scripts/impact_base.py               # model_v2 season-forward margins
python scripts/injury_select.py --build     # player games and roles (PR 34)
python scripts/depth_impact_select.py       # ~2 min -> data/exploration/depth_impact/
python scripts/depth_impact_slots.py
python scripts/depth_impact_week.py --week 12   # -> /mnt/project-files/reports/depth-impact/week12_2026.{md,csv}
# the local app's Players page (player toggles): check the per-team recompute, pack, upload to R2
python scripts/depth_impact_week.py --week 12 --out data/exploration/depth_impact_week
python scripts/availability_export.py --week 12 --publish   # -> extras/availability/2026_regular_12.zip
```

Needs `data/tables/depth_charts` (tar in `/mnt/project-files/depth-charts/`), the CFBDepth list,
the per-player checks and the CFBDepth exports in `/mnt/project-files/injury-reports/`.

## Method

**Slots.** Offense QB, RB, WR (X, Z, SLOT, or WIDE / WR when the exact spot is unknown) and TE.
Source order: the 2026 team charts (68 P4 + Notre Dame teams, any team, so a transfer keeps his 2026
spot), the CFBDepth WR export (Slot Rate >= 50% -> SLOT, else WIDE), the CFBD roster position of
that season, and >= 10 dropbacks in a game -> QB. For 2025 a player's 2026 chart spot is used only
when it uses the same kind of touch (dropback, carry, target) he mostly had in 2025.

**Full-strength chart, per team-game, pregame.** From the team's earlier games this season (at least
2 appearances): QB 1, RB 1, WR 3, TE 1, the players with the most touches per game when present. A
player listed out keeps his slot, because the chart comes from his usage while healthy, not from the
current team chart. A player who was a starter for the same team last season (QB >= 50% of
dropbacks, RB >= 30% of carries, WR >= 12% / TE >= 8% of targets, >= 4 games) and has not played this
season takes his slot when he is out and his touches per game were at least 80% of the in-season
leader's. Transfers are left out: in 2025 the transfers this caught had been beaten out (Mikey Keene at
Michigan, Kadin Semonza at Tulane).

**Value over the slot's replacement.** Cost of a starter out = max(starter rate - replacement rate,
0) x starter touches per game x P(out), in points. Rates are EPA per dropback / carry / target, shrunk
toward league means of starters (for the starter) or other players (for the replacement): QB 120
dropbacks (PR 34), RB 250 carries, WR 100 and TE 200 targets (from year-to-year reliability, 2018-2024).
QB rates add last season's dropbacks at half weight. The replacement is the healthy non-starter at
the same slot with the most touches per game (WR: same spot first), else the league backup rate.

**Starter rule.** `dc` and `dc_full`: most touches per game when present, so an out starter keeps his
slot for as long as he is out. `dc_total`: most touches so far this season (PR 34's QB rule), so a
starter who has been out long enough for his backup to pass him drops out of the slot.

**Baked-in discount (`dc`).** model_v2's ratings already contain the games played without him, and
str_power's prior counts 4 games, so after `m` missed games out of `k` the cost is multiplied by
1 - m / (4 + k).

**Absence information.**

| Test | Who is out |
| --- | --- |
| 2025 realized | a player with a role who did not play in the regular-season game (Josiah's assumption: injured). Bowls excluded. |
| 2026 listed | PR 34 listings: CFBDepth list + per-player check windows, conference reports where the list has no row; P(out) by status from 2025 conference reports (Out 0.998, Questionable 0.57) |
| 2026 realized | the 2025 rule applied to 2026, for a like-for-like season |

Game term = home cost minus away cost, subtracted from the model_v2 margin and passed through
model_v2's calibration slope, with nothing fitted (as in PR 34's `fixed_listed`).

## Results

Log loss change vs model_v2 and vs PR 34 `fixed_listed` (same absence information), paired bootstrap
by week (SE).

| Version | 2025 realized, FBS (n 762) | vs PR 34 | 2025 P4/ND (n 433) | vs PR 34 | 2026 listed, FBS (n 632) | vs PR 34 | 2026 P4/ND (n 353) | vs PR 34 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PR 34 fixed_listed | +0.0024 (0.0039) | | -0.0030 (0.0040) | | -0.0084 (0.0028) | | -0.0134 (0.0046) | |
| dc: out starter keeps his slot, discount | -0.0020 (0.0033) | -0.0043 (0.0017) | -0.0064 (0.0040) | -0.0034 (0.0013) | -0.0059 (0.0027) | +0.0025 (0.0019) | -0.0112 (0.0038) | +0.0022 (0.0017) |
| dc_full: dc, no discount | -0.0022 (0.0035) | -0.0046 (0.0017) | -0.0066 (0.0044) | -0.0036 (0.0015) | -0.0046 (0.0028) | +0.0038 (0.0022) | -0.0119 (0.0040) | +0.0015 (0.0017) |
| **dc_total: starter by season touches** | -0.0005 (0.0031) | -0.0028 (0.0013) | -0.0058 (0.0040) | -0.0028 (0.0013) | **-0.0082 (0.0032)** | +0.0001 (0.0013) | **-0.0144 (0.0050)** | -0.0010 (0.0015) |
| dc, QB only | -0.0022 (0.0034) | -0.0046 (0.0019) | -0.0066 (0.0038) | -0.0036 (0.0014) | -0.0046 (0.0029) | +0.0038 (0.0026) | -0.0097 (0.0040) | +0.0037 (0.0023) |
| dc, RB/WR/TE only | +0.0002 (0.0003) | | +0.0000 (0.0005) | | -0.0013 (0.0008) | | -0.0015 (0.0012) | |
| PR 34 QB + dc RB/WR/TE | +0.0003 (0.0036) | **-0.0021 (0.0007)** | -0.0056 (0.0042) | **-0.0026 (0.0009)** | -0.0086 (0.0033) | -0.0002 (0.0010) | -0.0145 (0.0053) | -0.0011 (0.0015) |
| dc, roster positions | -0.0020 (0.0033) | -0.0044 (0.0017) | -0.0064 (0.0039) | -0.0034 (0.0013) | -0.0058 (0.0027) | +0.0026 (0.0020) | -0.0109 (0.0038) | +0.0025 (0.0018) |
| dc, no season-long absentees | -0.0019 (0.0033) | -0.0042 (0.0017) | -0.0065 (0.0040) | -0.0035 (0.0013) | -0.0058 (0.0026) | +0.0026 (0.0020) | -0.0112 (0.0038) | +0.0022 (0.0017) |
| dc starters, PR 26 share points for RB/WR/TE | -0.0021 (0.0035) | -0.0045 (0.0014) | -0.0062 (0.0041) | -0.0032 (0.0012) | -0.0061 (0.0025) | +0.0023 (0.0016) | -0.0113 (0.0037) | +0.0021 (0.0016) |

model_v2 log loss: 2025 0.5115 (P4/ND 0.4542), 2026 0.5453 (P4/ND 0.5326). PR 34's skill-only part
(share points): 2025 +0.0022 (0.0008), 2026 -0.0012 (0.0015).

**Pooled, 2025 realized + 2026 listed (n 1,394; P4/ND 786):**

| Version | vs model_v2 | vs PR 34 | P4/ND vs model_v2 | P4/ND vs PR 34 |
| --- | --- | --- | --- | --- |
| PR 34 fixed_listed | -0.0025 (0.0028) | | -0.0077 (0.0031) | |
| dc | -0.0037 (0.0023) | -0.0012 (0.0015) | -0.0086 (0.0028) | -0.0009 (0.0012) |
| dc_total | **-0.0040 (0.0024)** | **-0.0015 (0.0010)** | **-0.0097 (0.0032)** | **-0.0020 (0.0010)** |
| PR 34 QB + dc RB/WR/TE | -0.0037 (0.0027) | -0.0012 (0.0006) | -0.0096 (0.0033) | -0.0019 (0.0008) |

2026 realized (the 2025 rule on 2026, FBS / P4): PR 34 -0.0027 / -0.0083, dc -0.0041 / -0.0080,
dc_total -0.0062 / -0.0111, PR 34 QB + dc skill -0.0057 / -0.0108.

Accuracy: 2025 model_v2 73.0%, PR 34 72.7%, dc 73.8%, dc_total 73.4%; 2026 listed model_v2 71.5%,
PR 34 71.8%, dc_total 71.5%.

In-sample slope of the residual on the cost (1.0 = costs right on average): 2026 listed PR 34 1.26,
dc 1.16, dc_total 1.33; 2025 realized PR 34 0.38, dc 0.79, dc_total 0.50. The depth-chart costs
are closer to face value than PR 34's in 2025, and slightly too small in 2026.

Ten versions were compared on the same two seasons, so the best pooled numbers carry some selection
optimism. The ranking of `dc` vs `dc_total` flips between 2025 and 2026; the skill-position result
(depth chart beats share points) holds in both.

## Cost by slot

See `scripts/depth_impact_slots.py` output (`data/exploration/depth_impact/slots.csv`), 2025-2026
regular season, realized absences: residual when only that starter was out, the mean depth-chart cost
for those games, and the slope of the residual on the cost.

| Slot | Starter-games out | Only starter out: n | Residual (SE) | Mean cost | Points lost per point of cost (SE) |
| --- | ---: | ---: | --- | ---: | --- |
| All starters | 2,187 | 902 | -0.12 (0.53) | 0.61 | 0.54 (0.19) |
| QB | 434 | 159 | -1.53 (1.19) | 2.61 | 0.53 (0.20) |
| RB | 276 | 97 | +0.21 (1.45) | 0.59 | 2.36 (1.25) |
| TE | 493 | 239 | +0.84 (1.11) | 0.00 | not measurable |
| WR slot | 203 | 71 | -1.75 (2.12) | 0.19 | 7.0 (4.2) |
| WR wide (CFBDepth export) | 357 | 141 | -0.65 (1.31) | 0.23 | 0.2 (2.5) |
| WR X (chart) | 113 | 48 | -0.32 (2.08) | 0.16 | -1.5 (4.6) |
| WR Z (chart) | 109 | 48 | +1.17 (2.42) | 0.18 | 1.1 (5.7) |
| WR, spot unknown | 202 | 99 | +0.89 (1.40) | 0.12 | -4.0 (4.8) |

With realized absences a QB's cost is about half right (0.53, SE 0.20). Receiver costs are small and
noisy, and splitting WRs by chart spot shows nothing. TE costs are about 0: league backup tight ends
average more EPA per target than starters (0.72 vs 0.68 in 2018-2024), so the replacement rate is
never lower. EPA per target does not see a tight end's blocking.

## Charts: backtest vs this week

Josiah (2026-11-19): the adjusted charts are for testing against each week of this season; this week
uses the current charts less every player listed on the injury reports.

**Backtest chart** (`adjusted_depth_charts_backtest_2026.csv`): the 68 team charts (offense and
defense) with listed-out players who have a real role moved back to their natural spot: 118 players
(63 offense, 55 defense; 78 at first team). Role = CFBDepth export starter share >= 0.5 (snaps per
game / team's top-N mean) or an offensive full-strength starter. Spot: his chart spot if he is still
on offense / defense, else the export alignment with most snaps (OL LT/LG/C/RG/RT, DB CB / slot / FS,
DL gaps vs outside, LB off-ball vs on the line), at the team's label for that role whose first-team
player has the lower starter share. Depth: first team for an offensive starter, else behind every
player at the label with a larger starter share. OL and defensive players are placed but not priced:
PR 26 and PR 34 found no measurable cost, and box scores do not show offensive linemen, so 2025 cannot
test it.

**Backtest on 2026 with that chart** (charted teams; other teams as `dc_total`): for each game the
players listed for that game are taken off and the next man up at the label replaces the starter.

| Version | 2026 listed FBS vs v2 | vs dc_total | 2026 listed P4/ND vs v2 | vs dc_total | 2026 realized FBS vs v2 | vs dc_total |
| --- | --- | --- | --- | --- | --- | --- |
| chart_adj: starters = backtest chart first team | -0.0074 (0.0029) | +0.0008 (0.0008) | -0.0129 (0.0044) | +0.0015 (0.0015) | -0.0061 (0.0041) | +0.0001 (0.0007) |
| chart_adj with discount | -0.0076 (0.0027) | +0.0007 (0.0006) | -0.0120 (0.0042) | +0.0024 (0.0014) | -0.0064 (0.0039) | -0.0003 (0.0008) |
| dc_total starters, next man up on the backtest chart | -0.0076 (0.0031) | +0.0007 (0.0004) | -0.0132 (0.0046) | +0.0012 (0.0008) | -0.0054 (0.0041) | +0.0007 (0.0004) |

The chart versions are within noise of `dc_total` (usage replacement) and slightly behind it. The
2026-11-19 chart is hindsight for earlier weeks (it lists players who only emerged later), and charts
for those weeks do not exist, so this is the closest test available.

**This week's chart** (`depth_charts_week12_2026.csv`): the current charts less every player the
CFBDepth list has out for the week 12 game with P(out) >= 0.5: Out, Out for Season and Questionable
(P(out) 0.57); 219 rows removed (148 Out, 71 Questionable), depth renumbered. Probable players stay.

## This week (week 12, 2026)

`impact_scores_week12_2026.csv`: every full-strength offensive starter (QB, RB, 3 WR, TE; most touches
so far this season, `dc_total`) for each FBS team's week 12 game: touches per game, EPA rate, the
replacement (next man up on this week's chart for the 68 charted teams, usage replacement for the
rest) and his rate, value over the replacement, P(out), cost. 816 starters, 86 listed out; 23 cost at
least 1 point. Largest: Ohio State QB Julian Sayin 9.6 (next: Luke Fahey), Mississippi State QB
Kamario Taylor 8.4 (next: Parker Puckett, because chart QB1 AJ Swann is Questionable), North Texas QB
Tayven Jackson 8.3, Northwestern QB Aidan Chiles 7.7, Pittsburgh QB Mason Heintschel 7.5 (next: Corey
Dailey; chart QB1 Holden Geriner is Questionable).

`week12_2026.{md,csv}` (`scripts/depth_impact_week.py`): model_v2 plus this cost, next to PR 34's
fixed_listed. 70 games, 20 move 1+ point, 5 picks flip vs model_v2 (Northwestern at Minnesota, Akron at
UMass, North Texas at Tulane, South Florida at FAU, Georgia State at Marshall). The depth-chart
adjustments are mostly smaller than PR 34's (Northwestern +7.7 vs +9.7, Pittsburgh +7.5 vs +9.6).

## Limits

- The 2025 test uses realized absences as if they were listed: it knows before kickoff who did not
  play, counts benchings, redshirts and departures as injuries, and reads a receiver with no target as
  absent. It favours every version equally, so the comparisons between versions are what it measures.
- One season with real listings (2026).
- Questionable players come off this week's chart (Josiah: less any player listed); their cost is
  still weighted by P(out) 0.57, so a Questionable QB1 makes the QB2 behind him the replacement.
- 2026 chart spots applied to 2025 assume players kept their spot; the result is the same with roster
  positions.
- EPA per target for 2024 (last season for 2025 rates) records fewer incompletion targets (64% of
  dropbacks vs 88% in 2025), so 2024 receiver rates run high; only QBs use last season's rates.
- The test fits model_v2 on 2015-2025 for 2026 (as in PR 26 and PR 34).

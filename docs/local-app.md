# Local app (Windows)

The local app is a Streamlit UI over the weekly report bundles and the warehouse, plus a local
engine (`cfb.engine`) that does the whole weekly pipeline on the PC: CFBD pulls, core tables,
features, predictions, the report and a backup to Cloudflare R2. With the engine on a schedule
the PC needs no cloud session. Plans: `/mnt/project-files/plan/local-package.md`,
`/mnt/project-files/plan/local-engine.md`.

## Editions

Two editions install side by side and share nothing (`cfb.edition`), so the experimental one can
never overwrite main's data:

| | Main | Experimental |
| --- | --- | --- |
| App, Start menu entry | CFB Predictor | CFB Predictor Experimental |
| Code | `main` | the experimental branch (depth charts, injuries, availability) |
| Data, logs, engine status | `%LOCALAPPDATA%\cfb\` | `%LOCALAPPDATA%\cfb-exp\` |
| User config | `%APPDATA%\cfb\config.toml` | `%APPDATA%\cfb-exp\config.toml` |
| Keys (Windows Credential Manager) | service `cfb` | service `cfb-exp`; without its own `cfbd_api_key` it reads main's (R2 keys are never shared) |
| R2 bucket | `cfb-warehouse` | `cfb-warehouse-exp` (the main bucket is refused) |
| Scheduled tasks | "CFB Predictor weekly update", Monday 05:30 | "CFB Predictor Experimental weekly update", Monday 06:00 |
| Release tags | `build-<n>` | `exp-build-<n>` (pre-release) |

The edition is stamped into the build (`cfb/_build.py`, `EDITION`); `CFB_EDITION` overrides it
for tests and source checkouts.

## Install (packaged app)

1. Get `CFB-Predictor-Setup-build-<n>.exe` (experimental: `CFB-Predictor-Experimental-Setup-build-<n>.exe`)
   from a `windows-app` workflow run's artifact, or from a GitHub Release when one was published
   (`.github/workflows/windows-app.yml`; releases only on a manual run with "publish" ticked).
2. Run it. It installs per user (no admin rights) under `%LOCALAPPDATA%\Programs\CFB Predictor`
   (or `...\CFB Predictor Experimental`). The installer is unsigned, so Windows SmartScreen asks
   first: "More info" > "Run anyway".
3. Start the app from the Start menu. A console window opens (closing it stops the UI, not an
   engine run) and the app opens in the browser at `http://127.0.0.1:8501` (another free port if
   that one is taken).
4. Settings page, **Cloudflare connection**: paste the edition's R2 token's Access Key ID, Secret
   Access Key and the account ID (or the S3 endpoint `https://<account id>.r2.cloudflarestorage.com`
   shown with the token), then **Save and test**. This is needed once: the keys go to Windows
   Credential Manager (or, if that does not keep them, `secrets.json` in the edition's
   `%APPDATA%` folder, encrypted for the Windows user with DPAPI) and load on every launch, also
   after installing a new build. A read-only token syncs; backing up after each engine run needs
   Object Read & Write on the edition's bucket. Under **Other keys**, save `cfbd_api_key` for the
   engine's CFBD pulls ("Test the CFBD key" shows the tier and the calls left). Pasted values are
   cleaned: surrounding spaces and quotes and a leading `Bearer ` are dropped.
5. Data page: **Bootstrap from the cloud** (first time), then **Run weekly update now**.
6. Settings > Schedule: turn on the weekly update (and, if wanted, the daily light refresh).

Uninstalling removes the scheduled tasks; data and keys stay.

## Local engine

```
"CFB Predictor.exe" engine status
"CFB Predictor.exe" engine bootstrap [--source auto|r2|cfbd] [--force]
"CFB Predictor.exe" engine refresh  [--season N]     pull the season (--force, dated copies)
"CFB Predictor.exe" engine build    [--season N]     core tables, unit ratings, the season's features
"CFB Predictor.exe" engine week     [--season N --week N]   predict and write the report
"CFB Predictor.exe" engine all      the weekly update: bootstrap if needed, refresh, build, week, back up
"CFB Predictor.exe" engine light    games, lines, polls, weather (about 7 calls); core; the week's report
"CFB Predictor.exe" engine check    rerun the cloud's report week on the bootstrapped data and compare
"CFB Predictor.exe" engine backup   publish to R2 (slim and full channels)
"CFB Predictor.exe" engine schedule show | install weekly|daily [--day MON] [--time 05:30] | remove all
```

`"CFB Engine.exe"` (same folder) is the same program without a console window; Task Scheduler and
the app's buttons use it. From a source checkout: `python -m cfb.engine <command>`.

- **Bootstrap.** Downloads the R2 full channel (about 580 MB: raw API JSON, tables, core, features,
  the dated SP+/FPI/SRS/CORE copies and the cloud's latest report) into the data dir, copies the
  dated copies to `%LOCALAPPDATA%\cfb\asof`, and keeps the report as the reference for `check`.
  Interrupted downloads resume. `--source cfbd` pulls everything from CFBD instead (about 2,100
  calls; `/stats/player/success/game` takes about 12 s per call and refuses parallel requests,
  so about an hour, plus about 40 minutes of feature building). The CFBD pull cannot recover
  the dated rating copies the cloud saved, so prefer R2.
- **all** is one `cfb.run_week` pass, the same as the cloud's weekly run: refresh the current
  season from CFBD (about 220 to 280 calls), core, unit ratings and features for the season,
  every configured model, the report in `data\reports\<season>_<type>_<week>\`, then the backup.
  February to July it does nothing (no games); `--season`/`--week` run a given week anyway.
- **check** reruns the reference week with no refresh and the stored features and compares every
  model's win probability (and the primary's margin and total) game by game, within the bundle's
  rounding (4 and 2 decimals). It is exact only before the first refresh.
- **Backup** (Settings > Backup, on by default) publishes the slim channel (what the app's Sync and
  cloud threads read) and the full channel (a complete copy, raw included) to the edition's bucket.
  A failed backup does not fail the run; the status says why (a read-only key is the usual cause).

Only one engine run at a time (`engine.lock`); a second one exits with code 3. Exit codes: 0 done
or nothing to do, 1 failed, 2 done with a warning (failed requests, failed backup), 3 busy.

| What | Where (main edition) |
| --- | --- |
| Engine data dir | `%LOCALAPPDATA%\cfb\data\` |
| Dated rating copies | `%LOCALAPPDATA%\cfb\asof\` |
| Logs (last 60 runs) | `%LOCALAPPDATA%\cfb\logs\engine-<UTC time>-<command>.log` |
| Last run, every run | `%LOCALAPPDATA%\cfb\engine-status.json`, `runs.jsonl` |
| Preferences | `%LOCALAPPDATA%\cfb\engine.json` (`{"backup": true}`) |
| Reference report for `check` | `%LOCALAPPDATA%\cfb\reference\<key>\bundle.json` |
| Cloud copy (Data > Sync now) | `%LOCALAPPDATA%\cfb\warehouse\` (`versions/<version>/`, `CURRENT`) |
| Keys | Windows Credential Manager, service `cfb`; fallback `%APPDATA%\cfb\secrets.json` (DPAPI) |

The pages read the engine's data dir or the synced cloud copy, whichever has the newer report
week (the engine's on a tie).

## Schedule (Windows Task Scheduler)

Settings > Schedule creates the tasks (`cfb.schedule`, `schtasks /Create /XML`); times are the
PC's local time:

- **Weekly update**: `engine all`, default Monday 05:30 (experimental 06:00), after Sunday's data settles.
- **Daily light refresh** (optional): `engine light`, default Tuesday to Sunday 06:00 (experimental 06:30).

Both run as soon as possible after a missed start (PC off or asleep), wake the PC to run (Windows
power options must allow wake timers), run on battery, at most one at a time, and stop after
4 hours. They run while the user is signed in, a locked screen included: the keys are in that
user's Credential Manager, so no password is stored and no admin rights are needed. CFBD calls:
about 220 a week plus about 7 a day, far under the 75,000 a month of Tier 3.

## Pages

- **This week**: picks, win probability, spread and total against the market, filters, top reasons.
- **Picks vs results**: the chosen week's picks as its weekly run published them (bundle.json, never
  recomputed) next to the final scores, with right/wrong, log loss, the spread and total against the line at
  the run, and FBS vs FBS totals per model. Scores come from the synced database (later reports' last-week
  sections fill in when its tables are missing); games without one show as pending until the next sync.
- **Matchup**: any two teams at any date, with a QB-out override. It computes features live:
  about 15 s with model_v2, about 2 s with model_v1 (measured on the cloud container).
- **Teams**: model top 50 against the AP, Coaches and CFP polls; each team's games.
- **Performance**: season-to-date log loss, Brier and accuracy against the baselines, calibration.
- **Data**: local engine (bootstrap, Run weekly update now, Light refresh, Check against the
  cloud, Back up now, progress of a running job, the last run and recent runs) and the cloud copy
  (Sync now).
- **Settings**: edition, Cloudflare connection (Save and test, Test connection, Forget), paths and key
  store, other keys (Test the CFBD key), schedule, backup.

## From a source checkout

```bash
pip install -e '.[ui]'
cfb-ui                 # or: python -m cfb.ui.launch --port 8501 --no-browser
python -m cfb.engine status
```

It reads `data/` (or `$CFB_DATA_DIR`), so it works in a cloud thread on a restored snapshot too.

## Building the app by hand (Windows)

```bash
pip install ".[ui,dev]" pyinstaller
printf 'BUILD = "0"\nCOMMIT = "local"\nEDITION = "main"\n' > src/cfb/_build.py   # optional
pyinstaller packaging/cfb_app.spec --noconfirm
python packaging/smoke.py "dist/CFB Predictor/CFB Predictor.exe"
iscc /DBuild=0 /DEdition=main packaging\installer.iss
```

The build is a PyInstaller one-folder app with two programs, `CFB Predictor.exe` (console) and
`CFB Engine.exe` (no window), from `packaging/cfb_app.py`; `packaging/cfb_app.spec` lists the
bundled files (Streamlit's static files, `cfb/ui/app.py`, `dashboard/config.toml`). If
`selftest` reports a missing module after a code change, add it to the spec's hidden imports.
The CI smoke test runs `selftest --system` (a Credential Manager round trip and a Task Scheduler
register, read back and delete), the engine end to end on the synthetic store, and the UI server.

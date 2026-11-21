# Local app (Windows)

The local app is a Streamlit UI over the weekly report bundles and the warehouse. It runs on
the PC and never calls the CFBD API. The cloud weekly run builds the data and publishes it to
Cloudflare R2, and the app syncs it from there (`cfb.warehouse_sync`, built in its own step).
Plan: `/mnt/project-files/plan/local-package.md`.

## Install (packaged app)

1. Download `CFB-Predictor-Setup-build-<n>.exe` from the repo's latest GitHub Release
   (built by `.github/workflows/windows-app.yml` when it is run by hand on main; merges do not publish builds).
2. Run it. It installs per user (no admin rights) under `%LOCALAPPDATA%\Programs\CFB Predictor`.
   The installer is unsigned, so Windows SmartScreen asks first: "More info" > "Run anyway".
3. Start "CFB Predictor" from the Start menu. A console window opens (closing it stops the app)
   and the app opens in the browser at `http://127.0.0.1:8501` (another free port if that one is taken).
4. Settings page: save the R2 read-only keys (`r2_access_key_id`, `r2_secret_access_key`,
   `r2_account_id`). They go to Windows Credential Manager. Then Data > Sync now.

The sidebar shows "Update available" when a newer build is on GitHub Releases. Run the new
installer over the old one; data and keys are kept.

| What | Where |
| --- | --- |
| Synced database | `%LOCALAPPDATA%\cfb\warehouse\` (`versions/<version>/`, `CURRENT`) |
| User config (overrides the bundled `dashboard/config.toml`) | `%APPDATA%\cfb\config.toml` |
| Keys | Windows Credential Manager, service `cfb` |

`"CFB Predictor.exe" sync` downloads the newest database without the UI (for Task Scheduler).
`"CFB Predictor.exe" selftest` checks the build.

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
- **Data**: database version, CFBD quota at the last publish, Sync now.
- **Settings**: paths and keys.

## From a source checkout

```bash
pip install -e '.[ui]'
cfb-ui                 # or: python -m cfb.ui.launch --port 8501 --no-browser
```

It reads `data/` (or `$CFB_DATA_DIR`), so it works in a cloud thread on a restored snapshot too.

## Building the app by hand (Windows)

```bash
pip install ".[ui,dev]" pyinstaller
pyinstaller packaging/cfb_app.spec --noconfirm
python packaging/smoke.py "dist/CFB Predictor/CFB Predictor.exe"
iscc /DBuild=0 packaging\installer.iss
```

The build is a PyInstaller one-folder app: `packaging/cfb_app.py` is the entry point and
`packaging/cfb_app.spec` lists the bundled files (Streamlit's static files, `cfb/ui/app.py`,
`dashboard/config.toml`). If `selftest` reports a missing module after a code change, add it to
the spec's hidden imports.

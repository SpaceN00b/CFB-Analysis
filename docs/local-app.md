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
4. Settings page, **Cloudflare connection**: paste the R2 read-only token's Access Key ID, Secret
   Access Key and the account ID (or the S3 endpoint `https://<account id>.r2.cloudflarestorage.com`
   shown with the token), then **Save and test**. This is needed once: the keys go to Windows
   Credential Manager (or, if that does not keep them, `%APPDATA%\cfb\secrets.json`, encrypted for
   the Windows user with DPAPI) and load on every launch, also after installing a new build.
   With "Sync the database when the app starts" on (the default), each launch downloads the
   changed files in the background; Data > Sync now still works by hand.

The sidebar shows "Update available" when a newer build is on GitHub Releases. Run the new
installer over the old one; data and keys are kept.

| What | Where |
| --- | --- |
| Synced database | `%LOCALAPPDATA%\cfb\warehouse\` (`versions/<version>/`, `CURRENT`) |
| User config (overrides the bundled `dashboard/config.toml`) | `%APPDATA%\cfb\config.toml` |
| Keys | Windows Credential Manager, service `cfb`; fallback `%APPDATA%\cfb\secrets.json` (DPAPI) |
| App preferences (sync at launch) | `%APPDATA%\cfb\app.json` |
| Player availability (test model) | `%LOCALAPPDATA%\cfb\availability\` (`<week>.zip`, `overrides.json`) |

`"CFB Predictor.exe" sync` downloads the newest database without the UI (for Task Scheduler).
`"CFB Predictor.exe" selftest` checks the build.

## Pages

- **This week**: picks, win probability, spread and total against the market, filters, top reasons.
- **Matchup**: any two teams at any date, with a QB-out override. It computes features live:
  about 15 s with model_v2, about 2 s with model_v1 (measured on the cloud container).
- **Teams**: model top 50 against the AP, Coaches and CFP polls; each team's games.
- **Performance**: season-to-date log loss, Brier and accuracy against the baselines, calibration.
- **Data**: database version, CFBD quota at the last publish, Sync now.
- **Players** (test model, PR 34/35 branch builds): this week's picks from model_v2 plus the depth-chart
  injury term, and a table per team to mark QB / RB / WR / TE players **Available** or **Out** over the
  injury list. A team's cost is recomputed on the PC (`cfb.ui.availability`, about 50 ms) with the same
  code as the cloud run, so with no changes it shows the cloud's numbers. Settings are saved per week in
  `overrides.json`. The week's inputs come from the bucket (`extras/availability/`), downloaded with
  each sync; the cloud builds them with `scripts/availability_export.py --publish` (docs/depth-impact.md).
  Defense (PR 37, packages from week 12 on): DL / LB / DB regulars can be marked too; an out regular
  costs his disruption value (sacks, TFL, INT, pass breakups, hurries) over the next man up on this
  week's defensive chart (docs/defense-impact.md). "Include defense" on the Picks tab turns the
  defensive term off; it tested at no measurable gain. Run `scripts/defense_impact_select.py --week N`
  before the export to include it.
- **Settings**: Cloudflare connection (save once, test, forget), sync at launch, paths, other keys.

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

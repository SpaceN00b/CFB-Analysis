# CFB-Analysis

FBS game predictor on the CFBD API: ingest -> core tables -> feature groups -> models -> weekly report.

## Layout
- `src/cfb/ingest.py` CFBD pull into `data/raw` (json.gz) + `data/tables` (parquet); `store.py` storage.
- `src/cfb/core/` game index, as-of views (`AsOfView`), leak test. `src/cfb/features/<area>/` feature groups.
- `src/cfb/model/`, `src/cfb/predict.py` models (`model_v2` is the report's primary). `src/cfb/eval/` metrics, baselines.
- `src/cfb/report/` bundle, report.md, dashboard, summary. `src/cfb/run_week.py` the weekly pipeline.
- `docs/` one file per area (foundations, data-coverage, strength, matchup, coaching, context, eval, model,
  points, priors, qb, report). Read only the one the task needs.

## Commands
- Setup: `pip install -e '.[dev]'`
- Tests without data (CI): `python -m pytest -q`; snapshot tests skip when `data/` is empty.
  Run only the affected test file while iterating.
- Restore data (about 430 MB from project files, only when the task needs real data):
  `tar -xf /mnt/project-files/cfbd-data/$(cat /mnt/project-files/cfbd-data/LATEST)` then
  `tar -xf /mnt/project-files/cfbd-data/$(cat /mnt/project-files/cfbd-data/FEATURES_LATEST)`
- Weekly run: `python -m cfb.run_week --publish /mnt/project-files/reports` (restores both tars itself).
  It prints `summary.md` last; post that instead of reading report.md.

## Keep context small
- Do not cat `bundle.json` (~600 KB), `dashboard.html` (~700 KB), `report.md` (~50 KB) or files under
  `data/`; query them with a short script and print only what is needed.
- Pipe long command output through `tail`/`grep`.

## Project files rules
- Never delete or overwrite `/mnt/project-files/cfbd-data/asof/` (dated rating copies; cannot be re-pulled).
  Weekly ingest must pass `--asof-archive /mnt/project-files/cfbd-data/asof`.
- New snapshot or features tars get a new dated name; update `LATEST` / `FEATURES_LATEST` and
  `cfbd-data/README.md`. Never overwrite an existing tar.

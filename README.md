# CFB-Analysis

Predictive model for NCAA Division I FBS football, built on the
[CollegeFootballData.com](https://collegefootballdata.com/) API.

## Plan

1. **Data pipeline** (this package, `cfb.ingest`): pull 2014 through 2026 into a local store.
2. **Features**: one row per game from pre-kickoff information only. Opponent-adjusted
   efficiency, SOS, strength of record, home/road effects, run/pass matchup, and coaching
   (record vs. the spread, 4th-down aggressiveness, run/pass mix, offensive flexibility).
3. **Model**: winner and win probability. Train on seasons through 2025, test on 2026
   walk-forward (predict week N from data through week N-1). Baselines: closing-line
   implied probability and a simple rating model.
4. **Outputs**: dashboard and weekly report. Spread and totals come after step 3 performs well.

## Setup

```bash
pip install -e '.[dev]'
export CFBD_API_KEY=...        # never commit the key
python -m cfb.ingest --list    # dataset names
python -m cfb.ingest --start 2014 --end 2026
python -m cfb.ingest --start 2026 --end 2026 --force   # refresh the current season
pytest
```

## Data layout

`data/` is git-ignored.

- `data/raw/<dataset>/<key>.json.gz`: exact API responses (source of truth).
- `data/tables/<dataset>/<key>.parquet`: flattened copy. Nested objects become dotted
  columns; lists inside a record are stored as JSON strings.

Load a dataset with `cfb.store.Store("data").load("games")`.

## Status

The endpoint paths and parameters in `cfb/ingest.py` were written from the CFBD v2 API
documentation and have **not yet been run against the live API**. The unit tests cover
retry logic, storage, and task planning only.

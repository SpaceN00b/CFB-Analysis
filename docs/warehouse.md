# Warehouse publish and sync (Cloudflare R2)

The cloud weekly run publishes the Parquet warehouse to the private R2 bucket `cfb-warehouse`;
the PC app syncs it. The cloud is the only writer. Code: `src/cfb/warehouse_sync.py`.

## What is published

Every file under `tables/`, `core/`, `derived/`, `features/`, `predictions/`, `reports/` of the
data dir. `raw/` (API JSON) and the dated-copy archive stay in the cloud.

First publish, week 12 2026 (2026-11-18): 2,200 files, 1,989 distinct blobs, 302 MB
(tables 217 MB, core 49 MB, features 23 MB, derived 12 MB, reports 1.4 MB, predictions 0.9 MB).
Upload from the cloud took 1 min 42 s; a full sync back took 1 min 15 s; a second publish
with no changes uploaded nothing.

## Bucket layout

| Key | Content |
| --- | --- |
| `blobs/<sha256>` | file bytes, one object per distinct content |
| `manifests/<version>.json` | each kept manifest (last 3) |
| `manifests/index.json` | `{"versions": [...]}`, oldest first |
| `manifest.json` | current manifest, written after its blobs |

Manifest fields: `schema` (1), `version` (`<key>-<UTC stamp>`), `key`, `season`, `season_type`,
`week`, `created_utc`, `git_commit`, `cfbd_quota` (CFBD `/info`: tier, monthly_limit, remaining,
reset_at), `total_bytes`, `files` (`relpath -> {size, sha256}`). Blobs that no kept manifest
references are deleted after each publish.

## Local layout (PC)

```
<base>/CURRENT                         version in use (replaced with os.replace)
<base>/versions/<version>/             a full data dir + manifest.json; pass to Warehouse
<base>/sync.lock                       OS-locked while a sync runs (released if the app exits)
```

Sync downloads only blobs not in the current version, checks size and sha256, hard-links
unchanged files (copies where linking fails), then switches `CURRENT`. A failed sync leaves the
current version in place; an interrupted one resumes from `versions/.staging-<version>/`.
Two versions are kept; deleting a locked one on Windows is retried at the next sync.
`current_data_dir(base)` returns the data dir in use, or None before the first sync.

## Commands

```
python -m cfb.warehouse_sync publish --data-dir data [--key 2026_regular_12] [--dry-run] [--force]
python -m cfb.warehouse_sync sync --base <dir>
python -m cfb.warehouse_sync status [--base <dir>]
```

`python -m cfb.run_week --publish ...` runs publish as step 8 when R2 settings exist
(`--no-warehouse` turns it off, `--warehouse` forces it). Runs without `--publish` (ad-hoc,
past weeks) never publish. A failed upload leaves the report in place, logs the error and
makes the command exit 2.

## Settings and credentials

Environment first, then a TOML file (`$CFB_R2_CONFIG`, else `/mnt/project-files/warehouse/r2.toml`)
for the non-secret `account_id` and `bucket`.

| Variable | Use |
| --- | --- |
| `R2_ACCOUNT_ID`, `R2_BUCKET` (default `cfb-warehouse`) | where |
| `CLOUDFLARE_API_TOKEN` | Cloudflare REST API, Bearer. Unset in the cloud: the proxy injects a custom token with "Workers R2 Storage: Edit" for `api.cloudflare.com` |
| `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | S3 API instead (boto3: `pip install 'cfb[r2]'`); the PC uses a read-only pair |
| `CFB_WAREHOUSE_LOCAL` | a folder standing in for the bucket (tests, dry runs) |

Bucket-scoped R2 tokens work only with the S3 API; the REST object endpoints return
"Authentication error" (code 10000) for them (checked 2026-11-18).
Tests never reach the bucket: `tests/conftest.py` clears these settings.

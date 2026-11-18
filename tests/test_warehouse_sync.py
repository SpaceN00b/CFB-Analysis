"""cfb.warehouse_sync: publish to a fake bucket (a folder), sync to a local base, REST store calls."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from cfb import warehouse_sync as ws

T0 = datetime(2026, 11, 18, 6, 0, tzinfo=timezone.utc)


def quiet(_msg):
    pass


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    files = {
        "tables/games/2025.parquet": b"g2025",
        "tables/games/2026.parquet": b"g2026-v1",
        "core/game_index.parquet": b"core",
        "features/catalog.json": b"{}",
        "features/copy_of_core.parquet": b"core",  # same content as another file: one blob
        "predictions/model_v2/2026.parquet": b"pred",
        "reports/2026_regular_12/bundle.json": b"{\"x\": 1}",
        "raw/games/2026.json.gz": b"raw stays in the cloud",
        "asof/x.parquet": b"archive stays too",
    }
    for rel, b in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(b)
    return d


@pytest.fixture
def bucket(tmp_path):
    return ws.LocalStore(tmp_path / "bucket")


def blobs(store):
    return sorted(p.name for p in (store.root / "blobs").iterdir())


def test_publish_uploads_changed_files_only(data, bucket):
    r1 = ws.publish(data, bucket, now=T0, log=quiet)
    assert r1.changed and r1.uploaded_files == 6  # 7 published files, two share content
    m = json.loads(bucket.get("manifest.json"))
    assert m["key"] == "2026_regular_12" and (m["season"], m["season_type"], m["week"]) == (2026, "regular", 12)
    assert m["version"] == "2026_regular_12-20261118T060000Z"
    assert not any(k.startswith(("raw/", "asof/")) for k in m["files"])
    assert m["total_bytes"] == sum(f["size"] for f in m["files"].values())

    r2 = ws.publish(data, bucket, now=T0 + timedelta(hours=1), log=quiet)
    assert not r2.changed and r2.uploaded_files == 0 and r2.version == r1.version

    (data / "tables/games/2026.parquet").write_bytes(b"g2026-v2")
    r3 = ws.publish(data, bucket, now=T0 + timedelta(days=7), log=quiet)
    assert r3.changed and r3.uploaded_files == 1 and r3.uploaded_bytes == len(b"g2026-v2")
    assert json.loads(bucket.get("manifests/index.json"))["versions"] == [r1.version, r3.version]


def test_dry_run_writes_nothing(data, bucket):
    r = ws.publish(data, bucket, now=T0, dry_run=True, log=quiet)
    assert r.uploaded_files == 6 and bucket.get("manifest.json") is None


def test_prune_keeps_blobs_of_kept_manifests(data, bucket):
    versions = []
    for i in range(5):
        (data / "tables/games/2026.parquet").write_bytes(f"v{i}".encode())
        versions.append(ws.publish(data, bucket, now=T0 + timedelta(days=i), log=quiet).version)
    index = json.loads(bucket.get("manifests/index.json"))["versions"]
    assert index == versions[-ws.KEEP_MANIFESTS:]
    assert bucket.get(f"manifests/{versions[0]}.json") is None
    referenced = {f["sha256"] for v in index for f in json.loads(bucket.get(f"manifests/{v}.json"))["files"].values()}
    assert set(blobs(bucket)) == referenced


def test_sync_downloads_verifies_and_switches(data, bucket, tmp_path):
    base = tmp_path / "pc"
    assert ws.current_data_dir(base) is None
    ws.publish(data, bucket, now=T0, log=quiet)
    s1 = ws.sync(base, bucket, log=quiet)
    assert s1.changed and s1.downloaded_files == 6 and s1.linked_files == 1
    d1 = ws.current_data_dir(base)
    assert d1 == s1.data_dir and (d1 / "tables/games/2026.parquet").read_bytes() == b"g2026-v1"
    assert not (d1 / "raw").exists()

    s2 = ws.sync(base, bucket, log=quiet)
    assert not s2.changed and s2.downloaded_files == 0

    (data / "tables/games/2026.parquet").write_bytes(b"g2026-v2")
    (data / "reports/2026_regular_12/bundle.json").unlink()
    ws.publish(data, bucket, now=T0 + timedelta(days=7), log=quiet)
    s3 = ws.sync(base, bucket, log=quiet)
    assert s3.downloaded_files == 1 and s3.downloaded_bytes == len(b"g2026-v2")
    d3 = ws.current_data_dir(base)
    assert d3 != d1 and (d3 / "tables/games/2026.parquet").read_bytes() == b"g2026-v2"
    assert (d3 / "core/game_index.parquet").read_bytes() == b"core"
    assert not (d3 / "reports/2026_regular_12/bundle.json").exists()
    assert_unlocked(base)


def test_sync_rejects_bad_checksum_and_keeps_current(data, bucket, tmp_path):
    base = tmp_path / "pc"
    ws.publish(data, bucket, now=T0, log=quiet)
    ws.sync(base, bucket, log=quiet)
    before = ws.current_version(base)
    (data / "tables/games/2026.parquet").write_bytes(b"new")
    m = ws.publish(data, bucket, now=T0 + timedelta(days=1), log=quiet).manifest
    sha = m["files"]["tables/games/2026.parquet"]["sha256"]
    (bucket.root / "blobs" / sha).write_bytes(b"tampered")
    with pytest.raises(ws.WarehouseError, match="checksum"):
        ws.sync(base, bucket, log=quiet)
    assert ws.current_version(base) == before
    assert_unlocked(base)


def test_sync_old_versions_pruned(data, bucket, tmp_path):
    base = tmp_path / "pc"
    for i in range(4):
        (data / "tables/games/2026.parquet").write_bytes(f"v{i}".encode())
        ws.publish(data, bucket, now=T0 + timedelta(days=i), log=quiet)
        ws.sync(base, bucket, log=quiet)
    dirs = [p for p in (base / "versions").iterdir() if p.is_dir()]
    assert len(dirs) == 2 and ws.current_data_dir(base) in dirs


def assert_unlocked(base):
    with ws._Lock(base / "sync.lock"):
        pass


def test_sync_without_manifest_and_with_lock(bucket, tmp_path):
    with pytest.raises(ws.WarehouseError, match="no manifest"):
        ws.sync(tmp_path / "pc", bucket, log=quiet)
    base = tmp_path / "pc2"
    with ws._Lock(base / "sync.lock"):  # held by another sync (other handle, as another process would be)
        with pytest.raises(ws.SyncBusy, match="another sync"):
            ws.sync(base, bucket, log=quiet)
    assert_unlocked(base)


def test_leftover_lock_file_does_not_block(data, bucket, tmp_path):
    """A sync.lock left by a closed app or a killed sync (as build-5 left one) is not a lock."""
    base = tmp_path / "pc"
    base.mkdir()
    (base / "sync.lock").write_text("12345")
    ws.publish(data, bucket, now=T0, log=quiet)
    assert ws.sync(base, bucket, log=quiet).changed


def test_settings_env_over_config(tmp_path):
    cfg = tmp_path / "r2.toml"
    cfg.write_text('account_id = "acct-file"\nbucket = "b-file"\n')
    s = ws.r2_settings(cfg, env={})
    assert (s.account_id, s.bucket, s.backend, s.configured) == ("acct-file", "b-file", "cloudflare-api", True)
    s = ws.r2_settings(cfg, env={"R2_ACCOUNT_ID": "acct-env", "R2_ACCESS_KEY_ID": "k", "R2_SECRET_ACCESS_KEY": "s"})
    assert (s.account_id, s.bucket, s.backend) == ("acct-env", "b-file", "s3")
    s = ws.r2_settings(tmp_path / "missing.toml", env={})
    assert not s.configured and s.bucket == ws.DEFAULT_BUCKET


class FakeResp:
    def __init__(self, status, content=b""):
        self.status_code, self.content, self.text = status, content, content.decode(errors="replace")

    def iter_content(self, n):
        yield self.content


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, headers=None, timeout=None, **kw):
        self.calls.append((method, url, headers, kw.get("data")))
        return self.responses.pop(0)


def test_cloudflare_store_calls_and_retries(tmp_path):
    sess = FakeSession([FakeResp(404), FakeResp(503), FakeResp(200, b"{}"), FakeResp(200, b"abc"), FakeResp(403, b"no")])
    st = ws.CloudflareAPIStore("acct", "bkt", token=None, session=sess, sleep=lambda s: None)
    assert st.get("manifest.json") is None
    st.put_bytes("manifest.json", b"{}")  # 503 then 200
    st.download("blobs/x", tmp_path / "x")
    assert (tmp_path / "x").read_bytes() == b"abc"
    with pytest.raises(ws.WarehouseError, match="HTTP 403"):
        st.delete("blobs/y")
    url = "https://api.cloudflare.com/client/v4/accounts/acct/r2/buckets/bkt/objects/manifest.json"
    assert sess.calls[0][:2] == ("GET", url) and "Authorization" not in sess.calls[0][2]
    assert [c[0] for c in sess.calls] == ["GET", "PUT", "PUT", "GET", "DELETE"]
    assert sess.calls[2][2]["Content-Type"] == "application/json"  # headers kept on the retry
    st2 = ws.CloudflareAPIStore("acct", "bkt", token="t", session=FakeSession([FakeResp(200)]))
    st2.delete("k")
    assert st2.session.calls[0][2]["Authorization"] == "Bearer t"


def test_run_week_publishes_warehouse_with_publish(synthetic, tmp_path, monkeypatch):
    import shutil

    from cfb.core.synthetic import SEASONS
    from cfb.report.config import ReportConfig
    from cfb.run_week import run

    root = tmp_path / "data"
    shutil.copytree(synthetic.root, root)
    live = max(SEASONS)
    cfg = ReportConfig(primary="baseline_elo", baselines=("baseline_close",), market_models=("baseline_close",),
                       build_features=False)
    monkeypatch.setenv("CFB_WAREHOUSE_LOCAL", str(tmp_path / "bucket"))
    res = run(root, cfg, season=live, week=4, do_refresh=False, publish_dir=tmp_path / "pub",
              publish_warehouse_to_r2=None)
    assert res.warehouse_error is None and res.warehouse.changed
    m = json.loads((tmp_path / "bucket" / "manifest.json").read_bytes())
    assert m["key"] == res.slate.key and any(k.startswith("reports/") for k in m["files"])
    res2 = run(root, cfg, season=live, week=4, do_refresh=False, publish_warehouse_to_r2=None)
    assert res2.warehouse is None  # no --publish: no warehouse step

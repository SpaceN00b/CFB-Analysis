"""The PC's Cloudflare R2 connection (cfb.ui.connection): save the key pair once, where it went,
and the bucket check."""
from __future__ import annotations

import json

import pytest


def test_connection_save_status_and_forget(monkeypatch, tmp_path):
    from cfb.ui import connection as C

    acc = "0123456789abcdef0123456789abcdef"
    assert C.parse_account_id(f"https://{acc}.r2.cloudflarestorage.com") == acc
    assert C.parse_account_id(acc.upper()) == acc
    assert C.parse_account_id("nope") is None
    assert not C.status().configured
    with pytest.raises(ValueError):
        C.save("not an id", "k", "s")
    saved = C.save(f"https://{acc}.r2.cloudflarestorage.com/cfb-warehouse", "kid", "sec", "")
    assert set(saved) == {"r2_account_id", "r2_access_key_id", "r2_secret_access_key"}
    st = C.status()
    assert st.configured and st.where() == "the encrypted keys file"
    C.save("", "", "new-secret")  # blank fields keep the saved values
    from cfb import settings
    assert settings.secret("r2_access_key_id") == "kid" and settings.secret("r2_secret_access_key") == "new-secret"
    C.forget()
    assert not C.status().configured


def test_connection_check_reads_the_manifest(monkeypatch, tmp_path):
    from cfb.ui import connection as C

    bucket = tmp_path / "bucket"
    bucket.mkdir()
    (bucket / "manifest.json").write_text(json.dumps({"version": "v1"}))
    monkeypatch.setenv("CFB_WAREHOUSE_LOCAL", str(bucket))
    C.save("0123456789abcdef0123456789abcdef", "kid", "sec")
    assert C.check() == "v1"

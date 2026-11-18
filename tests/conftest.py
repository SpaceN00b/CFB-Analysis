"""Shared fixtures.

    synthetic   a small fake warehouse, always available (CI has no data snapshot)
    snapshot    the real data under data/ (or $CFB_DATA_DIR); tests that use it are
                skipped when the core tables are not there
"""
import os
from pathlib import Path

import pytest

from cfb.core import Warehouse
from cfb.core.synthetic import synthetic_warehouse


@pytest.fixture(autouse=True)
def no_r2(monkeypatch, tmp_path_factory):
    """Tests never reach the real bucket: no R2 env settings or keyring, no project-files r2.toml."""
    from cfb import settings

    monkeypatch.setattr(settings, "_keyring", lambda: None)
    for name in ("R2_ACCOUNT_ID", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
                 "CLOUDFLARE_API_TOKEN", "CFB_WAREHOUSE_LOCAL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CFB_R2_CONFIG", str(tmp_path_factory.getbasetemp() / "no-r2.toml"))


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory) -> Warehouse:
    return synthetic_warehouse(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture(scope="session")
def snapshot() -> Warehouse:
    root = Path(os.environ.get("CFB_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    if not (root / "core" / "game_index.parquet").exists():
        pytest.skip("no data snapshot with core tables (run python -m cfb.core.build)")
    return Warehouse(root)

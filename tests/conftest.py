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


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory) -> Warehouse:
    return synthetic_warehouse(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture(scope="session")
def snapshot() -> Warehouse:
    root = Path(os.environ.get("CFB_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    if not (root / "core" / "game_index.parquet").exists():
        pytest.skip("no data snapshot with core tables (run python -m cfb.core.build)")
    return Warehouse(root)

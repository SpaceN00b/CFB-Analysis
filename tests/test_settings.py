"""cfb.settings: path defaults (cloud vs packaged app), environment overrides, secrets."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from cfb import settings
from cfb.client import CFBDClient

ENV_VARS = ("CFB_DATA_DIR", "CFB_CONFIG", "CFB_WAREHOUSE_DIR", "CFB_PROJECT_FILES", "CFB_SNAPSHOT_DIR", "CFB_ASOF_ARCHIVE",
            *settings.SECRET_ENV.values())


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    # No real project files mount and no real keyring in these tests.
    monkeypatch.setenv("CFB_PROJECT_FILES", str(tmp_path / "no-mount"))
    monkeypatch.setattr(settings, "_keyring", lambda: None)
    monkeypatch.setenv("CFB_SECRETS_FILE", str(tmp_path / "secrets.json"))
    monkeypatch.delattr(sys, "frozen", raising=False)


def freeze(monkeypatch, bundle: Path | None = None):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    if bundle is not None:
        monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)


def test_source_checkout_defaults_match_previous_behaviour():
    assert settings.data_dir() == Path("data")
    assert settings.project_files() is None
    assert settings.snapshot_dir() is None
    assert settings.asof_archive() is None


def test_project_files_mount_gives_snapshot_and_archive(monkeypatch, tmp_path):
    pf = tmp_path / "pf"
    (pf / "cfbd-data" / "asof").mkdir(parents=True)
    monkeypatch.setenv("CFB_PROJECT_FILES", str(pf))
    assert settings.snapshot_dir() == pf / "cfbd-data"
    assert settings.asof_archive() == pf / "cfbd-data" / "asof"


def test_archive_is_none_when_folder_missing(monkeypatch, tmp_path):
    pf = tmp_path / "pf"
    (pf / "cfbd-data").mkdir(parents=True)
    monkeypatch.setenv("CFB_PROJECT_FILES", str(pf))
    assert settings.asof_archive() is None


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("CFB_SNAPSHOT_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("CFB_ASOF_ARCHIVE", str(tmp_path / "a"))
    monkeypatch.setenv("CFB_CONFIG", str(tmp_path / "c.toml"))
    assert settings.data_dir() == tmp_path / "d"
    assert settings.snapshot_dir() == tmp_path / "s"
    assert settings.asof_archive() == tmp_path / "a"
    assert settings.config_path() == tmp_path / "c.toml"


def test_frozen_app_uses_user_folders(monkeypatch, tmp_path):
    freeze(monkeypatch)
    monkeypatch.setattr(settings, "user_data_root", lambda: tmp_path / "local" / "cfb")
    assert settings.data_dir() == tmp_path / "local" / "cfb" / "data"
    assert settings.warehouse_dir() == tmp_path / "local" / "cfb" / "warehouse"
    assert settings.asof_archive() == tmp_path / "local" / "cfb" / "asof"  # the engine's dated copies


def test_windows_folders(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert settings.user_data_root() == tmp_path / "Local" / "cfb"
    assert settings.user_config_root() == tmp_path / "Roaming" / "cfb"


def test_config_path_order(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "bundle"
    (bundle / "dashboard").mkdir(parents=True)
    (bundle / "dashboard" / "config.toml").write_text("", encoding="utf-8")
    user = tmp_path / "user"
    monkeypatch.setattr(settings, "user_config_root", lambda: user)

    # Not frozen, no repo file: the repo path (load_config then uses defaults).
    assert settings.config_path() == settings.REPO_CONFIG
    # Frozen: the bundled copy, then the repo file, then the user's file win in turn.
    freeze(monkeypatch, bundle)
    assert settings.config_path() == bundle / "dashboard" / "config.toml"
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "dashboard" / "config.toml").write_text("", encoding="utf-8")
    assert settings.config_path() == settings.REPO_CONFIG
    user.mkdir()
    (user / "config.toml").write_text("", encoding="utf-8")
    assert settings.config_path() == user / "config.toml"


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, name):
        return self.store.get((service, name))

    def set_password(self, service, name, value):
        self.store[(service, name)] = value

    def delete_password(self, service, name):
        self.store.pop((service, name))


def test_secret_env_then_keyring(monkeypatch):
    kr = FakeKeyring()
    monkeypatch.setattr(settings, "_keyring", lambda: kr)
    assert settings.secret("cfbd_api_key") is None
    settings.set_secret("cfbd_api_key", "from-keyring")
    assert settings.secret("cfbd_api_key") == "from-keyring"
    monkeypatch.setenv("CFBD_API_KEY", "from-env")
    assert settings.secret("cfbd_api_key") == "from-env"
    monkeypatch.delenv("CFBD_API_KEY")
    settings.delete_secret("cfbd_api_key")
    assert settings.secret("cfbd_api_key") is None


def test_secret_survives_broken_keyring(monkeypatch):
    broken = types.SimpleNamespace(get_password=lambda *a: (_ for _ in ()).throw(RuntimeError("no backend")))
    monkeypatch.setattr(settings, "_keyring", lambda: broken)
    assert settings.secret("r2_bucket") is None


def test_set_secret_without_keyring_uses_the_secrets_file():
    assert settings.set_secret("r2_access_key_id", "plain-key-value") == "file"
    assert settings.secret("r2_access_key_id") == "plain-key-value"
    assert settings.secret_source("r2_access_key_id") == "file"
    raw = settings.secrets_file().read_text(encoding="utf-8")
    if sys.platform == "win32":
        assert "plain-key-value" not in raw  # DPAPI-encrypted for this Windows user
    settings.delete_secret("r2_access_key_id")
    assert settings.secret("r2_access_key_id") is None
    with pytest.raises(KeyError):
        settings.set_secret("not_a_secret", "x")


class ForgetfulKeyring(FakeKeyring):
    """Accepts a value but does not keep it (a backend that cannot persist)."""

    def set_password(self, service, name, value):
        pass


def test_set_secret_falls_back_when_keyring_does_not_keep_it(monkeypatch):
    monkeypatch.setattr(settings, "_keyring", lambda: ForgetfulKeyring())
    assert settings.set_secret("r2_secret_access_key", "s3cr3t") == "file"
    assert settings.secret("r2_secret_access_key") == "s3cr3t"


def test_set_secret_in_keyring_removes_the_file_copy(monkeypatch):
    settings.set_secret("r2_bucket", "old")            # no keyring yet: file
    kr = FakeKeyring()
    monkeypatch.setattr(settings, "_keyring", lambda: kr)
    assert settings.set_secret("r2_bucket", "new") == "keyring"
    assert settings.secret_source("r2_bucket") == "keyring"
    assert "r2_bucket" not in settings.secrets_file().read_text(encoding="utf-8")
    assert settings.secret("r2_bucket") == "new"


def test_experimental_keeps_its_own_files_and_reads_only_the_cfbd_key_from_main(monkeypatch, tmp_path):
    """Experimental: secrets file, app prefs and player data sit under cfb-exp; from main's secrets
    file it reads the CFBD key only, never R2 keys, and never writes there."""
    import json

    from cfb.ui import availability, connection

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    for v in ("CFB_SECRETS_FILE", "CFB_APP_PREFS", "CFB_AVAILABILITY_DIR", "CFBD_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(settings, "APP", "cfb-exp")
    monkeypatch.setattr(settings, "KEYRING_SERVICE", "cfb-exp")
    main_file = tmp_path / "cfg" / "cfb" / "secrets.json"
    main_file.parent.mkdir(parents=True)
    main_file.write_text(json.dumps({"cfbd_api_key": {"plain": "MAIN-CFBD"},
                                     "r2_access_key_id": {"plain": "MAIN-R2"}}))
    assert settings.secrets_file() == tmp_path / "cfg" / "cfb-exp" / "secrets.json"
    assert connection.prefs_path() == tmp_path / "cfg" / "cfb-exp" / "app.json"
    assert availability.local_dir() == tmp_path / "data" / "cfb-exp" / "availability"
    assert settings.main_secrets_file() == main_file
    assert settings.secret("cfbd_api_key") == "MAIN-CFBD"
    assert settings.secret_source("cfbd_api_key") == "main file"
    assert settings.secret("r2_access_key_id") is None
    assert settings.set_secret("r2_access_key_id", "EXP-R2") == "file"
    assert settings.secret("r2_access_key_id") == "EXP-R2"
    assert json.loads(main_file.read_text())["r2_access_key_id"] == {"plain": "MAIN-R2"}  # main untouched
    settings.set_secret("cfbd_api_key", "EXP-CFBD")
    assert settings.secret("cfbd_api_key") == "EXP-CFBD" and settings.secret_source("cfbd_api_key") == "file"


def test_keyring_backend_names(monkeypatch):
    assert settings.keyring_backend() is None
    fail = type("Keyring", (), {"__module__": "keyring.backends.fail"})()
    monkeypatch.setattr(settings, "_keyring", lambda: types.SimpleNamespace(get_keyring=lambda: fail))
    assert settings.keyring_backend() is None
    win = type("WinVaultKeyring", (), {"__module__": "keyring.backends.Windows"})()
    monkeypatch.setattr(settings, "_keyring", lambda: types.SimpleNamespace(get_keyring=lambda: win))
    assert settings.keyring_backend() == "Windows.WinVaultKeyring"


def test_client_key_resolution(monkeypatch):
    assert CFBDClient().api_key is None  # cloud: the proxy injects the key
    monkeypatch.setenv("CFBD_API_KEY", "k1")
    assert CFBDClient().api_key == "k1"
    assert CFBDClient(api_key="explicit").api_key == "explicit"


def test_run_week_snapshot_defaults(monkeypatch, tmp_path):
    from cfb import run_week

    assert run_week.default_snapshot() is None
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "a.tar").write_bytes(b"")
    (snap / "f.tar").write_bytes(b"")
    (snap / "LATEST").write_text("a.tar\n", encoding="utf-8")
    (snap / "FEATURES_LATEST").write_text("f.tar", encoding="utf-8")
    monkeypatch.setenv("CFB_SNAPSHOT_DIR", str(snap))
    assert run_week.default_snapshot() == snap / "a.tar"
    assert run_week.default_features_snapshot() == snap / "f.tar"

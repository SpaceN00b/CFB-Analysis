"""The PC's Cloudflare R2 connection: save the read-only key pair once, check it, and the app
preferences (sync at launch). No Streamlit here, so it is testable without the UI.

The keys go through `cfb.settings.set_secret` (Windows Credential Manager under the edition's
service, else a DPAPI-encrypted file in the edition's %APPDATA% folder), so they are there on every
later launch and after installing a new build. Preferences: app.json in the same folder
(%APPDATA%\\cfb for main, %APPDATA%\\cfb-exp for experimental). The bucket comes from
`warehouse_sync.r2_settings()`, which gives the edition's bucket and refuses main's in experimental.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .. import settings

R2_KEYS = ("r2_account_id", "r2_access_key_id", "r2_secret_access_key", "r2_bucket")
REQUIRED = ("r2_account_id", "r2_access_key_id", "r2_secret_access_key")
DEFAULT_PREFS = {"auto_sync": True}
_ACCOUNT_RE = re.compile(r"\b([0-9a-f]{32})\b", re.I)


def parse_account_id(text: str) -> str | None:
    """The 32-character account ID, from the ID itself or the S3 endpoint Cloudflare shows with the
    token (https://<account id>.r2.cloudflarestorage.com)."""
    m = _ACCOUNT_RE.search((text or "").strip())
    return m.group(1).lower() if m else None


@dataclass
class Status:
    missing: list[str]
    sources: dict[str, str | None]  # key -> environment | keyring | file | None

    @property
    def configured(self) -> bool:
        return not self.missing

    def where(self) -> str:
        places = {s for k, s in self.sources.items() if k in REQUIRED and s}
        names = {"keyring": "Windows Credential Manager", "file": "the encrypted keys file",
                 "environment": "environment variables"}
        return ", ".join(sorted(names.get(p, p) for p in places)) or "nowhere yet"


def status() -> Status:
    src = {k: settings.secret_source(k) for k in R2_KEYS}
    return Status([k for k in REQUIRED if not src[k]], src)


def save(account: str, access_key_id: str, secret_access_key: str, bucket: str | None = None) -> dict[str, str]:
    """Save the R2 connection (blank fields keep the saved value). Returns key -> where it went."""
    out = {}
    if account and account.strip():
        acc = parse_account_id(account)
        if acc is None:
            raise ValueError("Account ID: paste the 32-character ID or the S3 endpoint "
                             "(https://<account id>.r2.cloudflarestorage.com)")
        out["r2_account_id"] = settings.set_secret("r2_account_id", acc)
    if access_key_id and access_key_id.strip():
        out["r2_access_key_id"] = settings.set_secret("r2_access_key_id", access_key_id.strip())
    if secret_access_key and secret_access_key.strip():
        out["r2_secret_access_key"] = settings.set_secret("r2_secret_access_key", secret_access_key.strip())
    if bucket and bucket.strip():
        out["r2_bucket"] = settings.set_secret("r2_bucket", bucket.strip())
    return out


def forget() -> None:
    for k in R2_KEYS:
        settings.delete_secret(k)


def check() -> str:
    """Read the bucket's current manifest with the saved keys. Returns the database version;
    raises with a readable message when the connection does not work."""
    from .. import warehouse_sync as W

    st = status()
    if not st.configured:
        raise RuntimeError("missing " + ", ".join(st.missing))
    store = W.open_store(W.r2_settings())
    try:
        raw = store.get(W.MANIFEST)
    except Exception as exc:
        raise RuntimeError(f"could not read the bucket: {exc}") from exc
    if raw is None:
        raise RuntimeError("connected, but the bucket has no database yet")
    return json.loads(raw).get("version", "?")


# ---- app preferences ---------------------------------------------------------------------------

def prefs_path() -> Path:
    env = os.environ.get("CFB_APP_PREFS")
    return Path(env) if env else settings.user_config_root() / "app.json"


def prefs() -> dict:
    try:
        d = json.loads(prefs_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    return {**DEFAULT_PREFS, **(d if isinstance(d, dict) else {})}


def set_pref(name: str, value) -> None:
    d = prefs()
    d[name] = value
    p = prefs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)

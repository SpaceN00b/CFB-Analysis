"""Minimal CollegeFootballData.com API client.

The key comes from cfb.settings.secret("cfbd_api_key"): the CFBD_API_KEY environment
variable, then the OS keyring (packaged app). If neither is set the request is sent
without an Authorization header, which supports environments where a proxy injects
the credential.
"""
from __future__ import annotations

import time
from typing import Any

import requests

from . import settings

BASE_URL = "https://api.collegefootballdata.com"
RETRY_STATUS = {429, 500, 502, 503, 504}


class CFBDError(RuntimeError):
    pass


class CFBDClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        session: requests.Session | None = None,
        max_retries: int = 5,
        backoff: float = 2.0,
        timeout: float = 120.0,
        sleep=time.sleep,
    ):
        self.api_key = api_key if api_key is not None else settings.secret("cfbd_api_key")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout
        self._sleep = sleep
        self.calls = 0

    def get(self, path: str, **params: Any) -> Any:
        """GET `path` and return parsed JSON. None-valued params are dropped."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        query = {k: v for k, v in params.items() if v is not None}
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=query, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                last = repr(exc)
            else:
                self.calls += 1
                if resp.status_code == 200:
                    return resp.json()
                last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in RETRY_STATUS:
                    raise CFBDError(f"GET {path} {query} failed. {last}")
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit() and attempt < self.max_retries:
                    self._sleep(float(retry_after))
                    continue
            if attempt < self.max_retries:
                self._sleep(self.backoff * 2**attempt)
        raise CFBDError(f"GET {path} {query} failed after {self.max_retries + 1} attempts. {last}")

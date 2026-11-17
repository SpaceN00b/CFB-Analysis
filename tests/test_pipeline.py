"""Unit tests for client retry logic, storage, and task planning. No network."""
import pytest

from cfb.client import CFBDClient, CFBDError
from cfb.ingest import BY_NAME, run, tasks
from cfb.store import Store, flatten


class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.requests.append((url, params, headers))
        return self.responses.pop(0)


def client(responses, **kw):
    sleeps = []
    c = CFBDClient(api_key="k", session=FakeSession(responses), sleep=sleeps.append, **kw)
    return c, sleeps


def test_get_sends_bearer_and_drops_none_params():
    c, _ = client([FakeResp(200, [{"id": 1}])])
    assert c.get("/games", year=2025, week=None) == [{"id": 1}]
    url, params, headers = c.session.requests[0]
    assert url.endswith("/games") and params == {"year": 2025}
    assert headers["Authorization"] == "Bearer k"


def test_retries_then_succeeds():
    c, sleeps = client([FakeResp(503), FakeResp(429, headers={"Retry-After": "7"}), FakeResp(200, [])])
    assert c.get("/games", year=2025) == []
    assert sleeps == [2.0, 7.0]


def test_non_retryable_raises_immediately():
    c, sleeps = client([FakeResp(401, "bad key")])
    with pytest.raises(CFBDError, match="401"):
        c.get("/games", year=2025)
    assert sleeps == []


def test_gives_up_after_max_retries():
    c, _ = client([FakeResp(500)] * 3, max_retries=2)
    with pytest.raises(CFBDError, match="3 attempts"):
        c.get("/games", year=2025)


def test_flatten_nested_and_lists():
    df = flatten([{"id": 1, "venue": {"name": "X"}, "lines": [{"spread": -3}]}], {"season": 2025})
    assert df.loc[0, "venue.name"] == "X"
    assert df.loc[0, "lines"] == '[{"spread": -3}]'
    assert df.loc[0, "season"] == 2025


def test_store_roundtrip(tmp_path):
    s = Store(tmp_path)
    assert s.write("games", "2025_regular", [{"id": 1}, {"id": 2}], {"season": 2025}) == 2
    assert s.has("games", "2025_regular")
    assert s.read_raw("games", "2025_regular") == [{"id": 1}, {"id": 2}]
    assert list(s.load("games")["id"]) == [1, 2]


def test_week_tasks():
    keys = [k for k, _, _ in tasks(BY_NAME["plays"], 2025, {"regular": [1, 2], "postseason": [1]})]
    assert keys == ["2025_regular_01", "2025_regular_02", "2025_postseason_01"]


def test_run_skips_cached_and_records_failures(tmp_path):
    s = Store(tmp_path)
    c, _ = client([FakeResp(200, [{"team": "A"}]), FakeResp(404, "nope")])
    failures = run(c, s, [2024, 2025], ["coaches"])
    assert len(failures) == 1 and "coaches/2025" in failures[0]
    c2, _ = client([FakeResp(200, [{"team": "B"}])])
    assert run(c2, s, [2024, 2025], ["coaches"]) == []
    assert len(c2.session.requests) == 1  # 2024 was cached

"""
Live integration tests for the Aivery Fabric API and Cortex agent.

Requirements
------------
Set these environment variables before running:

    AIVERY_API_KEY   — Bearer token for the Fabric API (required)
    AIVERY_BASE_URL  — Fabric API base URL  (default: http://localhost:5129)
    AIVERY_CORTEX_URL — Cortex agent base URL (default: http://localhost:5130)

Example against a self-hosted deployment:

    AIVERY_API_KEY=aiv_... \\
    AIVERY_BASE_URL=https://api.example.com \\
    AIVERY_CORTEX_URL=https://cortex.example.com \\
    pytest tests/integration/ -v

The suite creates a unique agent ID per run and cleans up via delete_user
where possible, so repeated runs do not accumulate test data.
"""

import os
import sys
import uuid
import time
import requests
import pytest
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Config — all from environment, no hardcoded credentials
# ---------------------------------------------------------------------------

API_URL    = os.environ.get("AIVERY_BASE_URL",   "http://localhost:5129")
CORTEX_URL = os.environ.get("AIVERY_CORTEX_URL", "http://localhost:5130")
API_KEY    = os.environ.get("AIVERY_API_KEY",    "")

if not API_KEY:
    pytest.skip(
        "AIVERY_API_KEY is not set — skipping live integration tests.",
        allow_module_level=True,
    )

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

AGENT_ID = f"sdk-integration-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def api(path: str, payload: dict) -> requests.Response:
    return requests.post(f"{API_URL}{path}", json=payload, headers=HEADERS, timeout=30)

def cortex_chat(payload: dict) -> requests.Response:
    return requests.post(
        f"{CORTEX_URL}/api/agent/chat", json=payload, headers=HEADERS, timeout=60
    )

def write(content: str, type_: str = "semantic") -> requests.Response:
    return api("/memory/write", {
        "agent_id": AGENT_ID,
        "content": content,
        "type": type_,
        "timestamp": now_iso(),
    })

def seed_and_wait(content: str, wait: float = 3.0) -> None:
    r = write(content)
    assert r.status_code == 200, f"Seed write failed: {r.text}"
    time.sleep(wait)


# ---------------------------------------------------------------------------
# Fabric — Write
# ---------------------------------------------------------------------------

class TestFabricWrite:
    def test_write_returns_200(self):
        r = write("Integration test: the sky is blue.")
        assert r.status_code == 200, r.text

    def test_write_returns_id(self):
        r = write("Integration test: Alice loves hiking.", type_="episodic")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data.get("success") is True, data
        assert data.get("id") is not None, data


# ---------------------------------------------------------------------------
# Fabric — Retrieve
# ---------------------------------------------------------------------------

class TestFabricRetrieve:
    def setup_method(self):
        seed_and_wait("Integration test: Bob works at a tech startup in Austin.")

    def test_retrieve_returns_results(self):
        r = api("/memory/retrieve", {
            "agent_id": AGENT_ID,
            "query": "where does Bob work?",
            "limit": 5,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert "memories" in data, data
        assert len(data["memories"]) > 0

    def test_retrieve_top_result_relevant(self):
        r = api("/memory/retrieve", {
            "agent_id": AGENT_ID,
            "query": "Bob's employer",
            "limit": 5,
        })
        memories = r.json().get("memories", [])
        top = memories[0]["content"] if memories else ""
        assert any(kw in top for kw in ("Bob", "startup", "Austin")), \
            f"Top result not relevant: {top!r}"


# ---------------------------------------------------------------------------
# Fabric — Context
# ---------------------------------------------------------------------------

class TestFabricContext:
    def setup_method(self):
        seed_and_wait("Integration test: Carol is a software engineer in Seattle.")

    def test_context_returns_200(self):
        r = api("/memory/context", {
            "agent_id": AGENT_ID,
            "task": "tell me about Carol",
            "limit": 5,
        })
        assert r.status_code == 200, r.text

    def test_context_has_content(self):
        r = api("/memory/context", {
            "agent_id": AGENT_ID,
            "task": "Carol's job",
            "limit": 5,
        })
        data = r.json()
        assert data.get("processed") is True, data
        assert data.get("context") is not None, data


# ---------------------------------------------------------------------------
# Fabric — Extract
# ---------------------------------------------------------------------------

class TestFabricExtract:
    def test_extract_returns_memories(self):
        r = api("/memory/extract", {
            "agent_id": AGENT_ID,
            "text": "David just got a promotion to senior engineer at his company.",
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert "memories" in data, data
        assert len(data["memories"]) > 0, f"Expected extracted memories: {data}"


# ---------------------------------------------------------------------------
# Fabric — Validate
# ---------------------------------------------------------------------------

class TestFabricValidate:
    def setup_method(self):
        seed_and_wait("Integration test: Eve lives in New York.")

    def test_validate_detects_conflict(self):
        r = api("/memory/validate", {
            "agent_id": AGENT_ID,
            "new_information": "Eve moved to Los Angeles.",
            "auto_resolve": False,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert "valid" in data and "action" in data, data


# ---------------------------------------------------------------------------
# Cortex — Chat
# ---------------------------------------------------------------------------

class TestCortexChat:
    def setup_method(self):
        seed_and_wait("Integration test: Frank's favorite color is green.", wait=8)

    def test_chat_returns_200(self):
        r = cortex_chat({
            "message": "Hello, can you hear me?",
            "agent_id": AGENT_ID,
            "stream": False,
        })
        assert r.status_code == 200, r.text

    def test_chat_returns_text(self):
        r = cortex_chat({
            "message": "Say 'pong' if you can read this.",
            "agent_id": AGENT_ID,
            "stream": False,
        })
        data = r.json()
        assert any(k in data for k in ("response", "content", "text", "message")), \
            f"No text field in response: {data}"

    def test_chat_recalls_memory(self):
        r = cortex_chat({
            "message": "What is Frank's favorite color?",
            "agent_id": AGENT_ID,
            "stream": False,
        })
        assert r.status_code == 200, r.text
        assert "green" in r.text.lower(), \
            f"Memory not recalled — response: {r.text[:400]}"


# ---------------------------------------------------------------------------
# SDK Memory client (sync wrapper)
# ---------------------------------------------------------------------------

class TestSDKClient:
    def test_add_and_search(self):
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
        from aivery import Memory

        agent = f"sdk-client-{uuid.uuid4().hex[:8]}"
        m = Memory(host=API_URL, api_key=API_KEY)
        try:
            result = m.add("Grace is a marathon runner from Boston.", user_id=agent)
            assert len(result.get("results", [])) > 0

            time.sleep(3)

            hits = m.search("Grace's sport", user_id=agent)
            assert any(
                "marathon" in h["memory"].lower() or "runner" in h["memory"].lower()
                for h in hits
            ), f"Expected marathon memory, got: {hits}"
        finally:
            m.delete_user(agent)
            m.close()

"""Unit tests for AsyncMemory — mock HTTP responses, no live server needed."""

import pytest
import aiohttp
from unittest.mock import AsyncMock, patch, MagicMock

from aivery import AsyncMemory


@pytest.fixture
def client():
    return AsyncMemory(host="http://localhost:5131", api_key="test-key")


# ------------------------------------------------------------------
# add()
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_string(client):
    extract_resp = {"memories": [{"content": "I love hiking", "type": "semantic", "confidence": 0.9}]}
    write_resp = {}

    with patch.object(client, "_post", new=AsyncMock(side_effect=[extract_resp, write_resp])):
        result = await client.add("I love hiking", user_id="alice")

    assert result["results"][0]["memory"] == "I love hiking"
    assert result["results"][0]["event"] == "ADD"


@pytest.mark.asyncio
async def test_add_empty_string(client):
    result = await client.add("   ", user_id="alice")
    assert result == {"results": []}


@pytest.mark.asyncio
async def test_add_conversation(client):
    extract_resp = {"memories": [{"content": "Bob lives in NYC", "type": "semantic", "confidence": 0.85}]}
    write_resp = {}

    with patch.object(client, "_post", new=AsyncMock(side_effect=[extract_resp, write_resp])):
        result = await client.add(
            [
                {"role": "user", "content": "I just moved to NYC."},
                {"role": "assistant", "content": "Nice!"},
            ],
            user_id="bob",
        )

    assert len(result["results"]) == 1
    assert "NYC" in result["results"][0]["memory"]


# ------------------------------------------------------------------
# search()
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_returns_list(client):
    context_resp = {
        "context": [
            {"content": "I love hiking", "relevance": 0.92, "id": "abc"},
            {"content": "I live in SF", "relevance": 0.7, "id": "def"},
        ]
    }

    with patch.object(client, "_post", new=AsyncMock(return_value=context_resp)):
        results = await client.search("outdoor activities", user_id="alice")

    assert len(results) == 2
    assert results[0]["memory"] == "I love hiking"
    assert results[0]["score"] == 0.92
    assert results[0]["id"] == "abc"


@pytest.mark.asyncio
async def test_search_empty_context(client):
    with patch.object(client, "_post", new=AsyncMock(return_value={"context": []})):
        results = await client.search("nothing", user_id="alice")
    assert results == []


# ------------------------------------------------------------------
# context()
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_formats_bullet_list(client):
    mock_search = AsyncMock(return_value=[
        {"memory": "I love hiking", "score": 0.9, "id": "a"},
        {"memory": "I live in SF", "score": 0.7, "id": "b"},
    ])
    with patch.object(client, "search", mock_search):
        ctx = await client.context("tell me about alice", user_id="alice")

    assert ctx == "- I love hiking\n- I live in SF"


@pytest.mark.asyncio
async def test_context_empty_when_no_memories(client):
    with patch.object(client, "search", AsyncMock(return_value=[])):
        ctx = await client.context("anything", user_id="alice")
    assert ctx == ""


# ------------------------------------------------------------------
# add() with timestamp conversion
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_converts_unix_timestamp(client):
    calls = []

    async def capture_post(path, payload):
        calls.append((path, payload))
        if path == "/memory/extract":
            return {"memories": [{"content": "test", "type": "semantic", "confidence": 0.8}]}
        return {}

    with patch.object(client, "_post", side_effect=capture_post):
        await client.add("test", user_id="alice", timestamp=1_000_000)

    extract_call = next(p for (path, p) in calls if path == "/memory/extract")
    assert "session_date" in extract_call
    assert "1970" in extract_call["session_date"] or "2001" in extract_call["session_date"]

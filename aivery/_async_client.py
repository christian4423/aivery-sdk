"""
Async HTTP client for the Aivery memory API.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import aiohttp
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:5131"
_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000000"


def _epoch_to_iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _messages_to_text(messages: list[dict]) -> str:
    return "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in messages
        if m.get("content", "").strip()
    )


class AsyncMemory:
    """Async Aivery memory client.

    Usage::

        async with AsyncMemory() as m:
            await m.add("I love hiking", user_id="alice")
            results = await m.search("outdoor activities", user_id="alice")
    """

    def __init__(
        self,
        host: str | None = None,
        api_key: str | None = None,
        org_id: str | None = None,
        max_retries: int = 5,
        retry_delay: float = 3.0,
        rpm: int = 100_000,
        timeout: float = 120.0,
    ):
        self.host = (host or os.getenv("AIVERY_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.getenv("AIVERY_API_KEY", "")
        self.org_id = org_id or os.getenv("AIVERY_ORG_ID", _DEFAULT_ORG_ID)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.limiter = AsyncLimiter(rpm, 60)
        self._session: aiohttp.ClientSession | None = None

    @property
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.org_id and self.org_id != "00000000-0000-0000-0000-000000000000":
            h["X-Org-Id"] = self.org_id
        return h

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=self.timeout,
                connector=aiohttp.TCPConnector(limit=0),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> AsyncMemory:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def _post(self, path: str, payload: dict) -> Any:
        session = await self._get_session()
        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(f"{self.host}{path}", json=payload) as resp:
                        if resp.status >= 500:
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history, status=resp.status
                            )
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                logger.warning("%s attempt %d/%d failed: %s", path, attempt + 1, self.max_retries, str(exc)[:200])
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def add(
        self,
        messages: list[dict[str, str]] | str,
        user_id: str,
        timestamp: int | None = None,
        session_date: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Extract memories from messages and write each to Aivery.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts, or a plain string.
            user_id:  Agent / user identifier (scopes memory storage).
            timestamp: Unix epoch (seconds) of when the conversation occurred.
            session_date: ISO-8601 date string (alternative to timestamp).
            metadata: Reserved for future use.

        Returns:
            ``{"results": [{"memory": "...", "event": "ADD"}, ...]}``
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        text = _messages_to_text(messages)
        if not text.strip():
            return {"results": []}

        obs_date: str | None = session_date
        if timestamp is not None:
            obs_date = _epoch_to_iso(timestamp)

        # Step 1 — extract structured memories from raw text
        extract_payload: dict[str, Any] = {
            "agent_id": user_id,
            "text": text,
            "source": "sdk",
            "max_memories": 20,
        }
        if obs_date:
            extract_payload["session_date"] = obs_date

        data = await self._post("/memory/extract", extract_payload)
        memories = data.get("memories", [])

        # Step 2 — write each extracted memory
        write_tasks = [
            self._write_one(user_id, mem, obs_date)
            for mem in memories
            if mem.get("content", "").strip()
        ]
        write_results = await asyncio.gather(*write_tasks, return_exceptions=True)

        results = []
        for mem, ok in zip(memories, write_results):
            if ok is True:
                results.append({"memory": mem.get("content", ""), "event": "ADD"})

        return {"results": results}

    async def _write_one(self, user_id: str, mem: dict, timestamp: str | None) -> bool:
        payload: dict[str, Any] = {
            "agent_id": user_id,
            "content": mem.get("content", ""),
            "type": mem.get("type", "semantic"),
            "confidence": mem.get("confidence", 0.7),
            "timestamp": timestamp or datetime.now(tz=timezone.utc).isoformat(),
        }
        if mem.get("entities"):
            payload["entities"] = mem["entities"]
        if mem.get("facts"):
            payload["facts"] = mem["facts"]

        try:
            await self._post("/memory/write", payload)
            return True
        except Exception as exc:
            logger.warning("write failed for user=%s: %s", user_id, str(exc)[:200])
            return False

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Retrieve memories relevant to a query.

        Returns a list of ``{"memory": str, "score": float, "id": str}`` dicts,
        ordered by relevance (highest first).
        """
        payload: dict[str, Any] = {
            "agent_id": user_id,
            "agent_ids": [user_id],
            "task": query,
            "limit": top_k,
        }
        data = await self._post("/memory/context", payload)
        items = data.get("context") or []
        results = []
        for m in items:
            if not m.get("content"):
                continue
            entry: dict[str, Any] = {
                "memory": m.get("content", ""),
                "score": round(float(m.get("relevance", 0.0)), 4),
                "id": str(m.get("id", "")),
            }
            if m.get("created_at"):
                entry["created_at"] = m["created_at"]
            results.append(entry)
        return results

    async def context(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
    ) -> str:
        """Retrieve memories and return a formatted string ready for LLM injection.

        Returns a newline-separated block like::

            - I love hiking in the mountains
            - Alice lives in San Francisco
        """
        memories = await self.search(query, user_id=user_id, top_k=top_k)
        if not memories:
            return ""
        return "\n".join(f"- {m['memory']}" for m in memories)

    async def get_all(self, user_id: str) -> list[dict]:
        """Return all stored memories for a user (unordered)."""
        session = await self._get_session()
        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.get(
                        f"{self.host}/api/memories",
                        headers={"X-Agent-Id": user_id},
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                logger.warning("get_all attempt %d/%d failed: %s", attempt + 1, self.max_retries, str(exc)[:200])
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
        return []

    async def delete_user(self, user_id: str) -> bool:
        """Delete all memories for a user. Returns True if all deletions succeeded."""
        session = await self._get_session()
        try:
            async with self.limiter:
                async with session.get(
                    f"{self.host}/api/memories",
                    headers={"X-Agent-Id": user_id},
                ) as resp:
                    resp.raise_for_status()
                    memories = await resp.json()

            if not memories:
                return True

            tasks = [self._delete_one(session, m["id"]) for m in memories if m.get("id")]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failed = sum(1 for r in results if isinstance(r, Exception))
            if failed:
                logger.warning("delete_user %s: %d/%d failed", user_id, failed, len(tasks))
            return failed == 0

        except Exception as exc:
            logger.warning("delete_user failed for %s: %s", user_id, exc)
            return False

    async def _delete_one(self, session: aiohttp.ClientSession, memory_id: str) -> None:
        async with self.limiter:
            async with session.delete(f"{self.host}/api/memories/{memory_id}") as resp:
                resp.raise_for_status()

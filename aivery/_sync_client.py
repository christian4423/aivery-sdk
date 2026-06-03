"""
Synchronous wrapper around AsyncMemory — runs a dedicated background event loop.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from typing import Any

from ._async_client import AsyncMemory


class Memory:
    """Synchronous Aivery memory client.

    Usage::

        from aivery import Memory

        m = Memory()
        m.add("I love hiking on weekends", user_id="alice")
        results = m.search("outdoor activities", user_id="alice")

    Configuration (env vars or constructor args):

        AIVERY_BASE_URL   — default: http://localhost:5131
        AIVERY_API_KEY    — Bearer token (required for auth-enabled deployments)
        AIVERY_ORG_ID     — Org header (default: 00000000-0000-0000-0000-000000000000)
    """

    def __init__(
        self,
        host: str | None = None,
        api_key: str | None = None,
        org_id: str | None = None,
        timeout: float = 120.0,
    ):
        self._async = AsyncMemory(host=host, api_key=api_key, org_id=org_id, timeout=timeout)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def _run(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        """Release the underlying HTTP session and stop the background loop."""
        if self._loop.is_running():
            self._run(self._async.close())
            self._loop.call_soon_threadsafe(self._loop.stop)

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def add(
        self,
        messages: list[dict[str, str]] | str,
        user_id: str,
        timestamp: int | None = None,
        session_date: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Extract memories from messages and write them.

        Args:
            messages: Conversation as a list of ``{"role", "content"}`` dicts, or a plain string.
            user_id:  Agent / user identifier.
            timestamp: Unix epoch (seconds) of the conversation.
            session_date: ISO-8601 date string (alternative to timestamp).

        Returns:
            ``{"results": [{"memory": "...", "event": "ADD"}, ...]}``
        """
        return self._run(
            self._async.add(
                messages,
                user_id=user_id,
                timestamp=timestamp,
                session_date=session_date,
                metadata=metadata,
            )
        )

    def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Retrieve memories relevant to *query*.

        Returns:
            List of ``{"memory": str, "score": float, "id": str}`` dicts,
            ordered by relevance (highest first).
        """
        return self._run(self._async.search(query, user_id=user_id, top_k=top_k))

    def context(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
    ) -> str:
        """Retrieve memories and return a formatted bullet list for LLM injection.

        Example output::

            - I love hiking in the mountains
            - Alice lives in San Francisco
        """
        return self._run(self._async.context(query, user_id=user_id, top_k=top_k))

    def get_all(self, user_id: str) -> list[dict]:
        """Return all stored memories for a user (unordered)."""
        return self._run(self._async.get_all(user_id))

    def delete_user(self, user_id: str) -> bool:
        """Delete all memories for a user. Returns True if successful."""
        return self._run(self._async.delete_user(user_id))

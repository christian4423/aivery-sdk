import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

_LOCOMO_TS_FMT = "%I:%M %p on %d %B, %Y"


def _to_iso8601(ts: str) -> str:
    if not ts:
        return datetime.now(timezone.utc).isoformat()
    try:
        return datetime.strptime(ts, _LOCOMO_TS_FMT).replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


load_dotenv()

_BASE_URL = os.getenv("AIVERY_BASE_URL", "https://api.aivery.systems")
_API_KEY  = os.getenv("AIVERY_API_KEY", "")
_ORG_ID   = os.getenv("AIVERY_ORG_ID", "00000000-0000-0000-0000-000000000000")


class AiveryClient:
    """Thin HTTP wrapper around the Aivery API for LOCOMO benchmarking."""

    def __init__(self, base_url: str = _BASE_URL, api_key: str = _API_KEY, org_id: str = _ORG_ID):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            **({"X-Org-Id": org_id} if org_id else {}),
        })

    def add(self, messages: list[dict], user_id: str, metadata: dict | None = None, retries: int = 3) -> None:
        text = self._messages_to_text(messages)
        timestamp = (metadata or {}).get("timestamp", "")

        for attempt in range(retries):
            try:
                extracted = self._extract(user_id, text, session_date=timestamp)
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise e

        for mem in extracted:
            self._write(
                agent_id=user_id,
                content=mem["content"],
                type_=mem.get("type", "semantic"),
                confidence=mem.get("confidence", 0.7),
                timestamp=timestamp,
                entities=mem.get("entities"),
                facts=mem.get("facts"),
            )

    def _extract(self, agent_id: str, text: str, session_date: str = "") -> list[dict]:
        payload: dict = {"agent_id": agent_id, "text": text, "source": "locomo", "max_memories": 20}
        if session_date:
            payload["session_date"] = session_date
        resp = self.session.post(f"{self.base_url}/memory/extract", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("memories", [])

    def _write(self, agent_id: str, content: str, type_: str, confidence: float, timestamp: str,
               entities: list | None = None, facts: list | None = None) -> None:
        payload = {
            "agent_id": agent_id,
            "content": content,
            "type": type_,
            "confidence": confidence,
            "timestamp": _to_iso8601(timestamp),
        }
        if entities:
            payload["entities"] = entities
        if facts:
            payload["facts"] = facts
        resp = self.session.post(f"{self.base_url}/memory/write", json=payload, timeout=120)
        resp.raise_for_status()

    def search(self, query: str, user_id: str, top_k: int = 10, reference_time: str | None = None) -> list[dict]:
        payload: dict = {"agent_id": user_id, "query": query, "limit": top_k}
        if reference_time:
            payload["reference_time"] = _to_iso8601(reference_time)
        resp = self.session.post(f"{self.base_url}/memory/retrieve", json=payload, timeout=30)
        resp.raise_for_status()
        return [
            {
                "memory": m.get("content", ""),
                "score": round(m.get("relevance", 0.0), 2),
                "metadata": {"timestamp": m.get("created_at", "")},
            }
            for m in resp.json().get("memories", [])
        ]

    def delete_all(self, user_id: str) -> None:
        resp = self.session.get(f"{self.base_url}/api/memories", headers={"X-Agent-Id": user_id}, timeout=30)
        if not resp.ok:
            return
        for m in (resp.json() or []):
            mem_id = m.get("id")
            if mem_id:
                self.session.delete(f"{self.base_url}/api/memories/{mem_id}", timeout=10)

    @staticmethod
    def _messages_to_text(messages: list[dict]) -> str:
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)

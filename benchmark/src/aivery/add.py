import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from src.aivery.client import AiveryClient

_ID_SCHEME = os.getenv("LOCOMO_ID_SCHEME", "conversation")


class AiveryAdd:
    """Ingests LOCOMO conversations into Aivery via extract → write."""

    def __init__(self, data_path: str | None = None, batch_size: int = 2, limit: int | None = None):
        self.client = AiveryClient()
        self.batch_size = batch_size
        self.data_path = data_path
        self.limit = limit
        self.data = None
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path) as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id: str, messages: list[dict], metadata: dict, retries: int = 3):
        for attempt in range(retries):
            try:
                self.client.add(messages, user_id=user_id, metadata=metadata)
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise e

    def add_memories_for_speaker(self, speaker: str, messages: list[dict], timestamp: str, desc: str):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch = messages[i: i + self.batch_size]
            self.add_memory(speaker, batch, metadata={"timestamp": timestamp})

    def _has_memories(self, user_id: str, threshold: int = 50) -> bool:
        try:
            resp = self.client.session.post(
                f"{self.client.base_url}/memory/retrieve",
                json={"agent_id": user_id, "query": "the", "limit": threshold},
                timeout=10,
            )
            if resp.ok:
                return len(resp.json().get("memories", [])) >= threshold
        except Exception:
            pass
        return False

    def process_conversation(self, item: dict, idx: int):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        if _ID_SCHEME == "conversation":
            id_a = f"conversation_{idx}"
            id_b = f"conversation_{idx}"
        else:
            id_a = f"{speaker_a}_{idx}"
            id_b = f"{speaker_b}_{idx}"

        if self._has_memories(id_a):
            print(f"Conversation {idx} already ingested — skipping.")
            return

        self.client.delete_all(user_id=id_a)
        if id_b != id_a:
            self.client.delete_all(user_id=id_b)

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            timestamp = conversation.get(key + "_date_time", "")
            chats = conversation[key]

            messages_a, messages_b = [], []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages_a.append({"role": "user",      "content": f"{speaker_a}: {chat['text']}"})
                    messages_b.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages_a.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_b.append({"role": "user",      "content": f"{speaker_b}: {chat['text']}"})

            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(id_a, messages_a, timestamp, f"Conv {idx} {speaker_a}"),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(id_b, messages_b, timestamp, f"Conv {idx} {speaker_b}"),
            )
            thread_a.start()
            thread_a.join()
            thread_b.start()
            thread_b.join()

        print(f"Conversation {idx} ingested.")

    def process_all_conversations(self, max_workers: int = 4):
        if not self.data:
            raise ValueError("No data loaded.")
        conversations = self.data[:self.limit] if self.limit is not None else self.data
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(conversations)]
            for future in futures:
                future.result()

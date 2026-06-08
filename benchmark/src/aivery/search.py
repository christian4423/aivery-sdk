import concurrent.futures
import json
import os
import threading
import time
from collections import defaultdict

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

_CORTEX_URL = os.getenv("CORTEX_BASE_URL", "https://cortex.aivery.systems")
_API_KEY    = os.getenv("AIVERY_API_KEY", "")
_ID_SCHEME  = os.getenv("LOCOMO_ID_SCHEME", "conversation")


class AiverySearch:
    """Answers LOCOMO questions via the Aivery Cortex agent endpoint.

    Cortex handles answer generation server-side — no OpenAI key needed for search.
    An OpenAI key is only required for the LLM judge step (evals.py).
    """

    def __init__(self, output_path: str = "results/aivery_results.json",
                 top_k: int = 50, limit: int | None = None,
                 max_workers: int = 3, question_delay: float = 0.5):
        self.output_path = output_path
        self.top_k = top_k
        self.limit = limit
        self.max_workers = max_workers
        self.question_delay = question_delay
        self.results = defaultdict(list)
        self.results_lock = threading.Lock()
        self._headers = {"Content-Type": "application/json"}
        if _API_KEY:
            self._headers["Authorization"] = f"Bearer {_API_KEY}"

    def _make_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update(self._headers)
        return s

    def answer_question(self, agent_id: str, question: str, max_retries: int = 3) -> dict:
        session = self._make_session()
        payload = {
            "message": question,
            "agent_id": agent_id,
            "agent_ids": [agent_id],
            "stream": False,
            "limit": self.top_k,
        }
        for attempt in range(max_retries):
            try:
                resp = session.post(f"{_CORTEX_URL}/api/agent/chat", json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                debug = data.get("debug", {}) or {}
                memories = debug.get("memories") or []
                return {
                    "response": data.get("response", ""),
                    "retrieved_memories": memories,
                }
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise e
        return {"response": "", "retrieved_memories": []}

    def _process_question(self, item):
        if self.question_delay > 0:
            time.sleep(self.question_delay)
        idx, q, agent_id = item
        result = self.answer_question(agent_id, q.get("question", ""))
        return idx, {
            "question":             q.get("question", ""),
            "answer":               q.get("answer", ""),
            "category":             q.get("category", -1),
            "evidence":             q.get("evidence", []),
            "response":             result["response"],
            "adversarial_answer":   q.get("adversarial_answer", ""),
            "speaker_1_memories":   [],
            "speaker_2_memories":   [],
            "num_speaker_1_memories": 0,
            "num_speaker_2_memories": 0,
            "speaker_1_graph_memories": None,
            "speaker_2_graph_memories": None,
            "response_time":        0,
            "retrieved_memories":   result["retrieved_memories"],
        }

    def process_data_file(self, file_path: str):
        with open(file_path) as f:
            data = json.load(f)

        data = data[:self.limit] if self.limit is not None else data

        all_items = []
        for idx, item in enumerate(data):
            conv = item["conversation"]
            if _ID_SCHEME == "conversation":
                agent_id = f"conversation_{idx}"
            else:
                agent_id = f"{conv['speaker_a']}_{idx}"

            for q in item["qa"]:
                if q.get("category") == 5:
                    continue
                all_items.append((idx, q, agent_id))

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_question, item) for item in all_items]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Questions"):
                try:
                    idx, record = future.result()
                    with self.results_lock:
                        self.results[idx].append(record)
                        completed += 1
                        if completed % 20 == 0:
                            with open(self.output_path, "w") as f:
                                json.dump(self.results, f, indent=4)
                except Exception as e:
                    tqdm.write(f"[warn] question failed: {e}")

        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

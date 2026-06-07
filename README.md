# Aivery Python SDK

AI memory that understands time — temporal-tree retrieval for LLM agents.

```python
from aivery import Memory

m = Memory()
m.add("I love hiking on weekends", user_id="alice")
results = m.search("outdoor activities", user_id="alice")
```

## Why Aivery?

Most memory systems store facts as a flat list. Aivery organizes memories into a **temporal tree**: related memories are parent/child, time flows from root to leaf, contradictions fork into branches. Retrieval walks the tree — you get a coherent branch of context, not a bag of facts.

Compared to mem0 on the LOCOMO benchmark (LLM judge score, higher = better):

| System | LLM Score |
|---|---|
| mem0 (platform) | 0.5571 |
| **Aivery (tree + heatmap, K=50)** | **0.6773** |
| **Aivery (wide retrieval, K=200→50)** | **0.8000** |

## Installation

```bash
pip install aivery
```

Requires a running Aivery server. See [Self-hosting](#self-hosting) below.

## Quickstart

```python
from aivery import Memory

m = Memory()

# Add from a plain string
m.add("I'm training for a marathon in April", user_id="alice")

# Add from a conversation
m.add(
    [
        {"role": "user", "content": "I just moved to San Francisco."},
        {"role": "assistant", "content": "Welcome! How are you finding it?"},
    ],
    user_id="alice",
)

# Search
results = m.search("where does alice live?", user_id="alice")
# [{"memory": "Alice lives in San Francisco", "score": 0.94, "id": "..."}]

# Get a context block for LLM injection
ctx = m.context("what do I know about alice?", user_id="alice")
# "- Alice is training for a marathon in April\n- Alice lives in San Francisco"
```

## Drop-in OpenAI agent

```python
import os
from openai import OpenAI
from aivery import Memory

openai = OpenAI()
memory = Memory()

def chat(message: str, user_id: str) -> str:
    ctx = memory.context(message, user_id=user_id, top_k=10)

    system = "You are a helpful assistant."
    if ctx:
        system += f"\n\nWhat you remember about the user:\n{ctx}"

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
    )
    answer = response.choices[0].message.content

    memory.add(
        [{"role": "user", "content": message}, {"role": "assistant", "content": answer}],
        user_id=user_id,
    )
    return answer
```

## Async

```python
from aivery import AsyncMemory

async with AsyncMemory() as m:
    await m.add("I love jazz", user_id="alice")
    results = await m.search("music preferences", user_id="alice")
```

## Configuration

| Method | Description |
|---|---|
| `Memory()` | Reads `AIVERY_BASE_URL`, `AIVERY_API_KEY`, `AIVERY_ORG_ID` from environment |
| `Memory(host="http://...")` | Explicit host |
| `Memory(api_key="aiv_...")` | Explicit API key |

Environment variables:

```bash
AIVERY_BASE_URL=http://localhost:5131   # default
AIVERY_API_KEY=aiv_...                  # required for auth-enabled deployments
AIVERY_ORG_ID=00000000-...             # multi-tenant org scoping
```

## Self-hosting

The Aivery server requires Postgres and Qdrant:

```bash
# Clone the server
git clone https://github.com/aivery-systems/aivery-api

# Start dependencies
docker compose up -d

# Start the API (port 5131)
dotnet run --project Memory/Memory.csproj
```

One-command Docker setup coming soon.

## License

MIT

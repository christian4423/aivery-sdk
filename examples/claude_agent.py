"""
Drop-in memory for a Claude chat agent.

Mirrors the openai_agent.py pattern — swap OpenAI for Claude.
Set ANTHROPIC_API_KEY before running.

Run: python examples/claude_agent.py
"""

import os
import anthropic
from aivery import Memory

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
memory = Memory()

USER_ID = "alice"
MODEL = "claude-sonnet-4-6"


def chat(message: str) -> str:
    ctx = memory.context(message, user_id=USER_ID, top_k=10)

    system = "You are a helpful personal assistant with a great memory."
    if ctx:
        system += f"\n\nWhat you remember about the user:\n{ctx}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": message}],
    )
    answer = response.content[0].text

    memory.add(
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ],
        user_id=USER_ID,
    )

    return answer


if __name__ == "__main__":
    turns = [
        "Hi! I just started training for my first marathon — it's in October.",
        "I've been running about 20 miles a week. Any advice on long-run nutrition?",
        "What do you remember about my fitness goals?",
    ]

    for msg in turns:
        print(f"User: {msg}")
        reply = chat(msg)
        print(f"Claude: {reply}\n")

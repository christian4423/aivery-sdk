"""
Multi-session personal assistant demo.

Simulates a user having five conversations spread across three months.
After each "session" the assistant remembers everything — so by the final
turn it can draw on a full picture of the user's life without any context
being passed in manually.

Run: python examples/personal_assistant.py
     ANTHROPIC_API_KEY must be set (or swap the LLM call for any provider).
"""

import os
import anthropic
from aivery import Memory

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
memory = Memory()

USER = "demo_personal"
MODEL = "claude-sonnet-4-6"


def session(date: str, conversation: list[dict[str, str]]) -> None:
    """Run a simulated past session: feed messages into memory without calling the LLM."""
    print(f"\n── Session {date} ──────────────────────────────────────────────")
    for m in conversation:
        print(f"  {m['role'].capitalize()}: {m['content']}")
    memory.add(conversation, user_id=USER, session_date=date)


def chat_now(message: str) -> str:
    """Live turn — retrieve memories, call LLM, persist the exchange."""
    ctx = memory.context(message, user_id=USER, top_k=12)
    system = "You are a warm, attentive personal assistant with a long memory."
    if ctx:
        system += f"\n\nEverything you know about this person:\n{ctx}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": message}],
    )
    answer = response.content[0].text

    memory.add(
        [{"role": "user", "content": message}, {"role": "assistant", "content": answer}],
        user_id=USER,
    )
    return answer


# ── Past sessions (simulated, no LLM calls) ───────────────────────────────────

session("2025-03-05", [
    {"role": "user", "content": "I'm Jordan. I work as a product designer at a fintech startup in New York."},
    {"role": "assistant", "content": "Great to meet you, Jordan! What brings you here today?"},
    {"role": "user", "content": "I want help staying on top of my goals. I'm trying to learn piano this year."},
    {"role": "assistant", "content": "That's a wonderful goal. How far along are you?"},
    {"role": "user", "content": "Just started — I can play a couple of simple songs."},
])

session("2025-03-28", [
    {"role": "user", "content": "My startup just got acquired! It's been a whirlwind week."},
    {"role": "assistant", "content": "Congratulations! How are you feeling about it?"},
    {"role": "user", "content": "Excited but nervous. I'll be staying on as a designer at the parent company."},
    {"role": "assistant", "content": "Big change. Is it the same city?"},
    {"role": "user", "content": "Yes, still New York. They have an office in Brooklyn which is actually closer to my apartment."},
])

session("2025-04-20", [
    {"role": "user", "content": "Piano is going well — I just learned Für Elise!"},
    {"role": "assistant", "content": "That's a real milestone. How long did it take you?"},
    {"role": "user", "content": "About six weeks of daily practice. My teacher says I'm moving fast."},
    {"role": "user", "content": "Also, I adopted a dog last weekend. A greyhound named Pixel."},
    {"role": "assistant", "content": "Pixel the greyhound — amazing name. Is this your first dog?"},
    {"role": "user", "content": "First dog living alone. A bit overwhelming but I love it."},
])

session("2025-05-15", [
    {"role": "user", "content": "Work has been hectic since the acquisition. Long hours, lots of new processes."},
    {"role": "assistant", "content": "That kind of transition is always draining. How's Pixel handling it?"},
    {"role": "user", "content": "He's my stress relief honestly. We run in Prospect Park every morning."},
    {"role": "user", "content": "I'm also thinking of going back to school part-time for an HCI master's."},
    {"role": "assistant", "content": "That would complement your design work nicely. Do you have a program in mind?"},
    {"role": "user", "content": "NYU. Applications open in September."},
])

# ── Live session today ────────────────────────────────────────────────────────

print("\n\n── Live conversation (today) ────────────────────────────────────────")

questions = [
    "Hey, it's been a while. What do you remember about me?",
    "I got into the NYU HCI program! Deferred to next fall though. Any thoughts on balancing grad school with full-time work?",
    "How do you think piano and design connect for someone like me?",
]

for q in questions:
    print(f"\nJordan: {q}")
    reply = chat_now(q)
    print(f"Assistant: {reply}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
memory.delete_user(USER)
print("\n\nDemo user deleted.")

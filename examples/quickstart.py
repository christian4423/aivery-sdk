"""
Quickstart — identical to the mem0 quickstart pattern.
Run: python examples/quickstart.py
"""

from aivery import Memory

m = Memory()

# Add a memory from a plain string
m.add("I love hiking in the mountains on weekends", user_id="alice")

# Add a memory from a conversation
m.add(
    [
        {"role": "user", "content": "I just moved to San Francisco last month."},
        {"role": "assistant", "content": "That's exciting! How are you finding it?"},
    ],
    user_id="alice",
)

# Search
results = m.search("where does alice live?", user_id="alice")
print("Search results:")
for r in results:
    print(f"  [{r['score']:.3f}] {r['memory']}")

# Get a context block ready to inject into an LLM prompt
ctx = m.context("what do I know about alice?", user_id="alice")
print("\nContext for LLM:")
print(ctx)

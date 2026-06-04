"""
Temporal memory demo — Aivery's core differentiator.

Most memory systems store facts as a flat list and return whichever chunk
scores highest. Aivery organises memories into a temporal tree: related
memories are parent/child, time flows root → leaf, contradictions fork into
branches. Retrieval walks the tree so you get a *coherent branch*, not a
bag of facts from different moments in time.

This demo simulates a user whose job and city change over six months, then
shows that the retrieved context reflects the *current* reality — not an
average of all past states.

Run: python examples/temporal_demo.py
"""

from aivery import Memory

m = Memory()
USER = "demo_temporal"

# ── Seed memories across six simulated months ─────────────────────────────────

print("Seeding memories across January → June ...\n")

# January
m.add(
    "I work as a backend engineer at Google in San Francisco.",
    user_id=USER,
    session_date="2025-01-10",
)

# February
m.add(
    [
        {"role": "user", "content": "I've been house-hunting in Austin — thinking of relocating."},
        {"role": "assistant", "content": "That's a big move! What's drawing you to Austin?"},
    ],
    user_id=USER,
    session_date="2025-02-14",
)

# March — layoff
m.add(
    "I was laid off from Google last week. Looking for a new role now.",
    user_id=USER,
    session_date="2025-03-03",
)

# April — new job, moved
m.add(
    [
        {"role": "user", "content": "I accepted an offer from Stripe! Starting in two weeks."},
        {"role": "assistant", "content": "Congratulations! That's a fantastic company."},
        {"role": "user", "content": "And I finally signed a lease in Austin — moving end of April."},
        {"role": "assistant", "content": "Exciting chapter. Austin will suit you."},
    ],
    user_id=USER,
    session_date="2025-04-15",
)

# June — settled
m.add(
    "I've been at Stripe for six weeks now. Love the team. Austin has been great.",
    user_id=USER,
    session_date="2025-06-01",
)

# ── Now query as if it's present day ─────────────────────────────────────────

queries = [
    ("Where does this person work?", "employer / job status"),
    ("What city does this person live in?", "location"),
    ("Tell me about this person's career", "career arc"),
]

for query, label in queries:
    print(f"Query: {query!r}  [{label}]")
    ctx = m.context(query, user_id=USER, top_k=8)
    print(ctx or "  (no memories)")
    print()

# ── Show raw scored results for the career query ──────────────────────────────

print("── Raw scored results for 'career' ─────────────────────────────────────")
results = m.search("career and job", user_id=USER, top_k=10)
for r in results:
    print(f"  [{r['score']:.3f}] {r['memory']}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
m.delete_user(USER)
print("\nDemo user deleted.")

"""
Async quickstart — use AsyncMemory when you're already in an async context
(FastAPI, LangChain async pipelines, etc.)
"""

import asyncio
from aivery import AsyncMemory


async def main() -> None:
    async with AsyncMemory() as m:
        await m.add("I'm training for a marathon", user_id="bob")
        await m.add(
            [
                {"role": "user", "content": "My doctor said I need to eat more protein."},
                {"role": "assistant", "content": "Got it, I'll keep that in mind."},
            ],
            user_id="bob",
        )

        results = await m.search("bob's fitness goals", user_id="bob")
        print("Results:")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['memory']}")

        ctx = await m.context("what should I recommend for bob?", user_id="bob")
        print("\nContext block:\n" + ctx)


if __name__ == "__main__":
    asyncio.run(main())

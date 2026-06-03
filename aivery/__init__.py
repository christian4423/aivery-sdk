"""
Aivery — AI memory that understands time.

Quick start::

    from aivery import Memory

    m = Memory()
    m.add("I love hiking on weekends", user_id="alice")
    results = m.search("outdoor activities", user_id="alice")

Async::

    from aivery import AsyncMemory

    async with AsyncMemory() as m:
        await m.add("I love hiking on weekends", user_id="alice")
        results = await m.search("outdoor activities", user_id="alice")
"""

from ._sync_client import Memory
from ._async_client import AsyncMemory

__version__ = "0.1.0"
__all__ = ["Memory", "AsyncMemory"]

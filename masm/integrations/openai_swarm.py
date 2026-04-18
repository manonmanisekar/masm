"""OpenAI Swarm adapter — MASM shared memory for Swarm agents.

Status: Stub — planned for v0.4.0.
Requires: pip install openai-swarm

Usage:
    from masm.integrations.openai_swarm import MASMSwarmMemory
    memory = MASMSwarmMemory(store=my_store)
"""

from typing import Optional

from masm.core.memory import MemoryRecord
from masm.store.base import SharedMemoryStore


class MASMSwarmMemory:
    """Shared memory adapter for OpenAI Swarm agents."""

    def __init__(self, store: SharedMemoryStore, tags: Optional[list[str]] = None):
        self.store = store
        self.tags = tags or ["swarm"]

    async def save(self, agent_id: str, content: str, **kwargs) -> MemoryRecord:
        """Save a memory from a swarm agent."""
        record = MemoryRecord(
            content=content,
            author_agent_id=agent_id,
            tags=self.tags + kwargs.get("tags", []),
            confidence=kwargs.get("confidence", 0.9),
        )
        written, _ = await self.store.write(record)
        return written

    async def recall(self, agent_id: str, query: str = "", limit: int = 10) -> list[MemoryRecord]:
        """Retrieve shared memories for a swarm agent."""
        return await self.store.read(
            agent_id=agent_id,
            query=query,
            tags=self.tags,
            limit=limit,
        )

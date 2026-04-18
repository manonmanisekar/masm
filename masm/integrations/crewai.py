"""CrewAI adapter — wraps MASM shared memory for CrewAI agents.

Status: Stub — full implementation coming in v0.3.0.
Requires: pip install crewai

Usage:
    from masm.integrations.crewai import MASMCrewMemory
    shared = MASMCrewMemory(store=my_store)
    agent_memory = shared.for_agent("researcher")
"""

from typing import Optional

from masm.core.memory import MemoryRecord
from masm.store.base import SharedMemoryStore


class MASMCrewMemory:
    """
    Shared memory adapter for CrewAI crews.

    Wraps a MASM SharedMemoryStore so all agents in a crew automatically
    share memories with conflict detection and relevance filtering.
    """

    def __init__(self, store: SharedMemoryStore, crew_tags: Optional[list[str]] = None):
        self.store = store
        self.crew_tags = crew_tags or ["crew"]

    def for_agent(self, agent_id: str, tags: Optional[list[str]] = None) -> "MASMAgentMemory":
        """Create an agent-specific memory view into shared store."""
        return MASMAgentMemory(
            store=self.store,
            agent_id=agent_id,
            tags=self.crew_tags + (tags or []),
        )


class MASMAgentMemory:
    """Per-agent memory view into a shared MASM store."""

    def __init__(self, store: SharedMemoryStore, agent_id: str, tags: list[str]):
        self.store = store
        self.agent_id = agent_id
        self.tags = tags

    async def save(self, content: str, confidence: float = 0.9) -> MemoryRecord:
        """Save a memory from this agent to shared store."""
        record = MemoryRecord(
            content=content,
            author_agent_id=self.agent_id,
            tags=self.tags,
            confidence=confidence,
        )
        written, _ = await self.store.write(record)
        return written

    async def recall(self, query: str = "", limit: int = 10) -> list[MemoryRecord]:
        """Retrieve relevant memories for this agent."""
        return await self.store.read(
            agent_id=self.agent_id,
            query=query,
            tags=self.tags,
            limit=limit,
        )

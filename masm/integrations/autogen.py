"""AutoGen adapter — attaches MASM shared memory to AutoGen agents.

Status: Stub — full implementation coming in v0.3.0.
Requires: pip install pyautogen

Usage:
    from masm.integrations.autogen import MASMAutoGenMemory
    memory = MASMAutoGenMemory(store=my_store)
    memory.attach(my_autogen_agent)
"""

from typing import Optional

from masm.core.memory import MemoryRecord
from masm.store.base import SharedMemoryStore


class MASMAutoGenMemory:
    """
    Shared memory adapter for AutoGen multi-agent conversations.

    Intercepts messages between AutoGen agents and stores them as shared
    memories, enabling cross-conversation context sharing.
    """

    def __init__(self, store: SharedMemoryStore, tags: Optional[list[str]] = None):
        self.store = store
        self.tags = tags or ["autogen"]
        self._attached_agents: list[str] = []

    def attach(self, agent) -> None:
        """Attach MASM memory to an AutoGen agent."""
        agent_name = getattr(agent, "name", str(id(agent)))
        self._attached_agents.append(agent_name)

    async def on_message(self, sender: str, receiver: str, content: str) -> None:
        """Hook to capture messages between agents as shared memories."""
        await self.store.write(MemoryRecord(
            content=content,
            author_agent_id=sender,
            tags=self.tags + [f"to:{receiver}"],
            confidence=0.9,
        ))

    async def get_context(self, agent_id: str, limit: int = 10) -> list[MemoryRecord]:
        """Retrieve shared context for an agent."""
        return await self.store.read(
            agent_id=agent_id,
            tags=self.tags,
            limit=limit,
        )

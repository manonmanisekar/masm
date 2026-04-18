"""LangChain/LangGraph adapter — wraps MASM shared memory as LangChain-compatible memory.

Status: Stub — full implementation coming in v0.3.0.
Requires: pip install langchain

Usage:
    from masm.integrations.langchain import MASMChatMemory
    memory = MASMChatMemory(store=my_store, agent_id="researcher")
"""

from typing import Any, Optional

from masm.core.memory import MemoryRecord
from masm.store.base import SharedMemoryStore


class MASMChatMemory:
    """
    LangChain-compatible chat memory backed by MASM shared store.

    Drop-in replacement for LangChain's ConversationBufferMemory that
    enables multi-agent memory sharing with conflict detection.
    """

    memory_key: str = "history"

    def __init__(self, store: SharedMemoryStore, agent_id: str, tags: Optional[list[str]] = None):
        self.store = store
        self.agent_id = agent_id
        self.tags = tags or ["conversation"]

    async def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Load relevant shared memories for this agent's context."""
        query = inputs.get("input", "")
        records = await self.store.read(
            agent_id=self.agent_id,
            query=query,
            tags=self.tags,
            limit=20,
        )
        history = "\n".join(f"[{r.author_agent_id}]: {r.content}" for r in records)
        return {self.memory_key: history}

    async def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save conversation turn to shared memory."""
        # Save user input
        if "input" in inputs:
            await self.store.write(MemoryRecord(
                content=inputs["input"],
                author_agent_id="user",
                tags=self.tags + ["user_input"],
            ))

        # Save agent output
        if "output" in outputs:
            await self.store.write(MemoryRecord(
                content=outputs["output"],
                author_agent_id=self.agent_id,
                tags=self.tags + ["agent_output"],
            ))

    async def clear(self) -> None:
        """Clear is not supported in shared memory — use forget() for specific records."""
        pass

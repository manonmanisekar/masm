"""Abstract SharedMemoryStore interface — all backends must implement this."""

from abc import ABC, abstractmethod
from typing import Optional

from masm.core.memory import ConflictEvent, MemoryRecord, MemoryState


class SharedMemoryStore(ABC):
    """
    Abstract interface for multi-agent shared memory stores.

    Any backend (Redis, SQLite, Postgres, in-memory) must implement this.
    The benchmark tests against this interface, making results comparable.
    """

    @abstractmethod
    async def write(
        self,
        record: MemoryRecord,
        conflict_strategy: str = "lww",
    ) -> tuple[MemoryRecord, list[ConflictEvent]]:
        """
        Write a memory record to shared store.

        Returns:
            - The written record (may be modified by merge)
            - List of conflicts detected during write

        Must handle:
            1. Check for duplicates (semantic similarity > threshold)
            2. Check for conflicts (same topic, different facts)
            3. Apply conflict resolution strategy
            4. Update vector clock
            5. Emit change notification to subscribed agents
            6. Write audit log entry
        """
        ...

    @abstractmethod
    async def read(
        self,
        agent_id: str,
        query: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
        tags: Optional[list[str]] = None,
        limit: int = 10,
        include_disputed: bool = False,
        consistency: str = "eventual",
    ) -> list[MemoryRecord]:
        """
        Read memories relevant to an agent's current context.

        Must handle:
            1. Filter by agent's read permissions
            2. Rank by relevance to query (semantic + recency + confidence)
            3. Exclude superseded/retracted unless explicitly requested
            4. Respect consistency level
            5. Log read for audit trail
        """
        ...

    @abstractmethod
    async def update(
        self,
        record_id: str,
        agent_id: str,
        updates: dict,
        reason: str = "",
    ) -> MemoryRecord:
        """
        Update a memory (creates new version, marks old as superseded).
        Updates are NEVER in-place — always version-creating.
        """
        ...

    @abstractmethod
    async def forget(
        self,
        record_id: str,
        agent_id: str,
        reason: str = "gdpr_request",
        cascade: bool = True,
    ) -> bool:
        """
        GDPR-compliant forgetting. Marks record as FORGOTTEN.
        Content is replaced with '[FORGOTTEN]' but metadata/audit trail preserved.
        """
        ...

    @abstractmethod
    async def get_conflicts(
        self,
        unresolved_only: bool = True,
    ) -> list[tuple[MemoryRecord, MemoryRecord]]:
        """Return all detected memory conflicts."""
        ...

    @abstractmethod
    async def get_audit_log(
        self,
        record_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        since: Optional[str] = None,
    ) -> list[dict]:
        """Full audit trail for compliance."""
        ...

    @abstractmethod
    async def subscribe(
        self,
        agent_id: str,
        tags: Optional[list[str]] = None,
        callback: Optional[callable] = None,
    ) -> str:
        """Subscribe an agent to memory change notifications. Returns subscription ID."""
        ...

    @abstractmethod
    async def stats(self) -> dict:
        """Return store statistics."""
        ...

    @abstractmethod
    async def list_records(
        self,
        states: Optional[list[MemoryState]] = None,
        agent_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[MemoryRecord]:
        """
        Enumerate all records, with optional filtering.

        Args:
            states: If provided, only return records in these states.
                    Defaults to all states (including SUPERSEDED, FORGOTTEN).
            agent_id: If provided, only return records authored by this agent.
            tags: If provided, only return records with at least one matching tag.

        Returns:
            List of MemoryRecord instances.
        """
        ...

    @abstractmethod
    async def list_conflicts(
        self,
        resolved: Optional[bool] = None,
    ) -> list[ConflictEvent]:
        """
        Return raw ConflictEvent objects (not just record pairs).

        Args:
            resolved: If True, only resolved conflicts. If False, only unresolved.
                      If None, return all conflicts.

        Returns:
            List of ConflictEvent dataclass instances.
        """
        ...

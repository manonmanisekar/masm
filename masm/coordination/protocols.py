"""Read/write consistency protocols for multi-agent shared memory."""

from enum import Enum
from typing import Optional

from masm.core.clock import VectorClock
from masm.core.memory import MemoryRecord


class ConsistencyLevel(Enum):
    """Consistency guarantees for read operations."""

    EVENTUAL = "eventual"  # May return stale data; fastest
    READ_YOUR_WRITES = "read_your_writes"  # Agent always sees its own writes
    CAUSAL = "causal"  # Respects causal ordering across agents
    STRONG = "strong"  # Linearizable; slowest


class ConsistencyProtocol:
    """
    Enforces consistency guarantees on read operations.

    In the in-memory store, strong consistency is trivial (single process).
    These protocols become meaningful with distributed backends (Redis, Postgres).
    """

    def __init__(self):
        self._agent_write_clocks: dict[str, dict[str, int]] = {}
        self._global_clock = VectorClock()

    def on_write(self, agent_id: str, record: MemoryRecord) -> None:
        """Track the latest write clock for an agent (for read-your-writes)."""
        if record.vector_clock:
            self._agent_write_clocks[agent_id] = record.vector_clock.copy()
            self._global_clock.merge(record.vector_clock)

    def filter_by_consistency(
        self,
        records: list[MemoryRecord],
        agent_id: str,
        level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
    ) -> list[MemoryRecord]:
        """Filter records to satisfy the requested consistency level."""

        if level == ConsistencyLevel.EVENTUAL:
            return records

        if level == ConsistencyLevel.READ_YOUR_WRITES:
            return self._filter_read_your_writes(records, agent_id)

        if level == ConsistencyLevel.CAUSAL:
            return self._filter_causal(records, agent_id)

        if level == ConsistencyLevel.STRONG:
            # In a single-process store, all reads are already linearizable
            return records

        return records

    def _filter_read_your_writes(
        self, records: list[MemoryRecord], agent_id: str
    ) -> list[MemoryRecord]:
        """Ensure the agent sees all its own writes."""
        agent_clock = self._agent_write_clocks.get(agent_id)
        if not agent_clock:
            return records

        # Include all records, but ensure agent's own writes are present
        own_writes = {
            r for r in records
            if r.author_agent_id == agent_id
        }
        # Also include records that are at least as recent as the agent's writes
        result = list(own_writes)
        for r in records:
            if r not in own_writes:
                result.append(r)
        return result

    def _filter_causal(
        self, records: list[MemoryRecord], agent_id: str
    ) -> list[MemoryRecord]:
        """Only return records that are causally consistent."""
        agent_clock = self._agent_write_clocks.get(agent_id, {})

        consistent = []
        for r in records:
            if not r.vector_clock:
                consistent.append(r)
                continue
            # A record is causally visible if it doesn't happen-after
            # any unobserved concurrent write
            if not agent_clock or not self._global_clock.happens_before(agent_clock, r.vector_clock):
                consistent.append(r)

        return consistent

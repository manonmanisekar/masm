"""Mem0 baseline adapter — wraps Mem0 in the SharedMemoryStore interface.

Requires: pip install masm[benchmarks]

NOTE: Mem0 is designed for single-agent memory. These benchmarks test it in
multi-agent scenarios it wasn't built for. Low scores reflect a mismatch
in use case, not a flaw in Mem0.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from masm.core.memory import AuditEntry, ConflictEvent, MemoryRecord, MemoryState
from masm.store.base import SharedMemoryStore


class Mem0Adapter(SharedMemoryStore):
    """
    Adapts Mem0 to the SharedMemoryStore interface for benchmark comparison.

    This adapter translates MASM operations into Mem0 API calls. Because Mem0
    doesn't natively support multi-agent concepts like conflict detection or
    vector clocks, many operations are approximated or no-ops.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._available = False
        self._client = None
        try:
            from mem0 import Memory
            config = {}
            if api_key:
                config["api_key"] = api_key
            self._client = Memory()
            self._available = True
        except ImportError:
            pass

        self._audit: list[AuditEntry] = []
        self._write_count = 0
        self._read_count = 0

    @property
    def available(self) -> bool:
        return self._available

    async def write(
        self,
        record: MemoryRecord,
        conflict_strategy: str = "lww",
    ) -> tuple[MemoryRecord, list[ConflictEvent]]:
        if not self._available:
            return record, []

        self._client.add(
            record.content,
            user_id=record.author_agent_id,
            metadata={
                "tags": ",".join(record.tags),
                "masm_id": record.id,
                "confidence": record.confidence,
            },
        )
        self._write_count += 1
        self._audit.append(
            AuditEntry(operation="write", agent_id=record.author_agent_id, record_id=record.id)
        )
        # Mem0 doesn't return conflict info
        return record, []

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
        if not self._available:
            return []

        search_query = query or (", ".join(tags) if tags else "")
        if not search_query:
            return []

        results = self._client.search(search_query, user_id=agent_id, limit=limit)
        self._read_count += 1

        records = []
        for item in results.get("results", []):
            record = MemoryRecord(
                id=item.get("metadata", {}).get("masm_id", str(uuid.uuid4())),
                content=item.get("memory", ""),
                author_agent_id=item.get("user_id", "unknown"),
                confidence=float(item.get("metadata", {}).get("confidence", 0.5)),
                tags=item.get("metadata", {}).get("tags", "").split(","),
            )
            records.append(record)

        self._audit.append(
            AuditEntry(operation="read", agent_id=agent_id)
        )
        return records

    async def update(
        self,
        record_id: str,
        agent_id: str,
        updates: dict,
        reason: str = "",
    ) -> MemoryRecord:
        # Mem0 doesn't support versioned updates — just write new
        new_record = MemoryRecord(
            id=str(uuid.uuid4()),
            author_agent_id=agent_id,
            **updates,
        )
        await self.write(new_record)
        return new_record

    async def forget(
        self,
        record_id: str,
        agent_id: str,
        reason: str = "gdpr_request",
        cascade: bool = True,
    ) -> bool:
        # Mem0 doesn't expose per-record deletion in the same way
        self._audit.append(
            AuditEntry(operation="forget", agent_id=agent_id, record_id=record_id)
        )
        return True

    async def get_conflicts(
        self, unresolved_only: bool = True
    ) -> list[tuple[MemoryRecord, MemoryRecord]]:
        return []  # Mem0 doesn't detect conflicts

    async def get_audit_log(
        self,
        record_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        since: Optional[str] = None,
    ) -> list[dict]:
        entries = self._audit
        if record_id:
            entries = [e for e in entries if e.record_id == record_id]
        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]
        return [
            {"id": e.id, "timestamp": e.timestamp.isoformat(), "operation": e.operation,
             "agent_id": e.agent_id, "record_id": e.record_id, "details": e.details}
            for e in entries
        ]

    async def subscribe(
        self,
        agent_id: str,
        tags: Optional[list[str]] = None,
        callback: Optional[callable] = None,
    ) -> str:
        return str(uuid.uuid4())  # No-op

    async def stats(self) -> dict:
        return {
            "total_memories": self._write_count,
            "active_memories": self._write_count,
            "conflicts_pending": 0,
            "total_writes": self._write_count,
            "total_reads": self._read_count,
            "audit_entries": len(self._audit),
            "subscriptions": 0,
        }

"""Naive baseline: simple shared Python dict with no dedup, no conflict detection, no ordering."""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from masm.core.memory import AuditEntry, ConflictEvent, MemoryRecord, MemoryState
from masm.store.base import SharedMemoryStore


class NaiveSharedDictStore(SharedMemoryStore):
    """
    Simplest possible shared memory: a Python dict.

    No deduplication, no conflict detection, no vector clocks, no relevance.
    This is the baseline that demonstrates why multi-agent coordination matters.
    """

    def __init__(self):
        self._data: dict[str, MemoryRecord] = {}
        self._audit: list[AuditEntry] = []
        self._write_count = 0
        self._read_count = 0

    async def write(
        self,
        record: MemoryRecord,
        conflict_strategy: str = "lww",
    ) -> tuple[MemoryRecord, list[ConflictEvent]]:
        """Just store it — no dedup, no conflict detection."""
        self._data[record.id] = record
        self._write_count += 1
        self._audit.append(
            AuditEntry(operation="write", agent_id=record.author_agent_id, record_id=record.id)
        )
        return record, []  # Never reports conflicts

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
        """Return everything — no filtering, no ranking."""
        results = list(self._data.values())
        if tags:
            results = [r for r in results if set(tags) & set(r.tags)]
        self._read_count += 1
        self._audit.append(
            AuditEntry(operation="read", agent_id=agent_id)
        )
        return results[:limit]

    async def update(
        self,
        record_id: str,
        agent_id: str,
        updates: dict,
        reason: str = "",
    ) -> MemoryRecord:
        """Replace record — no versioning."""
        from dataclasses import replace
        if record_id not in self._data:
            raise KeyError(f"Record {record_id} not found")
        record = self._data[record_id]
        valid_updates = {k: v for k, v in updates.items() if hasattr(record, k)}
        valid_updates["updated_at"] = datetime.now(timezone.utc)
        record = replace(record, **valid_updates)
        self._data[record_id] = record
        return record

    async def forget(
        self,
        record_id: str,
        agent_id: str,
        reason: str = "gdpr_request",
        cascade: bool = True,
    ) -> bool:
        """Just delete — no audit preservation, no cascade."""
        if record_id in self._data:
            del self._data[record_id]
            return True
        return False

    async def get_conflicts(
        self,
        unresolved_only: bool = True,
    ) -> list[tuple[MemoryRecord, MemoryRecord]]:
        """Never detects conflicts."""
        return []

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
            "total_memories": len(self._data),
            "active_memories": len(self._data),
            "conflicts_pending": 0,
            "total_writes": self._write_count,
            "total_reads": self._read_count,
            "audit_entries": len(self._audit),
            "subscriptions": 0,
        }

    async def list_records(
        self,
        states: Optional[list] = None,
        agent_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[MemoryRecord]:
        """Return all records with optional filtering."""
        results = list(self._data.values())
        if states is not None:
            results = [r for r in results if r.state in states]
        if agent_id is not None:
            results = [r for r in results if r.author_agent_id == agent_id]
        if tags:
            results = [r for r in results if set(tags) & set(r.tags)]
        return results

    async def list_conflicts(
        self,
        resolved: Optional[bool] = None,
    ) -> list[ConflictEvent]:
        """Naive store never detects conflicts."""
        return []

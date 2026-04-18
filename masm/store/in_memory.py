"""In-memory SharedMemoryStore implementation for testing and benchmarks."""

import asyncio
import logging
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Optional

from masm.core.clock import VectorClock
from masm.core.conflict import ConflictDetector, ConflictResolver
from masm.core.memory import (
    AuditEntry,
    ConflictEvent,
    ConflictStrategy,
    MemoryEdge,
    EdgeKind,
    MemoryRecord,
    MemoryState,
    MemoryType,
    WriteOutcome,
    WriteResult,
)
from masm.cognitive import (
    IntentClassifier,
    MemoryTypeClassifier,
    SemanticNormalizer,
    TrustEngine,
)
from masm.coordination.merge import MemoryMerger
from masm.explain import ConflictExplainer, ConflictExplanation
from masm.index import BruteForceIndex, VectorIndex
from masm.indexing import EntityAttributeIndex
from masm.semantic.dedup import SemanticDeduplicator
from masm.store.base import SharedMemoryStore

logger = logging.getLogger(__name__)


class InMemorySharedStore(SharedMemoryStore):
    """
    Dict-backed SharedMemoryStore for testing and benchmarks.

    Thread-safe via asyncio locks. All data lives in-process memory.
    Designed to be fast enough for benchmarks to complete in <60 seconds.
    """

    def __init__(
        self,
        dedup_threshold: float = 0.92,
        conflict_threshold: float = 0.85,
        vector_index: Optional[VectorIndex] = None,
        normalizer: Optional[SemanticNormalizer] = None,
        type_classifier: Optional[MemoryTypeClassifier] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        trust_engine: Optional[TrustEngine] = None,
        entity_attribute_index: Optional[EntityAttributeIndex] = None,
        conflict_explainer: Optional[ConflictExplainer] = None,
    ):
        """Create an in-memory store.

        Args:
            dedup_threshold: cosine-similarity threshold for dedup. Pairs at or
                above this are treated as the same fact (merged / superseded).
            conflict_threshold: similarity threshold used by `ConflictDetector`.
                Pairs *below* this with overlapping tags are flagged as
                conflicting facts. Kept independently from dedup because the
                two decisions answer different questions.
        """
        self._records: dict[str, MemoryRecord] = {}
        self._conflicts: list[ConflictEvent] = []
        self._edges: list[MemoryEdge] = []
        self._audit_log: list[AuditEntry] = []
        self._subscriptions: dict[str, dict] = {}
        self._vector_clock = VectorClock()
        self._conflict_detector = ConflictDetector(semantic_threshold=conflict_threshold)
        self._conflict_resolver = ConflictResolver(trust_engine=trust_engine)
        self._deduplicator = SemanticDeduplicator(threshold=dedup_threshold)
        self._merger = MemoryMerger()
        self._vector_index: VectorIndex = vector_index or BruteForceIndex()
        self._normalizer = normalizer
        self._type_classifier = type_classifier
        self._intent_classifier = intent_classifier
        self._trust_engine = trust_engine
        self._entity_index = entity_attribute_index
        self._conflict_explainer = conflict_explainer
        self._explanations: list[ConflictExplanation] = []
        self._dedup_threshold = dedup_threshold
        self._conflict_threshold = conflict_threshold
        self._lock = asyncio.Lock()
        self.last_write_result: Optional[WriteResult] = None
        self._write_count = 0
        self._read_count = 0
        self._write_latency = 0.0
        self._read_latency = 0.0
        
    async def write(
        self,
        record: MemoryRecord,
        conflict_strategy: Optional[str] = None,
    ) -> tuple[MemoryRecord, list[ConflictEvent]]:
        """Write a memory record, detecting duplicates and conflicts.

        When `conflict_strategy` is left unset, the store picks a sensible
        default: AUTHORITY_RANK (confidence-tiebroken) if a TrustEngine is
        attached, otherwise LWW. Callers can still override explicitly.

        Also stores the last WriteResult on self.last_write_result for callers
        that need the structured outcome.
        """
        t_start = time.perf_counter()
        async with self._lock:
            # 0. Cognitive preprocessing (normalizer + type classifier)
            record = self._preprocess(record)

            # 1. Update vector clock
            new_clock = self._vector_clock.increment(record.author_agent_id)
            record = replace(record, vector_clock=new_clock)

            # 2. Check for semantic duplicates (embedding-based). Merge through
            #    the coordination MemoryMerger so the tag-union and provenance
            #    metadata are preserved.
            duplicate_id = self._find_duplicate(record)
            if duplicate_id:
                existing = self._records[duplicate_id]
                merged, superseded = self._merger.merge(record, existing)

                # The merger picks the higher-confidence side as the winner.
                # If the incoming record won, store both merged + superseded.
                # If the existing record won, the incoming write is a no-op
                # at the record level but still produces a DEDUPLICATED outcome.
                if merged.id == record.id:
                    # Incoming record is the winner — carry the supersedes link
                    merged = replace(merged, supersedes_id=existing.id)
                    self._records[existing.id] = superseded
                    self._records[merged.id] = merged
                    self._deindex_record(existing.id)
                    self._index_record(merged)
                    self._edges.append(MemoryEdge(
                        source_id=merged.id,
                        target_id=existing.id,
                        kind=EdgeKind.SUPERSEDES,
                        metadata={"reason": "semantic_dedup"},
                    ))
                    self._edges.append(MemoryEdge(
                        source_id=merged.id,
                        target_id=existing.id,
                        kind=EdgeKind.DERIVED_FROM,
                        metadata={"via": "merge"},
                    ))
                    self._audit_log.append(
                        AuditEntry(
                            operation="dedup_merge",
                            agent_id=record.author_agent_id,
                            record_id=merged.id,
                            details={"merged_with": existing.id},
                        )
                    )
                    self._write_count += 1
                    self._write_latency += time.perf_counter() - t_start
                    self.last_write_result = WriteResult(
                        record=merged,
                        outcome=WriteOutcome.DEDUPLICATED,
                        duplicate_of=existing.id,
                    )
                    await self._notify_subscribers(merged)
                    return merged, []
                else:
                    # Existing wins — the incoming record is dropped but we
                    # still record the provenance edge so callers can audit it.
                    self._edges.append(MemoryEdge(
                        source_id=existing.id,
                        target_id=record.id,
                        kind=EdgeKind.DERIVED_FROM,
                        metadata={"via": "dedup_skip"},
                    ))
                    self._audit_log.append(
                        AuditEntry(
                            operation="dedup_skip",
                            agent_id=record.author_agent_id,
                            record_id=record.id,
                            details={"duplicate_of": duplicate_id},
                        )
                    )
                    self._write_latency += time.perf_counter() - t_start
                    self.last_write_result = WriteResult(
                        record=existing,
                        outcome=WriteOutcome.DEDUPLICATED,
                        duplicate_of=duplicate_id,
                    )
                    return existing, []

            # 3. Detect conflicts
            active_records = [
                r for r in self._records.values() if r.state == MemoryState.ACTIVE
            ]
            conflicts = self._conflict_detector.detect(record, active_records)

            # 4. Resolve conflicts
            if conflict_strategy is None:
                effective_strategy = (
                    "authority_rank" if self._trust_engine is not None else "lww"
                )
            else:
                effective_strategy = conflict_strategy
            strategy = ConflictStrategy(effective_strategy)
            resolved_conflicts = []
            for conflict in conflicts:
                existing = self._records[conflict.memory_a_id]
                winner, reasoning = self._conflict_resolver.resolve(
                    existing, record, strategy
                )
                resolved_conflict = replace(
                    conflict,
                    resolved=True,
                    strategy=strategy,
                    resolved_memory_id=winner.id,
                    reasoning=reasoning,
                )
                resolved_conflicts.append(resolved_conflict)

                # Typed edge: the two records contradict each other.
                self._edges.append(MemoryEdge(
                    source_id=record.id,
                    target_id=existing.id,
                    kind=EdgeKind.CONTRADICTS,
                    metadata={"strategy": strategy.value, "reason": reasoning},
                ))

                # LWW: loser gets superseded
                if winner.id == record.id:
                    self._records[existing.id] = replace(existing, state=MemoryState.SUPERSEDED)
                    self._deindex_record(existing.id)
                    if self._trust_engine is not None:
                        self._trust_engine.record_event(record.author_agent_id, "conflict_win")
                        self._trust_engine.record_event(existing.author_agent_id, "conflict_loss")
                    self._edges.append(MemoryEdge(
                        source_id=record.id,
                        target_id=existing.id,
                        kind=EdgeKind.SUPERSEDES,
                        metadata={"strategy": strategy.value},
                    ))
                else:
                    record = replace(record, state=MemoryState.SUPERSEDED)
                    self._edges.append(MemoryEdge(
                        source_id=existing.id,
                        target_id=record.id,
                        kind=EdgeKind.SUPERSEDES,
                        metadata={"strategy": strategy.value},
                    ))
                    if self._trust_engine is not None:
                        self._trust_engine.record_event(existing.author_agent_id, "conflict_win")
                        self._trust_engine.record_event(record.author_agent_id, "conflict_loss")

            self._conflicts.extend(resolved_conflicts)
            conflicts = resolved_conflicts

            # Structured explanations (optional)
            if self._conflict_explainer is not None:
                authority = (
                    self._trust_engine.as_authority_dict()
                    if self._trust_engine is not None
                    else None
                )
                # ConflictResolver currently returns raw authority_rank dicts,
                # but the explainer wants a flat agent_id → float mapping.
                flat_authority: Optional[dict[str, float]] = None
                if authority is not None:
                    flat_authority = {
                        k: v.get("authority_rank", 0.0) for k, v in authority.items()
                    }
                for resolved_conflict in resolved_conflicts:
                    a = self._records.get(resolved_conflict.memory_a_id)
                    b = self._records.get(resolved_conflict.memory_b_id) or record
                    if a is None:
                        continue
                    self._explanations.append(
                        self._conflict_explainer.explain(
                            resolved_conflict, a, b, authority=flat_authority,
                        )
                    )

            # 5. Store the record
            self._records[record.id] = record
            if record.state == MemoryState.ACTIVE:
                self._index_record(record)
            self._write_count += 1

            # 6. Audit
            self._audit_log.append(
                AuditEntry(
                    operation="write",
                    agent_id=record.author_agent_id,
                    record_id=record.id,
                    details={
                        "conflict_count": len(conflicts),
                        "strategy": effective_strategy,
                    },
                )
            )

            # 7. Notify subscribers
            await self._notify_subscribers(record)

            # 8. Build structured result
            if conflicts:
                outcome = WriteOutcome.CONFLICT_RESOLVED
            elif record.state == MemoryState.SUPERSEDED:
                outcome = WriteOutcome.SUPERSEDED
            else:
                outcome = WriteOutcome.ACCEPTED
            self.last_write_result = WriteResult(
                record=record,
                outcome=outcome,
                conflicts=tuple(conflicts),
            )

            self._write_latency += time.perf_counter() - t_start
            return record, conflicts

    async def read(
        self,
        agent_id: str,
        query: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
        tags: Optional[list[str]] = None,
        limit: int = 10,
        include_disputed: bool = False,
        consistency: str = "eventual",
        as_of: Optional[datetime] = None,
    ) -> list[MemoryRecord]:
        """Read memories filtered by agent permissions, tags, and relevance.

        Args:
            as_of: If provided, only return records whose validity window
                (valid_from … valid_until) contains this instant. Defaults to
                "now", which is equivalent to the legacy behavior plus
                filtering out records whose valid_until has already passed.
        """
        t_start = time.perf_counter()
        now = as_of or datetime.now(timezone.utc)

        async with self._lock:
            # Two code paths:
            #   (a) embedding query: ask the vector index for a widened top-K,
            #       then apply state/tag/temporal filters. This lets HNSW
            #       skip the O(n) loop entirely.
            #   (b) non-embedding query: fall back to the O(n) scan plus
            #       (confidence, created_at) sort.
            if query_embedding and len(self._vector_index) > 0:
                want = max(limit * 8, 32)
                candidates = self._vector_index.search(
                    query_embedding, k=want, threshold=0.0,
                )
                results: list[MemoryRecord] = []
                for rid, _sim in candidates:
                    record = self._records.get(rid)
                    if record is None:
                        continue
                    if not self._passes_read_filter(record, tags, include_disputed, now):
                        continue
                    results.append(record)
                    if len(results) >= limit:
                        break
            else:
                results = []
                for record in self._records.values():
                    if not self._passes_read_filter(record, tags, include_disputed, now):
                        continue
                    results.append(record)
                results.sort(key=lambda r: (r.confidence, r.created_at), reverse=True)

            results = results[:limit]

            self._read_count += 1
            self._audit_log.append(
                AuditEntry(
                    operation="read",
                    agent_id=agent_id,
                    details={"query": query, "tags": tags, "results": len(results), "as_of": now.isoformat()},
                )
            )
            self._read_latency += time.perf_counter() - t_start
            return results

    async def update(
        self,
        record_id: str,
        agent_id: str,
        updates: dict,
        reason: str = "",
    ) -> MemoryRecord:
        """Update creates a new version, marks old as superseded."""
        async with self._lock:
            if record_id not in self._records:
                raise KeyError(f"Record {record_id} not found")

            old = self._records[record_id]

            # Build new record fields
            new_fields = {
                "id": str(uuid.uuid4()),
                "version": old.version + 1,
                "supersedes_id": old.id,
                "author_agent_id": agent_id,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "vector_clock": self._vector_clock.increment(agent_id),
            }
            if "content" in updates:
                if not isinstance(updates["content"], str):
                    raise ValueError("content must be a string")
                new_fields["content"] = updates["content"]
            if "tags" in updates:
                t = updates["tags"]
                new_fields["tags"] = tuple(t) if isinstance(t, list) else t
            if "confidence" in updates:
                conf = updates["confidence"]
                if not (0.0 <= conf <= 1.0):
                    raise ValueError(f"confidence must be between 0.0 and 1.0, got {conf}")
                new_fields["confidence"] = conf

            new_record = replace(old, **new_fields)

            # Mark old as superseded
            self._records[old.id] = replace(old, state=MemoryState.SUPERSEDED, superseded_by_id=new_record.id)
            self._records[new_record.id] = new_record
            self._deindex_record(old.id)
            self._index_record(new_record)

            self._audit_log.append(
                AuditEntry(
                    operation="update",
                    agent_id=agent_id,
                    record_id=new_record.id,
                    details={"supersedes": old.id, "reason": reason},
                )
            )

            return new_record

    async def forget(
        self,
        record_id: str,
        agent_id: str,
        reason: str = "gdpr_request",
        cascade: bool = True,
    ) -> bool:
        """GDPR-compliant forgetting."""
        async with self._lock:
            if record_id not in self._records:
                return False

            record = self._records[record_id]
            self._records[record_id] = replace(
                record,
                state=MemoryState.FORGOTTEN,
                content="[FORGOTTEN]",
                content_embedding=None,
            )
            self._deindex_record(record_id)
            if self._trust_engine is not None:
                self._trust_engine.record_event(record.author_agent_id, "forgotten")

            self._audit_log.append(
                AuditEntry(
                    operation="forget",
                    agent_id=agent_id,
                    record_id=record_id,
                    details={"reason": reason, "cascade": cascade},
                )
            )

            if cascade:
                self._cascade_forget(record_id, agent_id)

            return True

    def _cascade_forget(self, root_id: str, requesting_agent: str) -> None:
        """Recursively forget all records in the supersedes chain."""
        to_visit = [root_id]
        visited = set()
        while to_visit:
            current_id = to_visit.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            # Find all records that supersede this one (children)
            for rid, r in self._records.items():
                if r.supersedes_id == current_id and r.state != MemoryState.FORGOTTEN:
                    self._records[rid] = replace(
                        r,
                        state=MemoryState.FORGOTTEN,
                        content="[FORGOTTEN]",
                        content_embedding=None,
                    )
                    self._deindex_record(rid)
                    self._audit_log.append(
                        AuditEntry(
                            operation="cascade_forget",
                            agent_id=requesting_agent,
                            record_id=rid,
                            details={"root_id": root_id},
                        )
                    )
                    to_visit.append(rid)
            # Also follow superseded_by_id forward links
            current = self._records.get(current_id)
            if current and current.superseded_by_id:
                fwd = self._records.get(current.superseded_by_id)
                if fwd and fwd.state != MemoryState.FORGOTTEN:
                    self._records[current.superseded_by_id] = replace(
                        fwd,
                        state=MemoryState.FORGOTTEN,
                        content="[FORGOTTEN]",
                        content_embedding=None,
                    )
                    self._deindex_record(current.superseded_by_id)
                    self._audit_log.append(
                        AuditEntry(
                            operation="cascade_forget",
                            agent_id=requesting_agent,
                            record_id=current.superseded_by_id,
                            details={"root_id": root_id},
                        )
                    )
                    to_visit.append(current.superseded_by_id)

    async def get_conflicts(
        self,
        unresolved_only: bool = True,
    ) -> list[tuple[MemoryRecord, MemoryRecord]]:
        """Return memory conflict pairs."""
        pairs = []
        for conflict in self._conflicts:
            if unresolved_only and conflict.resolved:
                continue
            a = self._records.get(conflict.memory_a_id)
            b = self._records.get(conflict.memory_b_id)
            if a and b:
                pairs.append((a, b))
        return pairs

    async def get_audit_log(
        self,
        record_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        since: Optional[str] = None,
    ) -> list[dict]:
        """Return filtered audit trail."""
        entries = self._audit_log
        if record_id:
            entries = [e for e in entries if e.record_id == record_id]
        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]
        if since:
            cutoff = datetime.fromisoformat(since)
            entries = [e for e in entries if e.timestamp >= cutoff]
        return [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "operation": e.operation,
                "agent_id": e.agent_id,
                "record_id": e.record_id,
                "details": e.details,
            }
            for e in entries
        ]

    async def subscribe(
        self,
        agent_id: str,
        tags: Optional[list[str]] = None,
        callback: Optional[callable] = None,
    ) -> str:
        """Subscribe to memory change notifications."""
        sub_id = str(uuid.uuid4())
        self._subscriptions[sub_id] = {
            "agent_id": agent_id,
            "tags": tags,
            "callback": callback,
        }
        return sub_id

    async def stats(self) -> dict:
        """Return store statistics."""
        async with self._lock:
            active = sum(1 for r in self._records.values() if r.state == MemoryState.ACTIVE)
            avg_write = self._write_latency / self._write_count if self._write_count else 0
            avg_read = self._read_latency / self._read_count if self._read_count else 0
            return {
                "total_memories": len(self._records),
                "active_memories": active,
                "conflicts_pending": sum(1 for c in self._conflicts if not c.resolved),
                "total_writes": self._write_count,
                "total_reads": self._read_count,
                "audit_entries": len(self._audit_log),
                "edges": len(self._edges),
                "explanations": len(self._explanations),
                "entity_attribute_entries": len(self._entity_index) if self._entity_index else 0,
                "subscriptions": len(self._subscriptions),
                "average_write_latency": avg_write,
                "average_read_latency": avg_read,
            }

    async def list_records(
        self,
        states: Optional[list] = None,
        agent_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[MemoryRecord]:
        """Enumerate all records with optional filtering."""
        results = list(self._records.values())
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
        """Return raw ConflictEvent objects with optional resolved filter."""
        results = list(self._conflicts)
        if resolved is not None:
            results = [c for c in results if c.resolved == resolved]
        return results

    async def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        kind: Optional[EdgeKind] = None,
    ) -> list[MemoryEdge]:
        """Query the typed-edge graph.

        Edges currently emitted:
          * SUPERSEDES   — winner → loser on conflict or dedup.
          * DERIVED_FROM — merged record → inputs it absorbed.
          * CONTRADICTS  — two records flagged by ConflictDetector.
          * SUPPORTS     — reserved for callers to add via `add_edge`.
        """
        results = list(self._edges)
        if source_id is not None:
            results = [e for e in results if e.source_id == source_id]
        if target_id is not None:
            results = [e for e in results if e.target_id == target_id]
        if kind is not None:
            results = [e for e in results if e.kind == kind]
        return results

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        kind: EdgeKind,
        metadata: Optional[dict] = None,
    ) -> MemoryEdge:
        """Append an arbitrary typed edge (e.g. SUPPORTS) to the graph."""
        if source_id not in self._records:
            raise KeyError(f"Unknown source record: {source_id}")
        if target_id not in self._records:
            raise KeyError(f"Unknown target record: {target_id}")
        edge = MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            kind=kind,
            metadata=dict(metadata or {}),
        )
        self._edges.append(edge)
        return edge

    # --- Internal helpers ---

    def _passes_read_filter(
        self,
        record: MemoryRecord,
        tags: Optional[list[str]],
        include_disputed: bool,
        now: datetime,
    ) -> bool:
        """Common filter used by both read() code paths."""
        if record.state in (MemoryState.SUPERSEDED, MemoryState.RETRACTED, MemoryState.FORGOTTEN):
            return False
        if record.state == MemoryState.DISPUTED and not include_disputed:
            return False
        if tags and not set(tags) & set(record.tags):
            return False
        if record.valid_from and record.valid_from > now:
            return False
        if record.valid_until and record.valid_until < now:
            return False
        return True

    def _preprocess(self, record: MemoryRecord) -> MemoryRecord:
        """Apply normalizer + type classifier before a record is written.

        Both components are optional. Normalizer rewrites `content` and
        `tags` in place via `dataclasses.replace`. The classifier only fires
        when the caller accepted the default `MemoryType.SEMANTIC` but the
        heuristic says something else — it never overrides an explicit choice.
        """
        changes: dict = {}
        if self._normalizer is not None:
            norm_content = self._normalizer.normalize_text(record.content)
            norm_tags = self._normalizer.normalize_tags(record.tags)
            if norm_content != record.content:
                changes["content"] = norm_content
            if tuple(norm_tags) != tuple(record.tags):
                changes["tags"] = norm_tags
        if self._type_classifier is not None and record.memory_type == MemoryType.SEMANTIC:
            inferred = self._type_classifier.classify(record.content)
            if inferred is not MemoryType.SEMANTIC:
                changes["memory_type"] = inferred
        return replace(record, **changes) if changes else record

    def classify_intent(self, query: str):
        """Expose the intent classifier if one was supplied at construction.

        No lock needed: _intent_classifier is assigned once in __init__ and never reassigned.
        """
        if self._intent_classifier is None:
            return None
        return self._intent_classifier.classify(query)

    def trust_score(self, agent_id: str) -> Optional[float]:
        """Current trust score for `agent_id`, or None if no engine is attached.

        No lock needed: _trust_engine is assigned once in __init__ and never reassigned.
        """
        if self._trust_engine is None:
            return None
        return self._trust_engine.score(agent_id)

    async def lookup_entity_attribute(
        self,
        entity: Optional[str] = None,
        attribute: Optional[str] = None,
    ) -> list[MemoryRecord]:
        """Resolve an `EntityAttributeIndex` query to full records.

        Returns an empty list if no entity-attribute index was attached.
        """
        if self._entity_index is None:
            return []
        ids = self._entity_index.lookup(entity=entity, attribute=attribute)
        return [self._records[i] for i in ids if i in self._records]

    async def get_explanations(
        self,
        conflict_id: Optional[str] = None,
        memory_id: Optional[str] = None,
    ) -> list[ConflictExplanation]:
        """Return structured conflict explanations produced by the explainer."""
        results = list(self._explanations)
        if conflict_id is not None:
            results = [e for e in results if e.conflict_id == conflict_id]
        if memory_id is not None:
            results = [
                e for e in results
                if e.memory_a_id == memory_id or e.memory_b_id == memory_id
            ]
        return results

    def _index_record(self, record: MemoryRecord) -> None:
        """Add `record` to every attached index (vector + structural)."""
        if record.content_embedding is not None:
            try:
                self._vector_index.add(record.id, record.content_embedding)
            except ValueError:
                logger.debug("Vector index dim mismatch for record %s", record.id)
        if self._entity_index is not None:
            self._entity_index.add(record)

    def _deindex_record(self, record_id: str) -> None:
        """Remove `record_id` from every attached index. Safe to call blindly."""
        self._vector_index.remove(record_id)
        if self._entity_index is not None:
            self._entity_index.remove(record_id)

    def _find_duplicate(self, record: MemoryRecord) -> Optional[str]:
        """Find a semantic duplicate of the given record via the vector index.

        Uses the pluggable `VectorIndex` so dedup cost is O(log n) with an
        HNSW backend instead of O(n) brute force. The same-author / state
        checks that `SemanticDeduplicator` enforces are re-applied here
        because the index doesn't know about them.
        """
        if record.content_embedding is None or len(self._vector_index) == 0:
            return None

        candidates = self._vector_index.search(
            record.content_embedding,
            k=8,
            threshold=self._dedup_threshold,
            exclude_ids=[record.id],
        )
        for cand_id, _sim in candidates:
            existing = self._records.get(cand_id)
            if existing is None:
                continue
            if existing.state != MemoryState.ACTIVE:
                continue
            if existing.author_agent_id == record.author_agent_id:
                continue  # same-agent dedup is treated as a repeated-write bug
            return cand_id
        return None

    def _rank_by_embedding(
        self, records: list[MemoryRecord], query_embedding: list[float], limit: int = 10,
    ) -> list[MemoryRecord]:
        """Rank records by cosine similarity using the vector index.

        Only requests `limit * 4` candidates from the index (capped at the
        eligible set) so HNSW can actually take the fast path. Brute force
        is unaffected — returning fewer rows costs the same matmul.
        """
        if len(self._vector_index) == 0:
            return records

        eligible = {r.id for r in records}
        want = min(max(limit * 4, limit), len(eligible)) if eligible else limit
        ranked = self._vector_index.search(
            query_embedding,
            k=max(want, 1),
            threshold=0.0,
        )
        by_id = {r.id: r for r in records}
        out: list[MemoryRecord] = []
        for rid, _sim in ranked:
            if rid in eligible and rid in by_id:
                out.append(by_id[rid])
        # Records without embeddings were not in the index — append at the end.
        seen = {r.id for r in out}
        out.extend(r for r in records if r.id not in seen)
        return out

    async def _notify_subscribers(self, record: MemoryRecord) -> list[str]:
        """Notify subscribed agents about a new/changed memory.

        Returns a list of subscription IDs whose callbacks failed.
        """
        failed: list[str] = []
        for sub_id, sub in self._subscriptions.items():
            if sub["tags"] and not set(sub["tags"]) & set(record.tags):
                continue
            if sub["callback"]:
                try:
                    result = sub["callback"](record)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.warning(
                        "Subscriber %s callback failed for record %s: %s",
                        sub_id, record.id, exc,
                    )
                    failed.append(sub_id)
        return failed

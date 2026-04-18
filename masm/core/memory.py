""""
Shared memory model, conflict detection, resolution utilities, and audit logging.

This module defines:
- Immutable dataclass-based records for agent-shared memories.
- Conflict events and resolution strategies.
- Structured audit logging for all operations.
- Vector clock comparison and semantic similarity utilities.

Design goals:
- Immutable records with copy-on-write updates.
- Full provenance for every memory item.
- Explicit conflict events instead of silent overwrites.
- Structured audit logging for all operations.
- Strategy-based conflict resolution with safe fallback behavior.

Components:
- `MemoryType`: Enum for categorizing memories.
- `MemoryState`: Enum for tracking memory lifecycle states.
- `ConflictStrategy`: Enum for conflict resolution strategies.
- `MemoryRecord`: Immutable memory record with provenance and metadata.
- `ConflictEvent`: Represents a detected conflict between two memories.
- `AuditEntry`: Represents an audit trail entry for operations.
- Utility functions for vector clock comparison, cosine similarity, and creating audit/conflict events.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum, unique
from typing import Any, Mapping, Optional
from uuid import uuid4
import math


# -------------------- ENUMS --------------------

@unique
class MemoryType(Enum):
    """
    Categorize memories by cognitive type.

    Attributes:
        EPISODIC: Specific event or interaction that happened.
        SEMANTIC: Factual knowledge extracted from interactions.
        PROCEDURAL: Learned behavior, rule, or workflow step.
        IMPLICIT: Inferred knowledge derived from evidence or patterns.
    """
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    IMPLICIT = "implicit"

    def is_inferred(self) -> bool:
        """Check if the memory type represents inferred knowledge."""
        return self is MemoryType.IMPLICIT

    def requires_evidence(self) -> bool:
        """Check if the memory type requires evidence."""
        return self in {MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL}

    def default_confidence(self) -> float:
        """Get the default confidence value for the memory type."""
        return {
            MemoryType.EPISODIC: 0.95,
            MemoryType.SEMANTIC: 1.0,
            MemoryType.PROCEDURAL: 0.9,
            MemoryType.IMPLICIT: 0.75,
        }[self]


@unique
class MemoryState(Enum):
    """
    Track lifecycle state of a memory record.

    Attributes:
        ACTIVE: Currently valid and usable.
        SUPERSEDED: Replaced by a newer version.
        DISPUTED: Conflict detected, not yet resolved.
        RETRACTED: Explicitly invalidated.
        FORGOTTEN: Removed for compliance or deletion purposes.
        DEPRECATED: Still retained, but no longer relevant.
    """
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"
    RETRACTED = "retracted"
    FORGOTTEN = "forgotten"
    DEPRECATED = "deprecated"


@unique
class ConflictStrategy(Enum):
    """
    Define how conflicting memories are resolved.

    Attributes:
        LAST_WRITE_WINS: Timestamp-based resolution.
        SEMANTIC_MERGE: Merge or synthesize similar memories.
        AUTHORITY_RANK: Resolve by authority ranking.
        MANUAL: Escalate to human review.
        HYBRID: Combine multiple strategies.
        DEFAULT: Fallback strategy for unsupported cases.
    """
    LAST_WRITE_WINS = "lww"
    SEMANTIC_MERGE = "semantic_merge"
    AUTHORITY_RANK = "authority_rank"
    MANUAL = "manual"
    HYBRID = "hybrid"
    DEFAULT = "default"


@unique
class OperationStatus(Enum):
    """
    Track the outcome of an operation.

    Attributes:
        SUCCESS: Operation completed successfully.
        FAILURE: Operation failed.
        PENDING: Operation requires later completion.
        PARTIAL: Operation partially succeeded.
    """
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    PARTIAL = "partial"


@unique
class ConflictSeverity(Enum):
    """
    Express the importance of a conflict.

    Attributes:
        LOW: Minor or low-impact conflict.
        MEDIUM: Standard conflict.
        HIGH: High-impact conflict.
        CRITICAL: Urgent conflict with major impact.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@unique
class WriteOutcome(Enum):
    """
    Distinguish write results so callers know what happened.

    Attributes:
        ACCEPTED: Record was stored as-is.
        DEDUPLICATED: Record was a semantic duplicate; existing record kept.
        CONFLICT_RESOLVED: Conflict detected and resolved via strategy.
        SUPERSEDED: Record was written but immediately superseded by existing winner.
    """
    ACCEPTED = "accepted"
    DEDUPLICATED = "deduplicated"
    CONFLICT_RESOLVED = "conflict_resolved"
    SUPERSEDED = "superseded"


@unique
class EdgeKind(Enum):
    """
    Typed edges in the memory graph.

    Attributes:
        SUPERSEDES:   source replaces target in the version chain.
        SUPPORTS:     source provides evidence for target.
        CONTRADICTS:  source and target assert incompatible facts.
        DERIVED_FROM: source was synthesized/merged from target.
    """
    SUPERSEDES = "supersedes"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"


@unique
class ClockRelation(Enum):
    """
    Represent the relationship between two vector clocks.

    Attributes:
        BEFORE: Clock A happens before Clock B.
        AFTER: Clock A happens after Clock B.
        CONCURRENT: Clocks A and B are concurrent.
        EQUAL: Clocks A and B are equal.
    """
    BEFORE = "before"
    AFTER = "after"
    CONCURRENT = "concurrent"
    EQUAL = "equal"


# -------------------- DATACLASSES --------------------

@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """
    Atomic immutable record for shared memory between agents.

    Attributes:
        id: Unique memory identifier.
        content: Human-readable content of the memory.
        content_embedding: Optional embedding vector for semantic comparison.
        memory_type: Cognitive category of the memory.
        author_agent_id: Agent that created the record.
        confidence: Confidence score from 0.0 to 1.0.
        created_at: Creation timestamp in UTC.
        updated_at: Last update timestamp in UTC.
        valid_from: Start of validity window.
        valid_until: End of validity window.
        vector_clock: Per-agent causality clock.
        state: Lifecycle state.
        tags: Categorization labels for the memory.
        metadata: Additional structured metadata.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    content_embedding: Optional[tuple[float, ...]] = None
    memory_type: MemoryType = MemoryType.SEMANTIC
    author_agent_id: str = ""
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    vector_clock: Mapping[str, int] = field(default_factory=dict)
    state: MemoryState = MemoryState.ACTIVE
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1
    supersedes_id: Optional[str] = None
    superseded_by_id: Optional[str] = None

    def __post_init__(self):
        """Validate field constraints after initialization."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.valid_from and self.valid_until and self.valid_from > self.valid_until:
            raise ValueError("valid_from cannot be after valid_until")
        if not self.author_agent_id:
            raise ValueError("author_agent_id is required")
        if self.content_embedding is not None and len(self.content_embedding) == 0:
            raise ValueError("content_embedding cannot be empty")

    def with_updates(self, **changes) -> "MemoryRecord":
        """
        Create a new record with modifications applied.

        Args:
            **changes: Fields to update in the new record.

        Returns:
            A new `MemoryRecord` instance with the updates applied.
        """
        changes.setdefault("updated_at", datetime.now(timezone.utc))
        return replace(self, **changes)

    def conflicts_with(
        self,
        other: "MemoryRecord",
        similarity_threshold: float = 0.85,
    ) -> bool:
        """
        Detect whether two records potentially conflict.

        Args:
            other: The other `MemoryRecord` to compare.
            similarity_threshold: The threshold for semantic similarity.

        Returns:
            True if the records conflict, False otherwise.
        """
        # Same record or identical content — not a conflict
        if self.id == other.id or self.content == other.content:
            return False
        if self.state in (MemoryState.RETRACTED, MemoryState.FORGOTTEN):
            return False
        if not self._overlapping_validity(other):
            return False
        # No tag overlap — different topics, not a conflict
        if self.tags and other.tags and not (set(self.tags) & set(other.tags)):
            return False
        # Semantic evidence required. Two records with overlapping tags may
        # be restating the same fact, complementing it, or contradicting it —
        # embeddings are the signal that tells these apart. If neither record
        # carries an embedding we have no way to distinguish a genuine
        # conflict from coexisting facts on the same topic, so we refuse to
        # fabricate a conflict event. (This matters for multi-step agent
        # workflows where each step writes prose without embeddings; the
        # old behavior caused every next-step write to supersede the
        # previous step's facts via LWW.)
        similarity = self._embedding_similarity(other)
        if similarity is None:
            return False
        if similarity >= similarity_threshold:
            return False  # Near-duplicate, handled by dedup, not conflict
        return True

    def _overlapping_validity(self, other: "MemoryRecord") -> bool:
        """
        Check if the validity windows of two records overlap.

        Args:
            other: The other `MemoryRecord` to compare.

        Returns:
            True if the validity windows overlap, False otherwise.
        """
        start_a = self.valid_from or datetime.min.replace(tzinfo=timezone.utc)
        end_a = self.valid_until or datetime.max.replace(tzinfo=timezone.utc)
        start_b = other.valid_from or datetime.min.replace(tzinfo=timezone.utc)
        end_b = other.valid_until or datetime.max.replace(tzinfo=timezone.utc)
        return max(start_a, start_b) <= min(end_a, end_b)

    def _embedding_similarity(self, other: "MemoryRecord") -> Optional[float]:
        """
        Compute cosine similarity between embeddings when available.

        Args:
            other: The other `MemoryRecord` to compare.

        Returns:
            A float representing the cosine similarity, or None if embeddings are missing.
        """
        if self.content_embedding is None or other.content_embedding is None:
            return None
        return cosine_similarity(self.content_embedding, other.content_embedding)


@dataclass(frozen=True, slots=True)
class ConflictEvent:
    """
    Record a detected conflict between two memories.

    Attributes:
        id: Unique identifier for the conflict event.
        memory_a_id: ID of the first conflicting memory.
        memory_b_id: ID of the second conflicting memory.
        detected_at: Timestamp when the conflict was detected.
        resolved: Whether the conflict has been resolved.
        strategy: Conflict resolution strategy used.
        severity: Severity of the conflict.
        reasoning: Explanation for the conflict or resolution.
        resolved_memory_id: ID of the resolved memory, if applicable.
        status: Status of the conflict resolution process.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    memory_a_id: str = ""
    memory_b_id: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    strategy: ConflictStrategy = ConflictStrategy.DEFAULT
    severity: ConflictSeverity = ConflictSeverity.MEDIUM
    reasoning: Optional[str] = None
    resolved_memory_id: Optional[str] = None
    status: OperationStatus = OperationStatus.PENDING


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """
    Record a single audit-trail entry.

    Attributes:
        id: Unique identifier for the audit entry.
        timestamp: Timestamp of the operation.
        operation: Description of the operation performed.
        agent_id: ID of the agent performing the operation.
        record_id: ID of the memory record involved in the operation.
        status: Status of the operation.
        reason: Explanation for the operation or its outcome.
        details: Additional structured metadata about the operation.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operation: str = ""
    agent_id: str = ""
    record_id: Optional[str] = None
    status: OperationStatus = OperationStatus.SUCCESS
    reason: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryEdge:
    """
    Typed edge connecting two `MemoryRecord` nodes in the memory graph.

    Attributes:
        id: Unique identifier for the edge.
        source_id: Record ID on the "from" side of the edge.
        target_id: Record ID on the "to" side of the edge.
        kind: Semantic kind of edge (SUPERSEDES / SUPPORTS / CONTRADICTS / DERIVED_FROM).
        created_at: Timestamp the edge was recorded.
        metadata: Optional structured annotation (e.g. {"reason": "lww"}).
    """
    source_id: str = ""
    target_id: str = ""
    kind: EdgeKind = EdgeKind.SUPERSEDES
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WriteResult:
    """
    Structured result from a write operation.

    Attributes:
        record: The final stored record (may differ from input after merge/dedup).
        outcome: What happened during the write.
        conflicts: Any conflicts detected and resolved.
        duplicate_of: If deduplicated, the ID of the existing record that was kept.
    """
    record: MemoryRecord
    outcome: WriteOutcome
    conflicts: tuple[ConflictEvent, ...] = ()
    duplicate_of: Optional[str] = None


# -------------------- UTILITY FUNCTIONS --------------------

def compare_vector_clocks(clock1: Mapping[str, int], clock2: Mapping[str, int]) -> ClockRelation:
    """
    Compare two vector clocks and determine their relationship.

    Args:
        clock1: The first vector clock.
        clock2: The second vector clock.

    Returns:
        A `ClockRelation` indicating the relationship between the clocks.
    """
    nodes = set(clock1) | set(clock2)
    has_less = False
    has_greater = False
    for node in nodes:
        t1 = clock1.get(node, 0)
        t2 = clock2.get(node, 0)
        if t1 < t2:
            has_less = True
        elif t1 > t2:
            has_greater = True

    if not has_less and not has_greater:
        return ClockRelation.EQUAL
    if has_less and not has_greater:
        return ClockRelation.BEFORE
    if has_greater and not has_less:
        return ClockRelation.AFTER
    return ClockRelation.CONCURRENT


def cosine_similarity(
    a: Optional[tuple[float, ...]],
    b: Optional[tuple[float, ...]],
) -> Optional[float]:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        a: The first embedding vector.
        b: The second embedding vector.

    Returns:
        A float representing the cosine similarity, or None if inputs are invalid.
    """
    if a is None or b is None or len(a) != len(b) or len(a) == 0:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return None
    return dot / (norm_a * norm_b)


def create_audit_entry(
    operation: str,
    agent_id: str,
    record_id: Optional[str] = None,
    status: OperationStatus = OperationStatus.SUCCESS,
    reason: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> AuditEntry:
    """
    Create an audit trail entry for an operation.

    Args:
        operation: The operation performed (e.g., "write", "read").
        agent_id: The ID of the agent performing the operation.
        record_id: The ID of the memory record involved in the operation.
        status: The status of the operation (default: SUCCESS).
        reason: Explanation for the operation or its outcome.
        details: Additional structured metadata about the operation.

    Returns:
        An `AuditEntry` instance representing the operation.
    """
    return AuditEntry(
        operation=operation,
        agent_id=agent_id,
        record_id=record_id,
        status=status,
        reason=reason,
        details=details or {},
    )
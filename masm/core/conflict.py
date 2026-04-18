"""Conflict detection and resolution strategies for shared memories."""
from dataclasses import replace

from masm.core.memory import ConflictEvent, ConflictStrategy, MemoryRecord, MemoryState
from masm.core.clock import VectorClock
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ConflictDetector:
    """
    Detects conflicts between memory records using tag overlap,
    vector clocks, and optional embedding similarity.
    """

    def __init__(self, semantic_threshold: float = 0.85):
        self.semantic_threshold = semantic_threshold
        self.vclock = VectorClock()

    def detect(
        self,
        new_record: MemoryRecord,
        existing_records: list[MemoryRecord],
    ) -> list[ConflictEvent]:
        """
        Check a new record against existing records for conflicts.

        A conflict is detected when records share overlapping tags
        AND have different content.
        """
        conflicts = []
        for existing in existing_records:
            if existing.state != MemoryState.ACTIVE:
                continue
            if existing.id == new_record.id:
                continue
            if not new_record.conflicts_with(existing, self.semantic_threshold):
                continue

            logger.info(f"Conflict detected between {existing.id} and {new_record.id}")
            conflicts.append(
                ConflictEvent(
                    memory_a_id=existing.id,
                    memory_b_id=new_record.id,
                )
            )
        return conflicts


class ConflictResolver:
    """Resolves conflicts between memory records using configurable strategies.

    A `TrustEngine` may be attached at construction time; when present, its
    dynamic per-agent scores are used for AUTHORITY_RANK resolution (and take
    precedence over any static `agents` dict passed to `resolve`).
    """

    def __init__(self, trust_engine: Optional[object] = None):
        # Duck-typed: any object with `as_authority_dict() -> dict` works.
        self.trust_engine = trust_engine

    def resolve(
        self,
        memory_a: MemoryRecord,
        memory_b: MemoryRecord,
        strategy: ConflictStrategy = ConflictStrategy.LAST_WRITE_WINS,
        agents: Optional[dict] = None,
    ) -> tuple[MemoryRecord, str]:
        """
        Resolve a conflict between two memories.

        Returns:
            (winning_record, reasoning)
        """
        if strategy == ConflictStrategy.LAST_WRITE_WINS:
            return self._resolve_lww(memory_a, memory_b)
        elif strategy == ConflictStrategy.AUTHORITY_RANK:
            logger.info(f"Resolving conflict using strategy: {strategy}")
            # Prefer dynamic trust scores when a TrustEngine is attached.
            effective_agents = agents or {}
            if self.trust_engine is not None:
                try:
                    effective_agents = self.trust_engine.as_authority_dict()
                except AttributeError:
                    pass
            return self._resolve_authority(memory_a, memory_b, effective_agents)
        else:
            # Default to LWW for strategies that need external services
            return self._resolve_lww(memory_a, memory_b)

    def _resolve_lww(
        self, a: MemoryRecord, b: MemoryRecord
    ) -> tuple[MemoryRecord, str]:
        """Last-write-wins: the more recent memory wins."""
        if b.created_at >= a.created_at:
            winner, loser = b, a
        else:
            winner, loser = a, b
        return winner, f"LWW: {winner.id} is more recent"

    def _resolve_authority(
        self, a: MemoryRecord, b: MemoryRecord, agents: dict
    ) -> tuple[MemoryRecord, str]:
        """Higher-authority agent's memory wins.

        Tiebreakers (in order): author-reported confidence, then LWW.
        This matters in practice: static authority ranks (and dynamic trust
        scores freshly initialised at their prior) are often equal at the
        moment of first conflict, and without a tiebreaker `>=` silently
        prefers the newer write — re-introducing LWW through the back door.
        """
        rank_a = agents.get(a.author_agent_id, {}).get("authority_rank", 0)
        rank_b = agents.get(b.author_agent_id, {}).get("authority_rank", 0)
        if rank_a != rank_b:
            if rank_b > rank_a:
                winner, loser = b, a
            else:
                winner, loser = a, b
            return (
                winner,
                f"Authority: {winner.author_agent_id} ({max(rank_a, rank_b):.3f}) "
                f"outranks {loser.author_agent_id} ({min(rank_a, rank_b):.3f})",
            )
        # Tie on authority — use confidence.
        if a.confidence != b.confidence:
            winner, loser = (a, b) if a.confidence > b.confidence else (b, a)
            return (
                winner,
                f"Authority tie ({rank_a:.3f}); confidence "
                f"{winner.confidence:.2f} > {loser.confidence:.2f}",
            )
        # Final fallback: LWW.
        winner, loser = (b, a) if b.created_at >= a.created_at else (a, b)
        return winner, f"Authority tie; LWW picks {winner.id}"

    def resolve_custom(self, strategy_func: Callable, *args, **kwargs):
        return strategy_func(*args, **kwargs)

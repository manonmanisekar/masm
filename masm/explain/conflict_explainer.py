"""Structured conflict explanations.

The existing `ConflictEvent.reasoning` field holds a single free-text string
("LWW: <id> is more recent"). That's fine for humans skimming an audit log,
but it doesn't help a policy engine, a UI, or an eval harness that wants to
know *why* one side won.

`ConflictExplainer` produces a `ConflictExplanation` instead: a factor
breakdown (recency / confidence / authority / semantic similarity / tag
overlap / vector-clock relation) plus a one-line summary. The explainer is
deterministic and dependency-free — it inspects the records, not an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from masm.core.clock import VectorClock  # noqa: F401  # re-exported for callers
from masm.core.memory import (
    ConflictEvent,
    ConflictStrategy,
    MemoryRecord,
    compare_vector_clocks,
    cosine_similarity,
    ClockRelation,
)


@dataclass(frozen=True)
class ExplanationFactor:
    """A single dimension contributing to the conflict outcome.

    Attributes:
        name: Short factor identifier ("recency", "confidence", ...).
        description: Human-readable sentence.
        value_a: Factor value for memory A (may be None if not applicable).
        value_b: Factor value for memory B.
        favors: "a", "b", or "tie" — which side this factor pushed toward.
        weight: Relative importance hint (0–1) used when ranking factors.
    """
    name: str
    description: str
    value_a: Optional[object] = None
    value_b: Optional[object] = None
    favors: str = "tie"
    weight: float = 0.5


@dataclass(frozen=True)
class ConflictExplanation:
    """Structured record of why one memory won a conflict."""

    conflict_id: str
    memory_a_id: str
    memory_b_id: str
    winner_id: Optional[str]
    strategy: ConflictStrategy
    factors: tuple[ExplanationFactor, ...] = field(default_factory=tuple)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "conflict_id": self.conflict_id,
            "memory_a_id": self.memory_a_id,
            "memory_b_id": self.memory_b_id,
            "winner_id": self.winner_id,
            "strategy": self.strategy.value,
            "summary": self.summary,
            "factors": [
                {
                    "name": f.name,
                    "description": f.description,
                    "value_a": f.value_a,
                    "value_b": f.value_b,
                    "favors": f.favors,
                    "weight": f.weight,
                }
                for f in self.factors
            ],
        }


class ConflictExplainer:
    """Generate structured explanations for conflict-resolution decisions.

    Usage:

        explainer = ConflictExplainer()
        explanation = explainer.explain(conflict_event, record_a, record_b)
        print(explanation.summary)
        print(explanation.to_dict())
    """

    # Factor weights — used to surface the "most important" reason in summaries.
    WEIGHT = {
        "recency": 0.35,
        "confidence": 0.25,
        "authority": 0.20,
        "semantic": 0.10,
        "tag_overlap": 0.05,
        "vector_clock": 0.05,
    }

    def explain(
        self,
        conflict: ConflictEvent,
        memory_a: MemoryRecord,
        memory_b: MemoryRecord,
        authority: Optional[dict[str, float]] = None,
    ) -> ConflictExplanation:
        """Return a structured explanation.

        Args:
            conflict: The `ConflictEvent` (resolved or unresolved).
            memory_a, memory_b: The two records referenced by the event.
            authority: Optional mapping of agent_id → authority score
                (e.g. `TrustEngine.score(...)` output). Enables the
                authority factor.
        """
        factors: list[ExplanationFactor] = []
        factors.append(self._recency(memory_a, memory_b))
        factors.append(self._confidence(memory_a, memory_b))
        if authority is not None:
            factors.append(self._authority(memory_a, memory_b, authority))
        factors.append(self._semantic(memory_a, memory_b))
        factors.append(self._tag_overlap(memory_a, memory_b))
        factors.append(self._vector_clock(memory_a, memory_b))

        winner_id = conflict.resolved_memory_id
        summary = self._summarize(
            memory_a, memory_b, conflict.strategy, winner_id, factors
        )

        return ConflictExplanation(
            conflict_id=conflict.id,
            memory_a_id=memory_a.id,
            memory_b_id=memory_b.id,
            winner_id=winner_id,
            strategy=conflict.strategy,
            factors=tuple(factors),
            summary=summary,
        )

    # ---- Individual factors ----

    def _recency(self, a: MemoryRecord, b: MemoryRecord) -> ExplanationFactor:
        favors = "a" if a.created_at > b.created_at else "b" if b.created_at > a.created_at else "tie"
        return ExplanationFactor(
            name="recency",
            description="Which record was written more recently.",
            value_a=a.created_at.isoformat(),
            value_b=b.created_at.isoformat(),
            favors=favors,
            weight=self.WEIGHT["recency"],
        )

    def _confidence(self, a: MemoryRecord, b: MemoryRecord) -> ExplanationFactor:
        favors = "a" if a.confidence > b.confidence else "b" if b.confidence > a.confidence else "tie"
        return ExplanationFactor(
            name="confidence",
            description="Author-reported confidence on each record.",
            value_a=a.confidence,
            value_b=b.confidence,
            favors=favors,
            weight=self.WEIGHT["confidence"],
        )

    def _authority(
        self,
        a: MemoryRecord,
        b: MemoryRecord,
        authority: dict[str, float],
    ) -> ExplanationFactor:
        ra = authority.get(a.author_agent_id, 0.0)
        rb = authority.get(b.author_agent_id, 0.0)
        favors = "a" if ra > rb else "b" if rb > ra else "tie"
        return ExplanationFactor(
            name="authority",
            description="Dynamic trust / authority of each authoring agent.",
            value_a=ra,
            value_b=rb,
            favors=favors,
            weight=self.WEIGHT["authority"],
        )

    def _semantic(self, a: MemoryRecord, b: MemoryRecord) -> ExplanationFactor:
        sim = cosine_similarity(a.content_embedding, b.content_embedding)
        return ExplanationFactor(
            name="semantic",
            description="Cosine similarity between the two embeddings — low similarity "
                        "suggests a genuine disagreement rather than paraphrase.",
            value_a=sim,
            value_b=sim,
            favors="tie",
            weight=self.WEIGHT["semantic"],
        )

    def _tag_overlap(self, a: MemoryRecord, b: MemoryRecord) -> ExplanationFactor:
        overlap = sorted(set(a.tags) & set(b.tags))
        return ExplanationFactor(
            name="tag_overlap",
            description="Shared tags indicate the records are on the same topic.",
            value_a=list(a.tags),
            value_b=list(b.tags),
            favors="tie",
            weight=self.WEIGHT["tag_overlap"] + (0.0 if overlap else -0.05),
        )

    def _vector_clock(self, a: MemoryRecord, b: MemoryRecord) -> ExplanationFactor:
        rel = compare_vector_clocks(a.vector_clock, b.vector_clock)
        favors = "tie"
        if rel == ClockRelation.BEFORE:
            favors = "b"  # B is causally after A → B is newer
        elif rel == ClockRelation.AFTER:
            favors = "a"
        return ExplanationFactor(
            name="vector_clock",
            description="Causal relationship between the two writes.",
            value_a=dict(a.vector_clock),
            value_b=dict(b.vector_clock),
            favors=favors,
            weight=self.WEIGHT["vector_clock"],
        )

    # ---- Summary ----

    def _summarize(
        self,
        a: MemoryRecord,
        b: MemoryRecord,
        strategy: ConflictStrategy,
        winner_id: Optional[str],
        factors: list[ExplanationFactor],
    ) -> str:
        if winner_id is None:
            return f"Unresolved conflict ({strategy.value}) between {a.id[:8]} and {b.id[:8]}."
        winner_side = "A" if winner_id == a.id else "B"
        # Pick the highest-weight factor that actually favored the winner.
        relevant = [
            f for f in factors
            if f.favors == winner_side.lower()
        ]
        relevant.sort(key=lambda f: f.weight, reverse=True)
        top = relevant[0] if relevant else None
        if top is None:
            reason = f"strategy '{strategy.value}'"
        else:
            reason = f"stronger {top.name} ({top.description.rstrip('.')})"
        return (
            f"Memory {winner_side} ({winner_id[:8]}) won via {strategy.value}: "
            f"{reason}."
        )

"""Per-agent relevance filtering — each agent only gets memories relevant to their role."""

import numpy as np
from typing import Optional

from masm.core.agent import Agent
from masm.core.memory import MemoryRecord


class RelevanceScorer:
    """
    Scores how relevant a memory is to a specific agent based on:
    1. Tag overlap with agent's interests
    2. Semantic similarity (embedding-based)
    3. Recency weighting
    4. Confidence weighting
    """

    def __init__(
        self,
        tag_weight: float = 0.3,
        semantic_weight: float = 0.4,
        recency_weight: float = 0.2,
        confidence_weight: float = 0.1,
    ):
        self.tag_weight = tag_weight
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self.confidence_weight = confidence_weight

    def score(
        self,
        record: MemoryRecord,
        agent: Agent,
        query_embedding: Optional[list[float]] = None,
        now: Optional[float] = None,
    ) -> float:
        """Compute a 0.0-1.0 relevance score for this record relative to this agent."""
        scores = []

        # Tag overlap
        if agent.tags_of_interest and record.tags:
            overlap = len(set(agent.tags_of_interest) & set(record.tags))
            max_possible = max(len(agent.tags_of_interest), 1)
            scores.append(self.tag_weight * (overlap / max_possible))
        else:
            scores.append(0.0)

        # Semantic similarity
        if query_embedding and record.content_embedding:
            q = np.array(query_embedding)
            e = np.array(record.content_embedding)
            sim = float(np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-10))
            scores.append(self.semantic_weight * max(sim, 0.0))
        else:
            scores.append(0.0)

        # Recency (decay over time — newer is better)
        if now is not None:
            age_seconds = now - record.created_at.timestamp()
            decay = 1.0 / (1.0 + age_seconds / 3600.0)  # Half-life of ~1 hour
            scores.append(self.recency_weight * decay)
        else:
            scores.append(self.recency_weight * 0.5)  # Default mid-range

        # Confidence
        scores.append(self.confidence_weight * record.confidence)

        return sum(scores)

    def filter_relevant(
        self,
        records: list[MemoryRecord],
        agent: Agent,
        query_embedding: Optional[list[float]] = None,
        threshold: float = 0.1,
        limit: int = 10,
    ) -> list[tuple[MemoryRecord, float]]:
        """Return records above the relevance threshold, sorted by score."""
        scored = [
            (record, self.score(record, agent, query_embedding))
            for record in records
        ]
        scored = [(r, s) for r, s in scored if s >= threshold]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

"""Semantic relevance scoring — determines which memories matter to which agents."""

import numpy as np
from typing import Optional

from masm.core.agent import Agent
from masm.core.memory import MemoryRecord


class SemanticRelevanceScorer:
    """
    Scores memory relevance as a weighted sum of four components:
      1. S_sem  — cosine similarity between query and memory embedding
      2. S_tag  — overlap between agent.tags_of_interest and record.tags
      3. S_temp — recency decay 1 / (1 + age_hours)
      4. κ      — record confidence in [0, 1]

        R(m, A, q) = α·S_sem + γ·S_tag + β·S_temp + δ·κ(m)

    Defaults: α=0.50, γ=0.25, β=0.15, δ=0.10.

    Requires the 'semantic' extra for embedding generation.
    Falls back to tag + recency + confidence scoring without embeddings.
    """

    def __init__(
        self,
        embedding_fn: Optional[callable] = None,
        similarity_weight: float = 0.5,
        tag_weight: float = 0.25,
        recency_weight: float = 0.15,
        confidence_weight: float = 0.10,
    ):
        self.embedding_fn = embedding_fn
        self.similarity_weight = similarity_weight
        self.tag_weight = tag_weight
        self.recency_weight = recency_weight
        self.confidence_weight = confidence_weight

    def score(
        self,
        record: MemoryRecord,
        agent: Agent,
        query: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
        now_ts: Optional[float] = None,
    ) -> float:
        """Compute relevance score for a record relative to an agent and query."""
        components = []

        # 1. Semantic similarity to query
        if query_embedding and record.content_embedding:
            sim = self._cosine_sim(query_embedding, record.content_embedding)
            components.append(self.similarity_weight * max(sim, 0.0))
        elif self.embedding_fn and query and record.content_embedding:
            qe = self.embedding_fn(query)
            sim = self._cosine_sim(qe, record.content_embedding)
            components.append(self.similarity_weight * max(sim, 0.0))
        else:
            components.append(0.0)

        # 2. Tag overlap
        if agent.tags_of_interest and record.tags:
            overlap = len(set(agent.tags_of_interest) & set(record.tags))
            max_tags = max(len(agent.tags_of_interest), 1)
            components.append(self.tag_weight * (overlap / max_tags))
        else:
            components.append(0.0)

        # 3. Recency
        if now_ts is not None:
            age_secs = max(now_ts - record.created_at.timestamp(), 0)
            decay = 1.0 / (1.0 + age_secs / 3600.0)
            components.append(self.recency_weight * decay)
        else:
            components.append(self.recency_weight * 0.5)

        # 4. Confidence
        components.append(self.confidence_weight * record.confidence)

        return sum(components)

    def rank(
        self,
        records: list[MemoryRecord],
        agent: Agent,
        query: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Rank and filter records by relevance."""
        scored = [
            (r, self.score(r, agent, query, query_embedding))
            for r in records
        ]
        scored = [(r, s) for r, s in scored if s >= threshold]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        a_np, b_np = np.array(a), np.array(b)
        denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
        if denom < 1e-10:
            return 0.0
        return float(np.dot(a_np, b_np) / denom)

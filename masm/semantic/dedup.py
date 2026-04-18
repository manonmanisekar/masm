"""Semantic deduplication — detect near-duplicate memories using embedding similarity."""

import numpy as np
from typing import Optional

from masm.core.memory import MemoryRecord


class SemanticDeduplicator:
    """
    Detect semantically duplicate memories using embedding similarity.

    Two memories are duplicates when both hold:
      1. Cosine similarity between their embeddings is >= threshold (default 0.92).
      2. They were authored by *different* agents. Same-agent near-duplicates
         are treated as a repeated-write bug elsewhere in the pipeline, not a
         cross-agent merge candidate.

    Note: tag overlap is deliberately NOT required — two agents may describe
    the same fact with non-overlapping tag sets, and embedding similarity is
    the authoritative signal. If you want tag-gated dedup, filter the
    `existing_records` list before calling `find_duplicates`.
    """

    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_np, b_np = np.array(a), np.array(b)
        norm_product = np.linalg.norm(a_np) * np.linalg.norm(b_np)
        if norm_product < 1e-10:
            return 0.0
        return float(np.dot(a_np, b_np) / norm_product)

    def find_duplicates(
        self,
        new_record: MemoryRecord,
        existing_records: list[MemoryRecord],
        embedding_fn: Optional[callable] = None,
    ) -> list[tuple[str, float]]:
        """
        Find existing records that are semantic duplicates of new_record.
        Returns list of (record_id, similarity_score) pairs.
        """
        if new_record.content_embedding is None:
            if embedding_fn is None:
                raise ValueError("No embedding available and no embedding_fn provided")
            from dataclasses import replace
            new_record = replace(new_record, content_embedding=tuple(embedding_fn(new_record.content)))

        duplicates = []
        for existing in existing_records:
            if existing.id == new_record.id:
                continue
            if existing.author_agent_id == new_record.author_agent_id:
                continue  # Same agent — not a multi-agent dedup case
            if existing.content_embedding is None:
                continue

            sim = self.cosine_similarity(
                new_record.content_embedding,
                existing.content_embedding,
            )
            if sim >= self.threshold:
                duplicates.append((existing.id, sim))

        return sorted(duplicates, key=lambda x: x[1], reverse=True)

    def is_duplicate(
        self,
        a: MemoryRecord,
        b: MemoryRecord,
    ) -> tuple[bool, float]:
        """Check if two specific records are duplicates. Returns (is_dup, similarity)."""
        if a.content_embedding is None or b.content_embedding is None:
            return False, 0.0

        sim = self.cosine_similarity(a.content_embedding, b.content_embedding)
        return sim >= self.threshold, sim

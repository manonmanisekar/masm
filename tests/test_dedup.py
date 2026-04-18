"""Tests for semantic deduplication."""

import pytest
import numpy as np
from masm.semantic.dedup import SemanticDeduplicator
from masm.core.memory import MemoryRecord


class TestSemanticDeduplicator:
    def _make_embedding(self, seed: int, dim: int = 64) -> list[float]:
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim)
        return (vec / np.linalg.norm(vec)).tolist()

    def _make_similar(self, base: list[float], noise: float = 0.03) -> list[float]:
        rng = np.random.RandomState(42)
        vec = np.array(base) + rng.randn(len(base)) * noise
        return (vec / np.linalg.norm(vec)).tolist()

    def test_cosine_similarity_identical(self):
        dedup = SemanticDeduplicator()
        vec = self._make_embedding(1)
        assert dedup.cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_orthogonal(self):
        dedup = SemanticDeduplicator()
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert dedup.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_finds_duplicate_above_threshold(self):
        dedup = SemanticDeduplicator(threshold=0.90)
        base_emb = self._make_embedding(1)
        similar_emb = self._make_similar(base_emb, noise=0.02)

        existing = MemoryRecord(
            content="original fact",
            content_embedding=base_emb,
            author_agent_id="agent_a",
        )
        new = MemoryRecord(
            content="same fact paraphrased",
            content_embedding=similar_emb,
            author_agent_id="agent_b",
        )

        dupes = dedup.find_duplicates(new, [existing])
        assert len(dupes) > 0
        assert dupes[0][1] >= 0.90

    def test_no_duplicate_below_threshold(self):
        dedup = SemanticDeduplicator(threshold=0.95)
        a = self._make_embedding(1)
        b = self._make_embedding(999)  # Completely different

        existing = MemoryRecord(
            content="fact A", content_embedding=a, author_agent_id="agent_a"
        )
        new = MemoryRecord(
            content="fact B", content_embedding=b, author_agent_id="agent_b"
        )

        dupes = dedup.find_duplicates(new, [existing])
        assert len(dupes) == 0

    def test_skips_same_agent(self):
        dedup = SemanticDeduplicator(threshold=0.5)
        emb = self._make_embedding(1)

        existing = MemoryRecord(
            content="fact", content_embedding=emb, author_agent_id="agent_a"
        )
        new = MemoryRecord(
            content="same fact", content_embedding=emb, author_agent_id="agent_a"
        )

        dupes = dedup.find_duplicates(new, [existing])
        assert len(dupes) == 0  # Same agent — not a multi-agent dedup case

    def test_is_duplicate_check(self):
        dedup = SemanticDeduplicator(threshold=0.90)
        base = self._make_embedding(1)
        similar = self._make_similar(base, noise=0.02)

        a = MemoryRecord(content="a", content_embedding=base, author_agent_id="x")
        b = MemoryRecord(content="b", content_embedding=similar, author_agent_id="y")

        is_dup, sim = dedup.is_duplicate(a, b)
        assert is_dup is True
        assert sim >= 0.90

    def test_no_embedding_raises(self):
        dedup = SemanticDeduplicator()
        new = MemoryRecord(content="no embedding", author_agent_id="a")
        existing = MemoryRecord(content="has embedding", content_embedding=[0.1, 0.2], author_agent_id="b")

        with pytest.raises(ValueError, match="No embedding"):
            dedup.find_duplicates(new, [existing])

    def test_embedding_fn_callback(self):
        dedup = SemanticDeduplicator(threshold=0.5)

        def fake_embed(text):
            return [float(ord(c)) for c in text[:64]]

        new = MemoryRecord(content="hello", author_agent_id="a")
        existing = MemoryRecord(
            content="world",
            content_embedding=[float(ord(c)) for c in "world"[:64]],
            author_agent_id="b",
        )

        # Should not raise — uses embedding_fn
        dupes = dedup.find_duplicates(new, [existing], embedding_fn=fake_embed)
        assert isinstance(dupes, list)

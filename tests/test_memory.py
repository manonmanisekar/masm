"""Tests for core memory data model."""

import pytest
from masm.core.memory import MemoryRecord, MemoryType, MemoryState, ConflictEvent


class TestMemoryRecord:
    def test_default_creation(self):
        record = MemoryRecord(content="test fact", author_agent_id="agent_1")
        assert record.content == "test fact"
        assert record.author_agent_id == "agent_1"
        assert record.state == MemoryState.ACTIVE
        assert record.memory_type == MemoryType.SEMANTIC
        assert record.version == 1
        assert record.id  # Should auto-generate

    def test_unique_ids(self):
        r1 = MemoryRecord(author_agent_id="a")
        r2 = MemoryRecord(author_agent_id="b")
        assert r1.id != r2.id

    def test_conflicts_with_same_tags_different_content(self):
        # Embeddings are required to distinguish contradiction from paraphrase;
        # dissimilar embeddings + same tags → conflict.
        r1 = MemoryRecord(
            content="budget is 50k",
            content_embedding=[1.0, 0.0, 0.0],
            tags=["budget"],
            author_agent_id="a",
        )
        r2 = MemoryRecord(
            content="budget is 75k",
            content_embedding=[0.0, 1.0, 0.0],
            tags=["budget"],
            author_agent_id="b",
        )
        assert r1.conflicts_with(r2)

    def test_no_conflict_without_embedding_evidence(self):
        # Without embeddings we cannot tell contradiction from coexistence,
        # so conflicts_with() refuses to fabricate a conflict event.
        r1 = MemoryRecord(content="budget is 50k", tags=["budget"], author_agent_id="a")
        r2 = MemoryRecord(content="budget is 75k", tags=["budget"], author_agent_id="b")
        assert not r1.conflicts_with(r2)

    def test_no_conflict_different_tags(self):
        r1 = MemoryRecord(content="budget is 50k", tags=["budget"], author_agent_id="a")
        r2 = MemoryRecord(content="team size is 10", tags=["team"], author_agent_id="b")
        assert not r1.conflicts_with(r2)

    def test_no_conflict_same_content(self):
        r1 = MemoryRecord(content="budget is 50k", tags=["budget"], author_agent_id="a")
        r2 = MemoryRecord(content="budget is 50k", tags=["budget"], author_agent_id="b")
        assert not r1.conflicts_with(r2)  # Duplicate, not conflict

    def test_no_conflict_with_self(self):
        r1 = MemoryRecord(content="test", tags=["t"], author_agent_id="a")
        assert not r1.conflicts_with(r1)

    def test_with_updates(self):
        r1 = MemoryRecord(content="original", tags=["t"], author_agent_id="a")
        r2 = r1.with_updates(content="modified")
        assert r2.content == "modified"
        assert list(r2.tags) == ["t"]
        assert r2.id == r1.id  # with_updates keeps same id unless overridden

    def test_with_updates_new_id(self):
        r1 = MemoryRecord(content="original", author_agent_id="a")
        r2 = r1.with_updates(id="new_id", content="new")
        assert r2.id == "new_id"
        assert r2.content == "new"

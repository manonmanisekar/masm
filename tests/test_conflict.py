"""Tests for conflict detection and resolution."""

import pytest
from masm.core.conflict import ConflictDetector, ConflictResolver
from masm.core.memory import MemoryRecord, MemoryState, ConflictStrategy


class TestConflictDetector:
    def test_detects_tag_based_conflict(self):
        detector = ConflictDetector()
        existing = MemoryRecord(
            content="budget is 50k",
            content_embedding=[1.0, 0.0, 0.0],
            tags=["budget"],
            author_agent_id="alice",
        )
        new = MemoryRecord(
            content="budget is 75k",
            content_embedding=[0.0, 1.0, 0.0],
            tags=["budget"],
            author_agent_id="bob",
        )
        conflicts = detector.detect(new, [existing])
        assert len(conflicts) == 1

    def test_no_conflict_different_tags(self):
        detector = ConflictDetector()
        existing = MemoryRecord(
            content="budget is 50k", tags=["budget"], author_agent_id="alice"
        )
        new = MemoryRecord(
            content="team is 10 people", tags=["team"], author_agent_id="bob"
        )
        conflicts = detector.detect(new, [existing])
        assert len(conflicts) == 0

    def test_ignores_superseded_records(self):
        detector = ConflictDetector()
        existing = MemoryRecord(
            content="old fact",
            tags=["fact"],
            author_agent_id="alice",
            state=MemoryState.SUPERSEDED,
        )
        new = MemoryRecord(
            content="new fact", tags=["fact"], author_agent_id="bob"
        )
        conflicts = detector.detect(new, [existing])
        assert len(conflicts) == 0


class TestConflictResolver:
    def test_lww_picks_more_recent(self):
        from datetime import datetime, timezone, timedelta

        resolver = ConflictResolver()
        older = MemoryRecord(
            content="old",
            author_agent_id="alice",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        newer = MemoryRecord(
            content="new",
            author_agent_id="bob",
            created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )
        winner, reason = resolver.resolve(older, newer, ConflictStrategy.LAST_WRITE_WINS)
        assert winner.content == "new"
        assert "LWW" in reason

    def test_authority_rank(self):
        resolver = ConflictResolver()
        junior = MemoryRecord(content="wrong", author_agent_id="junior")
        senior = MemoryRecord(content="right", author_agent_id="senior")
        agents = {
            "junior": {"authority_rank": 1},
            "senior": {"authority_rank": 10},
        }
        winner, reason = resolver.resolve(
            junior, senior, ConflictStrategy.AUTHORITY_RANK, agents
        )
        assert winner.content == "right"

    def test_manual_returns_reasoning(self):
        resolver = ConflictResolver()
        a = MemoryRecord(content="a", author_agent_id="alice")
        b = MemoryRecord(content="b", author_agent_id="bob")
        winner, reason = resolver.resolve(a, b, ConflictStrategy.MANUAL)
        # Manual strategy falls through to LWW in the current implementation
        assert winner is not None
        assert reason != ""

    def test_detect_conflict_basic(self):
        detector = ConflictDetector()
        new_record = MemoryRecord(
            tags=["tag1"],
            content="content1",
            content_embedding=[1.0, 0.0, 0.0],
            author_agent_id="a",
        )
        existing_record = MemoryRecord(
            tags=["tag1"],
            content="content2",
            content_embedding=[0.0, 1.0, 0.0],
            author_agent_id="b",
        )
        conflicts = detector.detect(new_record, [existing_record])
        assert len(conflicts) == 1

    def test_resolve_lww_basic(self):
        from datetime import datetime, timezone
        resolver = ConflictResolver()
        record_a = MemoryRecord(content="a", author_agent_id="x",
                                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        record_b = MemoryRecord(content="b", author_agent_id="y",
                                created_at=datetime(2025, 6, 1, tzinfo=timezone.utc))
        winner, reasoning = resolver.resolve(record_a, record_b, ConflictStrategy.LAST_WRITE_WINS)
        assert winner.id == record_b.id
        assert "LWW" in reasoning
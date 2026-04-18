"""Tests for InMemorySharedStore."""

import pytest
from masm.store.in_memory import InMemorySharedStore
from masm.core.memory import MemoryRecord, MemoryState


@pytest.mark.asyncio
class TestInMemoryStore:
    async def test_write_and_read(self, store):
        record = MemoryRecord(
            content="test fact",
            author_agent_id="alice",
            tags=["test"],
        )
        written, conflicts = await store.write(record)
        assert written.id == record.id
        assert conflicts == []

        results = await store.read(agent_id="bob", tags=["test"])
        assert len(results) == 1
        assert results[0].content == "test fact"

    async def test_conflict_detection(self, store):
        r1 = MemoryRecord(
            content="budget is 50k",
            content_embedding=[1.0, 0.0, 0.0],
            author_agent_id="alice",
            tags=["budget"],
            confidence=0.9,
        )
        await store.write(r1)

        r2 = MemoryRecord(
            content="budget is 75k",
            content_embedding=[0.0, 1.0, 0.0],
            author_agent_id="bob",
            tags=["budget"],
            confidence=0.9,
        )
        _, conflicts = await store.write(r2)
        assert len(conflicts) > 0

    async def test_update_creates_new_version(self, store):
        r1 = MemoryRecord(
            content="old fact",
            author_agent_id="alice",
            tags=["fact"],
        )
        await store.write(r1)

        r2 = await store.update(r1.id, "bob", {"content": "updated fact"}, reason="correction")
        assert r2.version == 2
        assert r2.supersedes_id == r1.id
        assert r2.content == "updated fact"

        # Old version should be superseded
        results = await store.read(agent_id="carol", tags=["fact"])
        assert all(r.content != "old fact" for r in results)

    async def test_forget_gdpr(self, store):
        record = MemoryRecord(
            content="sensitive data",
            author_agent_id="alice",
            tags=["pii"],
        )
        await store.write(record)

        success = await store.forget(record.id, "admin", reason="gdpr_request")
        assert success

        results = await store.read(agent_id="bob", tags=["pii"])
        assert len(results) == 0  # Forgotten records should not appear

    async def test_read_filters_by_tags(self, store):
        await store.write(MemoryRecord(
            content="fact about revenue", author_agent_id="alice", tags=["revenue"],
        ))
        await store.write(MemoryRecord(
            content="fact about team", author_agent_id="alice", tags=["team"],
        ))

        revenue = await store.read(agent_id="bob", tags=["revenue"])
        assert len(revenue) == 1
        assert revenue[0].content == "fact about revenue"

    async def test_read_filters_by_tags(self, store):
        await store.write(MemoryRecord(
            content="secret stuff",
            author_agent_id="alice",
            tags=["secret"],
        ))
        await store.write(MemoryRecord(
            content="public stuff",
            author_agent_id="alice",
            tags=["public"],
        ))

        # Tag-filtered read
        results = await store.read(agent_id="bob", tags=["secret"])
        assert len(results) == 1
        assert results[0].content == "secret stuff"

        # Different tag returns different results
        results = await store.read(agent_id="bob", tags=["public"])
        assert len(results) == 1
        assert results[0].content == "public stuff"

    async def test_stats(self, store):
        await store.write(MemoryRecord(content="fact", author_agent_id="alice", tags=["t"]))
        stats = await store.stats()
        assert stats["total_memories"] == 1
        assert stats["active_memories"] == 1
        assert stats["total_writes"] == 1

    async def test_audit_log(self, store):
        record = MemoryRecord(content="fact", author_agent_id="alice", tags=["t"])
        await store.write(record)
        await store.read(agent_id="bob")

        log = await store.get_audit_log()
        assert len(log) == 2
        assert log[0]["operation"] == "write"
        assert log[1]["operation"] == "read"

    async def test_subscribe_and_notify(self, store):
        notifications = []

        def on_change(record):
            notifications.append(record.content)

        await store.subscribe("bob", tags=["alerts"], callback=on_change)

        await store.write(MemoryRecord(
            content="alert!", author_agent_id="alice", tags=["alerts"],
        ))
        assert len(notifications) == 1
        assert notifications[0] == "alert!"

        # Unrelated tag should not trigger
        await store.write(MemoryRecord(
            content="no alert", author_agent_id="alice", tags=["other"],
        ))
        assert len(notifications) == 1

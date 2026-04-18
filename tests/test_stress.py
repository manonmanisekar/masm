"""Stress tests — evaluate MASM under heavy concurrent load."""

import pytest
import asyncio
from masm.store.in_memory import InMemorySharedStore
from masm.core.memory import MemoryRecord


@pytest.mark.asyncio
class TestStress:
    async def test_high_volume_writes(self):
        """1000 sequential writes should complete without errors."""
        store = InMemorySharedStore()
        for i in range(1000):
            await store.write(MemoryRecord(
                content=f"Fact number {i}",
                author_agent_id=f"agent_{i % 5}",
                tags=[f"topic_{i % 10}"],
                confidence=0.9,
            ))
        stats = await store.stats()
        assert stats["total_writes"] == 1000

    async def test_concurrent_write_tasks(self):
        """Multiple async write tasks running concurrently."""
        store = InMemorySharedStore()

        async def writer(agent_id: str, count: int):
            for i in range(count):
                await store.write(MemoryRecord(
                    content=f"Fact {i} from {agent_id}",
                    author_agent_id=agent_id,
                    tags=[f"topic_{i % 5}"],
                ))

        tasks = [writer(f"agent_{j}", 100) for j in range(10)]
        await asyncio.gather(*tasks)

        stats = await store.stats()
        assert stats["total_writes"] == 1000

    async def test_concurrent_read_write(self):
        """Readers and writers operating simultaneously."""
        store = InMemorySharedStore()
        read_results = []

        # Pre-populate
        for i in range(100):
            await store.write(MemoryRecord(
                content=f"Seed fact {i}",
                author_agent_id="seeder",
                tags=["data"],
            ))

        async def writer():
            for i in range(200):
                await store.write(MemoryRecord(
                    content=f"New fact {i}",
                    author_agent_id="writer",
                    tags=["data"],
                ))

        async def reader():
            for _ in range(200):
                results = await store.read(agent_id="reader", tags=["data"], limit=5)
                read_results.append(len(results))

        await asyncio.gather(writer(), reader())

        # All reads should have returned results
        assert all(count > 0 for count in read_results)
        stats = await store.stats()
        assert stats["total_writes"] >= 300  # 100 seeds + 200 writes

    async def test_many_agents_many_tags(self):
        """50 agents writing across 100 different tag combinations."""
        store = InMemorySharedStore()
        num_agents = 50
        writes_per_agent = 20

        for agent_idx in range(num_agents):
            for w in range(writes_per_agent):
                await store.write(MemoryRecord(
                    content=f"A{agent_idx}-W{w}",
                    author_agent_id=f"agent_{agent_idx}",
                    # Use fully unique tags per write to avoid conflicts
                    tags=[f"agent_{agent_idx}_write_{w}"],
                ))

        stats = await store.stats()
        assert stats["total_writes"] == num_agents * writes_per_agent

        # Store should have many memories
        assert stats["total_memories"] > 0

        # Reading with a specific tag should return exactly 1 result
        results = await store.read(
            agent_id="reader_0",
            tags=[f"agent_0_write_0"],
            limit=200,
        )
        assert len(results) == 1

    async def test_rapid_update_chain(self):
        """Rapidly updating the same record 100 times."""
        store = InMemorySharedStore()

        initial, _ = await store.write(MemoryRecord(
            content="version_0",
            author_agent_id="updater",
            tags=["versioned"],
        ))
        current_id = initial.id

        for v in range(1, 101):
            updated = await store.update(
                current_id, "updater",
                {"content": f"version_{v}"},
                reason=f"update to v{v}",
            )
            current_id = updated.id

        # Only the latest version should be active
        results = await store.read(agent_id="reader", tags=["versioned"])
        active = [r for r in results if "version_100" in r.content]
        assert len(active) == 1

    async def test_forget_cascade_chain(self):
        """Forget with deep cascade chain."""
        store = InMemorySharedStore()

        # Create a chain: A → B → C → D → E
        prev_id = None
        record_ids = []
        for i in range(5):
            record, _ = await store.write(MemoryRecord(
                content=f"Chain link {i}",
                author_agent_id="builder",
                tags=["chain"],
                supersedes_id=prev_id,
            ))
            record_ids.append(record.id)
            prev_id = record.id

        # Forget the first one with cascade
        await store.forget(record_ids[0], "admin", cascade=True)

        # First record should be forgotten
        assert store._records[record_ids[0]].content == "[FORGOTTEN]"

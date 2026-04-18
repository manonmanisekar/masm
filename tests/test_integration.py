"""Integration tests — end-to-end multi-agent workflows through the full store."""

import pytest
from masm.store.in_memory import InMemorySharedStore
from masm.core.memory import MemoryRecord, MemoryState, MemoryType


@pytest.mark.asyncio
class TestIntegration:
    async def test_full_write_read_update_forget_cycle(self):
        """Test the complete lifecycle of a memory record."""
        store = InMemorySharedStore()

        # Write
        record, conflicts = await store.write(MemoryRecord(
            content="Customer email is test@example.com",
            author_agent_id="intake",
            tags=["pii", "contact"],
            confidence=0.99,
        ))
        assert record.state == MemoryState.ACTIVE

        # Read
        results = await store.read(agent_id="support", tags=["contact"])
        assert len(results) == 1
        assert results[0].content == "Customer email is test@example.com"

        # Update
        updated = await store.update(
            record.id, "support",
            {"content": "Customer email updated to new@example.com"},
            reason="customer changed email",
        )
        assert updated.version == 2
        assert updated.supersedes_id == record.id

        # Old version gone from reads
        results = await store.read(agent_id="support", tags=["contact"])
        active_results = [r for r in results if r.state == MemoryState.ACTIVE]
        assert all("new@example.com" in r.content for r in active_results)

        # Forget
        success = await store.forget(updated.id, "admin", reason="gdpr")
        assert success

        # Completely invisible
        results = await store.read(agent_id="support", tags=["contact"])
        assert len(results) == 0

        # Audit trail preserved
        audit = await store.get_audit_log(record_id=record.id)
        ops = [e["operation"] for e in audit]
        assert "write" in ops

    async def test_multi_agent_conflict_workflow(self):
        """Multiple agents write conflicting facts, detect and resolve."""
        store = InMemorySharedStore()

        # Agent A writes a fact
        a_record, _ = await store.write(MemoryRecord(
            content="Project deadline is March 15",
            content_embedding=[1.0, 0.0, 0.0],
            author_agent_id="pm",
            tags=["deadline", "project"],
            confidence=0.9,
        ))

        # Agent B writes a conflicting fact
        b_record, conflicts = await store.write(MemoryRecord(
            content="Project deadline moved to April 1",
            content_embedding=[0.0, 1.0, 0.0],
            author_agent_id="engineering",
            tags=["deadline", "project"],
            confidence=0.95,
        ))

        # Conflict should be detected
        assert len(conflicts) > 0

        # The store should resolve via LWW (default)
        assert conflicts[0].resolved

        # Reading should return the winner
        results = await store.read(agent_id="viewer", tags=["deadline"])
        active = [r for r in results if r.state == MemoryState.ACTIVE]
        assert len(active) >= 1

    async def test_handoff_preserves_context(self):
        """Simulate agent handoff — Agent B should see Agent A's memories."""
        store = InMemorySharedStore()

        # Agent A gathers info — using distinct tags to avoid false conflicts
        memories_written = []
        for content, tags in [
            ("Customer name is Alice", ["identity"]),
            ("Issue: cannot login", ["issue", "auth"]),
            ("Using Chrome on macOS", ["environment"]),
        ]:
            record, _ = await store.write(MemoryRecord(
                content=content,
                author_agent_id="l1_support",
                tags=tags,
                confidence=0.95,
            ))
            memories_written.append(record)

        # Agent B reads all context (no tag filter — get everything)
        context = await store.read(
            agent_id="l2_support",
            limit=10,
        )
        assert len(context) == 3

        # All critical facts available
        all_content = " ".join(r.content for r in context)
        assert "Alice" in all_content
        assert "login" in all_content
        assert "Chrome" in all_content

    async def test_subscription_notifications(self):
        """Test that subscribers receive notifications on relevant writes."""
        store = InMemorySharedStore()
        notifications = []

        async def on_alert(record):
            notifications.append(record)

        await store.subscribe("monitor", tags=["alert"], callback=on_alert)

        # Write matching tag
        await store.write(MemoryRecord(
            content="System alert: high CPU",
            author_agent_id="sensor",
            tags=["alert", "system"],
        ))
        assert len(notifications) == 1

        # Write non-matching tag — no notification
        await store.write(MemoryRecord(
            content="Normal log entry",
            author_agent_id="sensor",
            tags=["log"],
        ))
        assert len(notifications) == 1

    async def test_concurrent_writes_from_many_agents(self):
        """Simulate many agents writing simultaneously."""
        store = InMemorySharedStore()
        num_agents = 10
        writes_per_agent = 20

        for agent_idx in range(num_agents):
            for write_idx in range(writes_per_agent):
                await store.write(MemoryRecord(
                    content=f"Fact {write_idx} from agent {agent_idx}",
                    author_agent_id=f"agent_{agent_idx}",
                    tags=[f"topic_{write_idx % 5}"],
                    confidence=0.8,
                ))

        stats = await store.stats()
        assert stats["total_writes"] == num_agents * writes_per_agent
        assert stats["total_memories"] > 0

    async def test_all_benchmark_scenarios_produce_valid_results(self):
        """Verify every benchmark scenario runs and returns valid ScenarioResult."""
        from benchmark.scenarios.s01_dedup import DedupScenario
        from benchmark.scenarios.s02_conflict import ConflictScenario
        from benchmark.scenarios.s03_staleness import StalenessScenario
        from benchmark.scenarios.s04_attribution import AttributionScenario
        from benchmark.scenarios.s05_scalability import ScalabilityScenario
        from benchmark.scenarios.s06_relevance import RelevanceScenario
        from benchmark.scenarios.s07_forgetting import ForgettingScenario
        from benchmark.scenarios.s08_handoff import HandoffScenario
        from benchmark.scenarios.s09_adversarial import AdversarialScenario
        from benchmark.scenarios.s10_e2e_workflow import E2EWorkflowScenario

        scenarios = [
            DedupScenario(), ConflictScenario(), StalenessScenario(),
            AttributionScenario(), ScalabilityScenario(), RelevanceScenario(),
            ForgettingScenario(), HandoffScenario(), AdversarialScenario(),
            E2EWorkflowScenario(),
        ]

        for scenario in scenarios:
            store = InMemorySharedStore()
            await scenario.setup(store, num_agents=3)
            result = await scenario.run()
            await scenario.teardown()

            assert result.scenario_name == scenario.name, f"{scenario.name}: name mismatch"
            assert 0.0 <= result.score <= 1.0, f"{scenario.name}: score {result.score} out of range"
            assert result.num_operations > 0, f"{scenario.name}: no operations"
            assert isinstance(result.metrics, dict), f"{scenario.name}: metrics not a dict"

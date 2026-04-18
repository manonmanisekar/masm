"""Tests for benchmark scenarios — validate they run and produce valid results."""

import pytest
from masm.store.in_memory import InMemorySharedStore
from benchmark.scenarios.s01_dedup import DedupScenario
from benchmark.scenarios.s02_conflict import ConflictScenario
from benchmark.scenarios.s08_handoff import HandoffScenario


@pytest.mark.asyncio
class TestBenchmarkScenarios:
    async def test_s01_dedup_runs(self):
        store = InMemorySharedStore()
        scenario = DedupScenario()
        await scenario.setup(store, num_agents=5)
        result = await scenario.run()
        await scenario.teardown()

        assert result.scenario_name == "s01_dedup"
        assert 0.0 <= result.score <= 1.0
        assert result.num_operations > 0
        assert "dedup_rate" in result.metrics
        assert "false_positive_rate" in result.metrics

    async def test_s02_conflict_runs(self):
        store = InMemorySharedStore()
        scenario = ConflictScenario()
        await scenario.setup(store, num_agents=5)
        result = await scenario.run()
        await scenario.teardown()

        assert result.scenario_name == "s02_conflict"
        assert 0.0 <= result.score <= 1.0
        assert result.num_operations > 0
        assert "conflict_detection_rate" in result.metrics
        assert "resolution_accuracy" in result.metrics

    async def test_s08_handoff_runs(self):
        store = InMemorySharedStore()
        scenario = HandoffScenario()
        await scenario.setup(store, num_agents=5)
        result = await scenario.run()
        await scenario.teardown()

        assert result.scenario_name == "s08_handoff"
        assert 0.0 <= result.score <= 1.0
        assert result.num_operations > 0
        assert "context_fidelity" in result.metrics
        assert "context_noise" in result.metrics

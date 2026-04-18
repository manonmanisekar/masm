"""Benchmark orchestrator — runs all scenarios against a SharedMemoryStore implementation."""

import json
from datetime import datetime, timezone
from typing import Optional

from benchmark.scenarios.scenario_base import ScenarioResult


class MASMBenchmarkRunner:
    """
    Orchestrates all benchmark scenarios against a SharedMemoryStore implementation.

    Usage:
        runner = MASMBenchmarkRunner(store=MyStore())
        results = await runner.run_all()
        runner.report(results, format="markdown")
    """

    # Weighted importance of each scenario
    WEIGHTS = {
        "s01_dedup": 1.0,
        "s02_conflict": 1.5,  # Conflicts are hardest
        "s03_staleness": 1.0,
        "s04_attribution": 1.0,
        "s05_scalability": 1.2,
        "s06_relevance": 1.0,
        "s07_forgetting": 1.3,  # Compliance is critical
        "s08_handoff": 1.2,
        "s09_adversarial": 1.0,
        "s10_e2e_workflow": 1.5,  # End-to-end matters most
    }

    def __init__(self, store, scenarios: Optional[list] = None):
        self.store = store
        self.scenarios = scenarios or self._load_default_scenarios()

    def _load_default_scenarios(self) -> list:
        """Load all available benchmark scenarios."""
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

        return [
            DedupScenario(),
            ConflictScenario(),
            StalenessScenario(),
            AttributionScenario(),
            ScalabilityScenario(),
            RelevanceScenario(),
            ForgettingScenario(),
            HandoffScenario(),
            AdversarialScenario(),
            E2EWorkflowScenario(),
        ]

    async def run_all(
        self,
        num_agents: int = 5,
        scenario_names: Optional[list[str]] = None,
    ) -> dict:
        """Run all (or selected) scenarios and return aggregated results."""
        results: dict[str, ScenarioResult] = {}

        for scenario in self.scenarios:
            if scenario_names and scenario.name not in scenario_names:
                continue
            print(f"  Running: {scenario.name} — {scenario.description}")
            await scenario.setup(self.store, num_agents=num_agents)
            result = await scenario.run()
            await scenario.teardown()
            results[scenario.name] = result
            print(f"    Score: {result.score:.2%}")

        # Compute weighted composite score
        total_weight = sum(self.WEIGHTS.get(name, 1.0) for name in results)
        if total_weight > 0:
            composite = sum(
                results[name].score * self.WEIGHTS.get(name, 1.0) for name in results
            ) / total_weight
        else:
            composite = 0.0

        return {
            "composite_score": round(composite, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_agents": num_agents,
            "store_implementation": type(self.store).__name__,
            "scenarios": {name: self._result_to_dict(result) for name, result in results.items()},
        }

    def report(self, results: dict, format: str = "markdown") -> str:
        """Generate human-readable report."""
        if format == "markdown":
            return self._markdown_report(results)
        elif format == "json":
            return json.dumps(results, indent=2, default=str)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _markdown_report(self, results: dict) -> str:
        lines = [
            "# MASM Benchmark Results",
            "",
            f"**Composite Score: {results['composite_score']:.2%}**",
            f"**Store: {results['store_implementation']}**",
            f"**Agents: {results['num_agents']}**",
            f"**Date: {results['timestamp']}**",
            "",
            "| Scenario | Score | P50 Latency | P99 Latency |",
            "|----------|-------|-------------|-------------|",
        ]
        for name, data in results["scenarios"].items():
            lines.append(
                f"| {name} | {data['score']:.2%} | "
                f"{data['latency_p50_ms']:.1f}ms | "
                f"{data['latency_p99_ms']:.1f}ms |"
            )
        return "\n".join(lines)

    @staticmethod
    def _result_to_dict(result: ScenarioResult) -> dict:
        return {
            "scenario_name": result.scenario_name,
            "score": result.score,
            "metrics": result.metrics,
            "latency_p50_ms": result.latency_p50_ms,
            "latency_p99_ms": result.latency_p99_ms,
            "num_agents": result.num_agents,
            "num_operations": result.num_operations,
            "errors": result.errors,
            "metadata": result.metadata,
        }

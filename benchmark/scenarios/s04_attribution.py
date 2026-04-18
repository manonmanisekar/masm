"""S04: Attribution and Provenance Tracking.

After a complex multi-agent workflow, trace every memory back to its source.
5 agents pass information through a chain, each transforming/summarizing.
Verify the final memory links back through the entire chain.
"""

import time

from benchmark.scenarios.scenario_base import BenchmarkScenario, ScenarioResult
from masm.core.memory import MemoryRecord, MemoryType


# Chain scenarios: each agent transforms the previous agent's memory
PROVENANCE_CHAINS = [
    {
        "id": "chain_001",
        "title": "Research synthesis chain",
        "steps": [
            {
                "agent": "crawler",
                "content": "Found article: 'AI spending to reach $200B by 2025'",
                "evidence": "Source: Gartner report Q3 2024",
                "type": "episodic",
            },
            {
                "agent": "reader",
                "content": "AI spending projected at $200B by 2025 according to Gartner",
                "evidence": "Extracted from crawler's finding",
                "type": "semantic",
            },
            {
                "agent": "analyst",
                "content": "AI market is growing rapidly — $200B projection implies 40% YoY growth",
                "evidence": "Derived from reader's extraction + market analysis",
                "type": "semantic",
            },
            {
                "agent": "synthesizer",
                "content": "Key trend: AI market at $200B (40% YoY growth) — major investment opportunity",
                "evidence": "Synthesized from analyst's analysis",
                "type": "semantic",
            },
            {
                "agent": "writer",
                "content": "The AI market is experiencing explosive growth at 40% YoY, projected to hit $200B",
                "evidence": "Final report paragraph from synthesizer's summary",
                "type": "procedural",
            },
        ],
    },
    {
        "id": "chain_002",
        "title": "Customer issue resolution chain",
        "steps": [
            {
                "agent": "intake",
                "content": "Customer John reported login failures since Tuesday",
                "evidence": "Direct customer message via chat",
                "type": "episodic",
            },
            {
                "agent": "triage",
                "content": "Login issue affects user John — likely auth token expiration bug",
                "evidence": "Correlated with known bug #4521",
                "type": "semantic",
            },
            {
                "agent": "engineer",
                "content": "Root cause: auth token TTL was set to 24h instead of 7d in last deploy",
                "evidence": "Found in commit abc123, deploy log from Monday",
                "type": "semantic",
            },
            {
                "agent": "resolver",
                "content": "Fix deployed: auth token TTL corrected to 7d. John's issue resolved.",
                "evidence": "Hotfix commit def456, verified in staging",
                "type": "episodic",
            },
        ],
    },
    {
        "id": "chain_003",
        "title": "Data pipeline chain",
        "steps": [
            {
                "agent": "collector",
                "content": "Ingested 1.2M rows from sales_db for Q4 analysis",
                "evidence": "ETL job run_id: etl_20250115",
                "type": "episodic",
            },
            {
                "agent": "cleaner",
                "content": "Cleaned dataset: removed 15K duplicates, 3K nulls. 1.182M rows remaining.",
                "evidence": "Cleaning report from collector's raw data",
                "type": "semantic",
            },
            {
                "agent": "modeler",
                "content": "Q4 revenue forecast: $12.5M with 95% CI [$11.8M, $13.2M]",
                "evidence": "XGBoost model trained on cleaner's dataset",
                "type": "semantic",
            },
        ],
    },
]


class AttributionScenario(BenchmarkScenario):
    """Benchmark: provenance tracking through multi-agent chains."""

    @property
    def name(self) -> str:
        return "s04_attribution"

    @property
    def description(self) -> str:
        return "Memory provenance and attribution tracking through agent chains"

    async def setup(self, store, num_agents: int = 5, **kwargs):
        self.store = store
        self.num_agents = num_agents
        self.latencies: list[float] = []

    async def run(self) -> ScenarioResult:
        errors = []
        total_memories = 0
        complete_provenance = 0
        correct_attribution = 0
        max_lineage_depth = 0
        total_operations = 0

        for chain in PROVENANCE_CHAINS:
            prev_record_id = None

            for step in chain["steps"]:
                record = MemoryRecord(
                    content=step["content"],
                    author_agent_id=step["agent"],
                    memory_type=MemoryType(step["type"]),
                    tags=[chain["id"], "provenance_test"],
                    confidence=0.95,
                    supersedes_id=prev_record_id,
                    metadata={"chain_id": chain["id"], "evidence": step["evidence"]},
                )
                t0 = time.perf_counter()
                written, _ = await self.store.write(record)
                self.latencies.append((time.perf_counter() - t0) * 1000)
                total_operations += 1
                total_memories += 1

                # Check provenance completeness
                has_author = bool(written.author_agent_id)
                has_evidence = bool(written.metadata.get("evidence"))
                has_timestamp = written.created_at is not None
                if has_author and has_evidence and has_timestamp:
                    complete_provenance += 1

                # Check attribution correctness
                if written.author_agent_id == step["agent"]:
                    correct_attribution += 1

                prev_record_id = written.id

            # Verify lineage: trace from last record back through chain
            lineage_depth = 0
            current_id = prev_record_id
            visited = set()

            while current_id and current_id not in visited:
                visited.add(current_id)
                # Read audit log for this record
                audit = await self.store.get_audit_log(record_id=current_id)
                total_operations += 1

                # Find the record to check supersedes chain
                results = await self.store.read(
                    agent_id="auditor",
                    tags=[chain["id"]],
                    limit=100,
                )
                total_operations += 1

                found = None
                for r in results:
                    if r.id == current_id:
                        found = r
                        break

                if found and found.supersedes_id:
                    lineage_depth += 1
                    current_id = found.supersedes_id
                else:
                    # Try to find in all records (including superseded)
                    all_results = await self.store.read(
                        agent_id="auditor",
                        tags=[chain["id"]],
                        limit=100,
                        include_disputed=True,
                    )
                    total_operations += 1

                    # Check store internals for supersedes chain
                    if hasattr(self.store, "_records"):
                        record_obj = self.store._records.get(current_id)
                        if record_obj and record_obj.supersedes_id:
                            lineage_depth += 1
                            current_id = record_obj.supersedes_id
                            continue
                    break

            max_lineage_depth = max(max_lineage_depth, lineage_depth)

        # Metrics
        provenance_completeness = complete_provenance / max(total_memories, 1)
        provenance_accuracy = correct_attribution / max(total_memories, 1)
        expected_max_depth = max(len(c["steps"]) - 1 for c in PROVENANCE_CHAINS)
        lineage_score = min(max_lineage_depth / max(expected_max_depth, 1), 1.0)

        score = (
            provenance_completeness * 0.35
            + provenance_accuracy * 0.35
            + lineage_score * 0.30
        )

        sorted_latencies = sorted(self.latencies) if self.latencies else [0.0]
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return ScenarioResult(
            scenario_name=self.name,
            score=round(score, 4),
            metrics={
                "provenance_completeness": round(provenance_completeness, 4),
                "provenance_accuracy": round(provenance_accuracy, 4),
                "max_lineage_depth": max_lineage_depth,
                "expected_max_depth": expected_max_depth,
                "lineage_score": round(lineage_score, 4),
                "total_memories_traced": total_memories,
            },
            latency_p50_ms=round(p50, 2),
            latency_p99_ms=round(p99, 2),
            num_agents=self.num_agents,
            num_operations=total_operations,
            errors=errors,
        )

    async def teardown(self):
        self.latencies.clear()

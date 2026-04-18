"""S06: Per-Agent Relevance Filtering.

Each agent has a role (researcher, coder, reviewer). Shared memory contains
many memories. Each agent should retrieve only relevant ones.
"""

import time
import numpy as np

from benchmark.scenarios.scenario_base import BenchmarkScenario, ScenarioResult
from masm.core.memory import MemoryRecord


# Memories with labeled relevance per agent role
MEMORIES = [
    # Research-relevant
    {"content": "Market size for AI tools is $50B", "tags": ["research", "market"], "relevant_to": ["researcher"]},
    {"content": "Competitor X launched a new product", "tags": ["research", "competitive"], "relevant_to": ["researcher"]},
    {"content": "User survey shows 80% want faster responses", "tags": ["research", "user_feedback"], "relevant_to": ["researcher", "coder"]},
    {"content": "Academic paper: Transformer efficiency gains 30%", "tags": ["research", "ml"], "relevant_to": ["researcher"]},
    {"content": "Industry report: top 3 trends in 2025", "tags": ["research", "trends"], "relevant_to": ["researcher"]},
    # Code-relevant
    {"content": "API endpoint /users has 500ms p99 latency", "tags": ["code", "performance"], "relevant_to": ["coder"]},
    {"content": "Memory leak in connection pool module", "tags": ["code", "bug"], "relevant_to": ["coder", "reviewer"]},
    {"content": "New caching layer reduces DB calls by 60%", "tags": ["code", "optimization"], "relevant_to": ["coder"]},
    {"content": "Migration to Python 3.12 completed", "tags": ["code", "infrastructure"], "relevant_to": ["coder", "reviewer"]},
    {"content": "Test coverage is at 78%", "tags": ["code", "testing"], "relevant_to": ["coder", "reviewer"]},
    # Review-relevant
    {"content": "PR #451 has 3 unresolved comments", "tags": ["review", "pr"], "relevant_to": ["reviewer"]},
    {"content": "Security audit flagged SQL injection risk", "tags": ["review", "security"], "relevant_to": ["reviewer"]},
    {"content": "Code style violations in auth module", "tags": ["review", "style"], "relevant_to": ["reviewer"]},
    {"content": "Architecture decision: move to microservices", "tags": ["review", "architecture"], "relevant_to": ["reviewer", "coder"]},
    {"content": "Dependency update: lodash 4.17.21 has CVE", "tags": ["review", "security"], "relevant_to": ["reviewer", "coder"]},
    # Irrelevant noise
    {"content": "Office lunch order for Friday", "tags": ["misc"], "relevant_to": []},
    {"content": "Team building event next month", "tags": ["misc", "social"], "relevant_to": []},
    {"content": "Parking lot construction update", "tags": ["misc", "facilities"], "relevant_to": []},
]

AGENT_ROLES = {
    "researcher": {"role": "research analyst", "query_tags": ["research"]},
    "coder": {"role": "software engineer", "query_tags": ["code"]},
    "reviewer": {"role": "code reviewer", "query_tags": ["review"]},
}


class RelevanceScenario(BenchmarkScenario):
    """Benchmark: per-agent relevance filtering quality."""

    @property
    def name(self) -> str:
        return "s06_relevance"

    @property
    def description(self) -> str:
        return "Per-agent relevance filtering precision and recall"

    async def setup(self, store, num_agents: int = 5, **kwargs):
        self.store = store
        self.num_agents = num_agents
        self.latencies: list[float] = []

    async def run(self) -> ScenarioResult:
        errors = []
        total_operations = 0

        # Write all memories to the store
        memory_map = {}  # content -> record_id
        for mem in MEMORIES:
            record = MemoryRecord(
                content=mem["content"],
                author_agent_id="system",
                tags=mem["tags"],
                confidence=0.9,
            )
            t0 = time.perf_counter()
            written, _ = await self.store.write(record)
            self.latencies.append((time.perf_counter() - t0) * 1000)
            total_operations += 1
            memory_map[mem["content"]] = written.id

        # Build ground truth relevance per agent
        ground_truth = {}
        for role_name in AGENT_ROLES:
            relevant_ids = set()
            for mem in MEMORIES:
                if role_name in mem["relevant_to"]:
                    relevant_ids.add(memory_map[mem["content"]])
            ground_truth[role_name] = relevant_ids

        # Each agent queries and we measure precision/recall
        precisions = []
        recalls = []
        irrelevance_rates = []

        for role_name, role_def in AGENT_ROLES.items():
            t0 = time.perf_counter()
            results = await self.store.read(
                agent_id=role_name,
                tags=role_def["query_tags"],
                limit=10,
            )
            self.latencies.append((time.perf_counter() - t0) * 1000)
            total_operations += 1

            retrieved_ids = {r.id for r in results}
            relevant_ids = ground_truth[role_name]

            # Precision@10: fraction of retrieved that are truly relevant
            true_positives = len(retrieved_ids & relevant_ids)
            precision = true_positives / max(len(retrieved_ids), 1)
            precisions.append(precision)

            # Recall: fraction of relevant that were retrieved
            recall = true_positives / max(len(relevant_ids), 1)
            recalls.append(recall)

            # Irrelevance rate: fraction of retrieved that are noise
            irrelevant = len(retrieved_ids - relevant_ids)
            irrelevance_rate = irrelevant / max(len(retrieved_ids), 1)
            irrelevance_rates.append(irrelevance_rate)

        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_irrelevance = np.mean(irrelevance_rates)

        score = avg_precision * 0.4 + avg_recall * 0.4 + (1.0 - avg_irrelevance) * 0.2

        sorted_latencies = sorted(self.latencies) if self.latencies else [0.0]
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return ScenarioResult(
            scenario_name=self.name,
            score=round(score, 4),
            metrics={
                "avg_precision_at_10": round(avg_precision, 4),
                "avg_recall": round(avg_recall, 4),
                "avg_irrelevance_rate": round(avg_irrelevance, 4),
                "per_agent_precision": {k: round(v, 4) for k, v in zip(AGENT_ROLES.keys(), precisions)},
                "per_agent_recall": {k: round(v, 4) for k, v in zip(AGENT_ROLES.keys(), recalls)},
            },
            latency_p50_ms=round(p50, 2),
            latency_p99_ms=round(p99, 2),
            num_agents=self.num_agents,
            num_operations=total_operations,
            errors=errors,
        )

    async def teardown(self):
        self.latencies.clear()

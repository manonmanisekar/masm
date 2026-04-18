"""S08: Agent-to-Agent Context Handoff.

Agent A handles a task then hands off to Agent B. Measures whether Agent B
receives all critical context without re-asking the user.
"""

import time
import numpy as np

from benchmark.scenarios.scenario_base import BenchmarkScenario, ScenarioResult
from masm.core.memory import MemoryRecord, MemoryType


# Handoff scenarios — simulating customer support L1 → L2 escalation
HANDOFF_SCENARIOS = [
    {
        "id": "hs_001",
        "title": "Billing dispute escalation",
        "agent_a": "l1_support",
        "agent_b": "l2_billing",
        "memories_written": [
            {
                "content": "Customer reports double charge on subscription",
                "tags": ["billing", "issue"],
                "memory_type": "episodic",
                "critical": True,
            },
            {
                "content": "Customer is on Pro plan",
                "tags": ["plan", "account"],
                "memory_type": "semantic",
                "critical": True,
            },
            {
                "content": "Customer tenure is 3 years",
                "tags": ["tenure", "loyalty"],
                "memory_type": "semantic",
                "critical": True,
            },
            {
                "content": "Customer prefers email communication",
                "tags": ["preferences", "communication"],
                "memory_type": "semantic",
                "critical": False,
            },
            {
                "content": "Customer mentioned they are considering cancellation",
                "tags": ["churn_risk", "retention"],
                "memory_type": "episodic",
                "critical": True,
            },
        ],
        "agent_b_query_tags": ["billing", "account", "churn_risk"],
    },
    {
        "id": "hs_002",
        "title": "Technical issue escalation",
        "agent_a": "l1_support",
        "agent_b": "l2_engineering",
        "memories_written": [
            {
                "content": "User cannot log in after password reset",
                "tags": ["auth", "login", "bug"],
                "memory_type": "episodic",
                "critical": True,
            },
            {
                "content": "Error message: 'Invalid token'",
                "tags": ["auth", "error", "technical"],
                "memory_type": "semantic",
                "critical": True,
            },
            {
                "content": "User is on Chrome 120, macOS Sonoma",
                "tags": ["environment", "technical"],
                "memory_type": "semantic",
                "critical": True,
            },
            {
                "content": "User tried clearing cookies, issue persists",
                "tags": ["troubleshooting", "technical"],
                "memory_type": "episodic",
                "critical": True,
            },
            {
                "content": "User said they are frustrated",
                "tags": ["sentiment"],
                "memory_type": "episodic",
                "critical": False,
            },
        ],
        "agent_b_query_tags": ["auth", "technical", "bug"],
    },
    {
        "id": "hs_003",
        "title": "Feature request handoff to product",
        "agent_a": "support_agent",
        "agent_b": "product_manager",
        "memories_written": [
            {
                "content": "Customer requests bulk export feature for reports",
                "tags": ["feature_request", "export"],
                "memory_type": "semantic",
                "critical": True,
            },
            {
                "content": "Customer is an Enterprise client with 500 users",
                "tags": ["account", "enterprise"],
                "memory_type": "semantic",
                "critical": True,
            },
            {
                "content": "Customer says this is blocking their quarterly reporting",
                "tags": ["impact", "urgency"],
                "memory_type": "episodic",
                "critical": True,
            },
            {
                "content": "Customer mentioned competitors have this feature",
                "tags": ["competitive", "churn_risk"],
                "memory_type": "episodic",
                "critical": False,
            },
        ],
        "agent_b_query_tags": ["feature_request", "impact", "account"],
    },
]


def _fake_embedding(text: str, dim: int = 64) -> list[float]:
    """Deterministic pseudo-embedding for benchmarks."""
    words = text.lower().split()
    vec = np.zeros(dim)
    for w in words:
        word_seed = sum(ord(c) for c in w) % (2**31)
        word_rng = np.random.RandomState(word_seed)
        vec += word_rng.randn(dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


class HandoffScenario(BenchmarkScenario):
    """Benchmark: context fidelity during agent-to-agent handoffs."""

    @property
    def name(self) -> str:
        return "s08_handoff"

    @property
    def description(self) -> str:
        return "Agent-to-agent context handoff fidelity"

    async def setup(self, store, num_agents: int = 5, **kwargs):
        self.store = store
        self.num_agents = num_agents
        self.latencies: list[float] = []

    async def run(self) -> ScenarioResult:
        errors = []
        total_critical = 0
        critical_retrieved = 0
        total_non_critical = 0
        non_critical_retrieved = 0
        total_operations = 0

        for scenario in HANDOFF_SCENARIOS:
            # Phase 1: Agent A writes memories during their interaction
            written_records = []
            for mem_def in scenario["memories_written"]:
                record = MemoryRecord(
                    content=mem_def["content"],
                    content_embedding=_fake_embedding(mem_def["content"]),
                    author_agent_id=scenario["agent_a"],
                    tags=mem_def["tags"],
                    memory_type=MemoryType(mem_def["memory_type"]),
                    confidence=0.9,
                )
                t0 = time.perf_counter()
                written, _ = await self.store.write(record)
                self.latencies.append((time.perf_counter() - t0) * 1000)
                total_operations += 1
                written_records.append((written, mem_def["critical"]))

            # Phase 2: Agent B reads shared memory to get context
            t0 = time.perf_counter()
            retrieved = await self.store.read(
                agent_id=scenario["agent_b"],
                tags=scenario["agent_b_query_tags"],
                limit=20,
            )
            self.latencies.append((time.perf_counter() - t0) * 1000)
            total_operations += 1

            retrieved_ids = {r.id for r in retrieved}

            # Phase 3: Check how many critical facts were retrieved
            for written_record, is_critical in written_records:
                if is_critical:
                    total_critical += 1
                    if written_record.id in retrieved_ids:
                        critical_retrieved += 1
                else:
                    total_non_critical += 1
                    if written_record.id in retrieved_ids:
                        non_critical_retrieved += 1

        # Metrics
        context_fidelity = critical_retrieved / max(total_critical, 1)
        # Noise: ratio of irrelevant (non-critical) memories in retrieved set
        total_retrieved = critical_retrieved + non_critical_retrieved
        context_noise = non_critical_retrieved / max(total_retrieved, 1) if total_retrieved > 0 else 0

        # Score: fidelity is most important, low noise is secondary
        score = context_fidelity * 0.8 + (1.0 - context_noise) * 0.2

        sorted_latencies = sorted(self.latencies) if self.latencies else [0.0]
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return ScenarioResult(
            scenario_name=self.name,
            score=round(score, 4),
            metrics={
                "context_fidelity": round(context_fidelity, 4),
                "context_noise": round(context_noise, 4),
                "critical_retrieved": critical_retrieved,
                "total_critical": total_critical,
                "non_critical_retrieved": non_critical_retrieved,
                "total_non_critical": total_non_critical,
            },
            latency_p50_ms=round(p50, 2),
            latency_p99_ms=round(p99, 2),
            num_agents=self.num_agents,
            num_operations=total_operations,
            errors=errors,
        )

    async def teardown(self):
        self.latencies.clear()

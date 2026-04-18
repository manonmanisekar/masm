"""S09: Adversarial / Conflicting Agent Inputs.

One agent is 'unreliable' (produces low-confidence or contradictory memories).
Measures the system's ability to isolate bad data.
"""

import time
import numpy as np

from benchmark.scenarios.scenario_base import BenchmarkScenario, ScenarioResult
from masm.core.memory import MemoryRecord, MemoryState


# Reliable facts written by trustworthy agents
RELIABLE_FACTS = [
    {"content": "Customer name is Alice Johnson", "tags": ["customer", "name"], "confidence": 0.95},
    {"content": "Account created on 2023-06-15", "tags": ["customer", "account"], "confidence": 0.98},
    {"content": "Subscription plan: Enterprise", "tags": ["customer", "plan"], "confidence": 0.99},
    {"content": "Annual contract value: $120,000", "tags": ["customer", "revenue"], "confidence": 0.95},
    {"content": "Primary contact: alice@company.com", "tags": ["customer", "contact"], "confidence": 0.97},
    {"content": "Company size: 500 employees", "tags": ["customer", "company"], "confidence": 0.90},
    {"content": "Industry: Financial Services", "tags": ["customer", "company"], "confidence": 0.92},
    {"content": "Support tier: Premium", "tags": ["customer", "support"], "confidence": 0.96},
]

# Adversarial facts — plausible but wrong (swapped, incorrect dates, wrong numbers)
ADVERSARIAL_FACTS = [
    {"content": "Customer name is Bob Smith", "tags": ["customer", "name"], "confidence": 0.60},
    {"content": "Account created on 2024-01-01", "tags": ["customer", "account"], "confidence": 0.55},
    {"content": "Subscription plan: Free tier", "tags": ["customer", "plan"], "confidence": 0.50},
    {"content": "Annual contract value: $12,000", "tags": ["customer", "revenue"], "confidence": 0.45},
    {"content": "Company size: 5 employees", "tags": ["customer", "company"], "confidence": 0.40},
]


class AdversarialScenario(BenchmarkScenario):
    """Benchmark: isolation of unreliable/adversarial agent memories."""

    @property
    def name(self) -> str:
        return "s09_adversarial"

    @property
    def description(self) -> str:
        return "Adversarial agent input isolation and trust calibration"

    async def setup(self, store, num_agents: int = 5, **kwargs):
        self.store = store
        self.num_agents = num_agents
        self.reliable_agents = [f"reliable_{i}" for i in range(4)]
        self.adversarial_agent = "adversarial_0"
        self.latencies: list[float] = []

    async def run(self) -> ScenarioResult:
        errors = []
        total_operations = 0

        # Phase 1: Reliable agents write correct facts
        reliable_ids = []
        for i, fact in enumerate(RELIABLE_FACTS):
            agent_id = self.reliable_agents[i % len(self.reliable_agents)]
            record = MemoryRecord(
                content=fact["content"],
                author_agent_id=agent_id,
                tags=fact["tags"],
                confidence=fact["confidence"],
            )
            t0 = time.perf_counter()
            written, _ = await self.store.write(record)
            self.latencies.append((time.perf_counter() - t0) * 1000)
            total_operations += 1
            reliable_ids.append(written.id)

        # Phase 2: Adversarial agent writes incorrect facts
        adversarial_ids = []
        for fact in ADVERSARIAL_FACTS:
            record = MemoryRecord(
                content=fact["content"],
                author_agent_id=self.adversarial_agent,
                tags=fact["tags"],
                confidence=fact["confidence"],
            )
            t0 = time.perf_counter()
            written, conflicts = await self.store.write(record)
            self.latencies.append((time.perf_counter() - t0) * 1000)
            total_operations += 1
            adversarial_ids.append(written.id)

        # Phase 3: A neutral agent reads — should prefer reliable facts
        t0 = time.perf_counter()
        results = await self.store.read(
            agent_id="neutral_reader",
            tags=["customer"],
            limit=20,
        )
        self.latencies.append((time.perf_counter() - t0) * 1000)
        total_operations += 1

        # Evaluate: are adversarial facts ranked lower than reliable ones?
        retrieved_ids = [r.id for r in results]
        adversarial_in_results = [rid for rid in adversarial_ids if rid in retrieved_ids]
        reliable_in_results = [rid for rid in reliable_ids if rid in retrieved_ids]

        # Bad memory isolation: how many adversarial facts are below reliable ones in ranking
        bad_isolated = 0
        for adv_id in adversarial_in_results:
            adv_rank = retrieved_ids.index(adv_id)
            # Check if all reliable facts with overlapping tags rank higher
            higher_reliable = sum(1 for rel_id in reliable_in_results if retrieved_ids.index(rel_id) < adv_rank)
            if higher_reliable > 0:
                bad_isolated += 1

        isolation_rate = bad_isolated / max(len(adversarial_in_results), 1)

        # Trust calibration: does confidence scoring correlate with reliability?
        # Reliable facts should have higher confidence than adversarial ones
        reliable_confidences = [r.confidence for r in results if r.id in reliable_ids]
        adversarial_confidences = [r.confidence for r in results if r.id in adversarial_ids]

        if reliable_confidences and adversarial_confidences:
            avg_reliable_conf = np.mean(reliable_confidences)
            avg_adversarial_conf = np.mean(adversarial_confidences)
            trust_calibration = 1.0 if avg_reliable_conf > avg_adversarial_conf else 0.5
        else:
            trust_calibration = 0.5

        # System corruption: do adversarial facts appear as the "winner" in any conflict?
        all_conflicts = await self.store.get_conflicts(unresolved_only=False)
        total_operations += 1
        corruption_count = 0
        for a, b in all_conflicts:
            if a.id in adversarial_ids and a.state == MemoryState.ACTIVE:
                corruption_count += 1
            if b.id in adversarial_ids and b.state == MemoryState.ACTIVE:
                corruption_count += 1

        corruption_rate = corruption_count / max(len(ADVERSARIAL_FACTS), 1)

        score = isolation_rate * 0.4 + trust_calibration * 0.3 + (1.0 - corruption_rate) * 0.3

        sorted_latencies = sorted(self.latencies) if self.latencies else [0.0]
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return ScenarioResult(
            scenario_name=self.name,
            score=round(score, 4),
            metrics={
                "bad_memory_isolation_rate": round(isolation_rate, 4),
                "trust_calibration": round(trust_calibration, 4),
                "system_corruption_rate": round(corruption_rate, 4),
                "adversarial_facts_written": len(ADVERSARIAL_FACTS),
                "reliable_facts_written": len(RELIABLE_FACTS),
                "adversarial_in_top_results": len(adversarial_in_results),
            },
            latency_p50_ms=round(p50, 2),
            latency_p99_ms=round(p99, 2),
            num_agents=self.num_agents,
            num_operations=total_operations,
            errors=errors,
        )

    async def teardown(self):
        self.latencies.clear()

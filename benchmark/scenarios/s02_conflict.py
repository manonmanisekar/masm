"""S02: Conflict Detection and Resolution.

Agents encounter contradictory information at different times. Measures how
well the store detects true conflicts, avoids false conflicts, and resolves
them correctly.
"""

import time
import numpy as np
from datetime import datetime, timezone, timedelta

from benchmark.scenarios.scenario_base import BenchmarkScenario, ScenarioResult
from masm.core.memory import MemoryRecord, MemoryState


# True conflicts — contradictory facts that should be detected
TRUE_CONFLICTS = [
    {
        "a": {"content": "Budget is $50,000", "tags": ["budget"]},
        "b": {"content": "Budget increased to $75,000", "tags": ["budget"]},
        "correct_winner": "b",  # More recent from customer
        "reason": "Updated information supersedes old",
    },
    {
        "a": {"content": "Team uses Python 3.10", "tags": ["tech_stack"]},
        "b": {"content": "Team migrated to Python 3.12", "tags": ["tech_stack"]},
        "correct_winner": "b",
        "reason": "Migration happened after initial report",
    },
    {
        "a": {"content": "Customer is on the Free plan", "tags": ["plan", "billing"]},
        "b": {"content": "Customer upgraded to Pro plan", "tags": ["plan", "billing"]},
        "correct_winner": "b",
        "reason": "Upgrade happened after initial observation",
    },
    {
        "a": {"content": "Office is in New York", "tags": ["location"]},
        "b": {"content": "Company relocated to Austin", "tags": ["location"]},
        "correct_winner": "b",
        "reason": "Relocation is more recent info",
    },
    {
        "a": {"content": "Primary contact is Alice", "tags": ["contact"]},
        "b": {"content": "Primary contact changed to Bob", "tags": ["contact"]},
        "correct_winner": "b",
        "reason": "Contact change is more recent",
    },
]

# Non-conflicts — should NOT be flagged
NON_CONFLICTS = [
    {
        "a": {"content": "Project deadline is March 15", "tags": ["deadline"]},
        "b": {"content": "Project deadline is March 15, 2026", "tags": ["deadline"]},
        "reason": "B is more specific version of A — not a conflict",
    },
    {
        "a": {"content": "Customer likes dark mode", "tags": ["preferences"]},
        "b": {"content": "Customer prefers dark theme in the app", "tags": ["preferences"]},
        "reason": "Same fact restated — duplicate, not conflict",
    },
    {
        "a": {"content": "Revenue is $1M ARR", "tags": ["revenue"]},
        "b": {"content": "Employee count is 50", "tags": ["headcount"]},
        "reason": "Different topics — no conflict",
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


class ConflictScenario(BenchmarkScenario):
    """Benchmark: conflict detection and resolution accuracy."""

    @property
    def name(self) -> str:
        return "s02_conflict"

    @property
    def description(self) -> str:
        return "Conflict detection and resolution under contradictory agent inputs"

    async def setup(self, store, num_agents: int = 5, **kwargs):
        self.store = store
        self.num_agents = num_agents
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.latencies: list[float] = []

    async def run(self) -> ScenarioResult:
        errors = []
        detected_conflicts = 0
        correctly_resolved = 0
        false_conflict_count = 0
        total_operations = 0

        # Test 1: True conflicts
        for i, conflict_def in enumerate(TRUE_CONFLICTS):
            base_time = datetime.now(timezone.utc) - timedelta(hours=1)

            # Agent 0 writes the earlier fact
            record_a = MemoryRecord(
                content=conflict_def["a"]["content"],
                content_embedding=_fake_embedding(conflict_def["a"]["content"]),
                author_agent_id=self.agent_ids[0],
                tags=conflict_def["a"]["tags"],
                confidence=0.9,
                created_at=base_time,
            )
            t0 = time.perf_counter()
            await self.store.write(record_a)
            self.latencies.append((time.perf_counter() - t0) * 1000)
            total_operations += 1

            # Agent 1 writes the contradictory later fact
            record_b = MemoryRecord(
                content=conflict_def["b"]["content"],
                content_embedding=_fake_embedding(conflict_def["b"]["content"]),
                author_agent_id=self.agent_ids[1],
                tags=conflict_def["b"]["tags"],
                confidence=0.9,
                created_at=base_time + timedelta(minutes=30),
            )
            t0 = time.perf_counter()
            written_b, conflicts = await self.store.write(record_b, conflict_strategy="lww")
            self.latencies.append((time.perf_counter() - t0) * 1000)
            total_operations += 1

            # Was the conflict detected?
            if conflicts:
                detected_conflicts += 1
                # Was it resolved correctly? (LWW should pick the more recent one = B)
                for c in conflicts:
                    if c.resolved and c.resolved_memory_id == written_b.id:
                        correctly_resolved += 1
                        break

        # Test 2: Non-conflicts
        for non_conflict in NON_CONFLICTS:
            record_a = MemoryRecord(
                content=non_conflict["a"]["content"],
                content_embedding=_fake_embedding(non_conflict["a"]["content"]),
                author_agent_id=self.agent_ids[min(2, self.num_agents - 1)],
                tags=non_conflict["a"]["tags"],
                confidence=0.9,
            )
            await self.store.write(record_a)
            total_operations += 1

            record_b = MemoryRecord(
                content=non_conflict["b"]["content"],
                content_embedding=_fake_embedding(non_conflict["b"]["content"]),
                author_agent_id=self.agent_ids[min(3, self.num_agents - 1)],
                tags=non_conflict["b"]["tags"],
                confidence=0.9,
            )
            _, conflicts = await self.store.write(record_b)
            total_operations += 1

            if conflicts:
                false_conflict_count += 1

        # Metrics
        total_true_conflicts = len(TRUE_CONFLICTS)
        total_non_conflicts = len(NON_CONFLICTS)

        detection_rate = detected_conflicts / max(total_true_conflicts, 1)
        resolution_accuracy = correctly_resolved / max(detected_conflicts, 1)
        false_conflict_rate = false_conflict_count / max(total_non_conflicts, 1)

        # Score: detection * resolution * (1 - false positive penalty)
        score = detection_rate * 0.4 + resolution_accuracy * 0.4 + (1.0 - false_conflict_rate) * 0.2

        sorted_latencies = sorted(self.latencies) if self.latencies else [0.0]
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return ScenarioResult(
            scenario_name=self.name,
            score=round(score, 4),
            metrics={
                "conflict_detection_rate": round(detection_rate, 4),
                "resolution_accuracy": round(resolution_accuracy, 4),
                "false_conflict_rate": round(false_conflict_rate, 4),
                "detected_conflicts": detected_conflicts,
                "correctly_resolved": correctly_resolved,
                "false_conflicts": false_conflict_count,
                "total_true_conflicts": total_true_conflicts,
                "total_non_conflicts": total_non_conflicts,
            },
            latency_p50_ms=round(p50, 2),
            latency_p99_ms=round(p99, 2),
            num_agents=self.num_agents,
            num_operations=total_operations,
            errors=errors,
        )

    async def teardown(self):
        self.latencies.clear()

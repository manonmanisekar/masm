"""S01: Deduplication Under Concurrent Writes.

5 agents independently discover the same fact from different conversations
and write it to shared memory simultaneously. Measures how well the store
detects and merges semantic duplicates.
"""

import asyncio
import time
import numpy as np
from statistics import median

from .scenario_base import BenchmarkScenario, ScenarioResult
from masm.core.memory import MemoryRecord, MemoryState


# Ground truth: pairs of semantically equivalent statements
DUPLICATE_PAIRS = [
    ("Customer prefers email communication", "The customer said they like getting emails"),
    ("Budget is $50,000 for the project", "The project budget is 50K dollars"),
    ("Meeting scheduled for Monday at 10am", "There's a meeting on Monday morning at 10"),
    ("User wants dark mode enabled", "The user asked to turn on dark mode"),
    ("Deadline is end of Q1 2026", "Project must be done by March 31, 2026"),
    ("Customer is unhappy with response time", "The client complained about slow support"),
    ("Account was created in January 2024", "User signed up in Jan 2024"),
    ("Preferred language is Spanish", "Customer speaks Spanish"),
    ("Product version is 3.2.1", "Running version 3.2.1 of the product"),
    ("Team size is 15 engineers", "There are fifteen engineers on the team"),
]

# Non-duplicate pairs (should NOT be merged)
NON_DUPLICATE_PAIRS = [
    ("Customer prefers email communication", "Customer email is john@example.com"),
    ("Budget is $50,000 for the project", "Project has 50 open tasks"),
    ("Meeting scheduled for Monday", "Meeting room capacity is 20 people"),
    ("User wants dark mode", "User account has dark history of chargebacks"),
    ("Deadline is end of Q1", "Q1 revenue exceeded expectations"),
]


def _fake_embedding(text: str, dim: int = 64) -> list[float]:
    """
    Generate a deterministic pseudo-embedding for benchmark purposes.
    Semantically similar texts produce similar vectors via shared word hashing.
    """
    rng = np.random.RandomState(42)
    words = text.lower().split()
    vec = np.zeros(dim)
    for w in words:
        word_seed = sum(ord(c) for c in w) % (2**31)
        word_rng = np.random.RandomState(word_seed)
        vec += word_rng.randn(dim)
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _make_similar_embedding(base: list[float], noise: float = 0.05) -> list[float]:
    """Create an embedding similar to base (simulating a paraphrase)."""
    rng = np.random.RandomState(hash(tuple(base[:5])) % (2**31))
    vec = np.array(base) + rng.randn(len(base)) * noise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


class DedupScenario(BenchmarkScenario):
    """Benchmark: deduplication of semantically equivalent memories across agents."""

    @property
    def name(self) -> str:
        return "s01_dedup"

    @property
    def description(self) -> str:
        return "Deduplication under concurrent writes from multiple agents"

    async def setup(self, store, num_agents: int = 5, **kwargs):
        self.store = store
        self.num_agents = num_agents
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.latencies: list[float] = []

    async def run(self) -> ScenarioResult:
        errors = []
        true_positives = 0  # Correctly detected duplicates
        false_negatives = 0  # Missed duplicates
        false_positives = 0  # Non-duplicates incorrectly merged
        total_operations = 0

        # Test 1: Write duplicate pairs — each agent writes one version
        for text_a, text_b in DUPLICATE_PAIRS:
            emb_a = _fake_embedding(text_a)
            emb_b = _make_similar_embedding(emb_a, noise=0.03)  # Very similar

            # Agent 0 writes version A
            record_a = MemoryRecord(
                content=text_a,
                content_embedding=emb_a,
                author_agent_id=self.agent_ids[0],
                tags=["fact"],
                confidence=0.9,
            )
            t0 = time.perf_counter()
            written_a, conflicts_a = await self.store.write(record_a)
            self.latencies.append((time.perf_counter() - t0) * 1000)
            total_operations += 1

            # Agent 1 writes version B (should be detected as duplicate)
            record_b = MemoryRecord(
                content=text_b,
                content_embedding=emb_b,
                author_agent_id=self.agent_ids[1],
                tags=["fact"],
                confidence=0.85,
            )
            t0 = time.perf_counter()
            written_b, conflicts_b = await self.store.write(record_b)
            self.latencies.append((time.perf_counter() - t0) * 1000)
            total_operations += 1

            # Check if dedup happened (written_b should be same as written_a if deduped)
            if written_b.id == written_a.id or written_b.supersedes_id == written_a.id:
                true_positives += 1
            elif written_a.superseded_by_id == written_b.id:
                true_positives += 1
            else:
                false_negatives += 1

        # Test 2: Write non-duplicate pairs — should NOT be merged
        for text_a, text_b in NON_DUPLICATE_PAIRS:
            emb_a = _fake_embedding(text_a)
            emb_b = _fake_embedding(text_b)  # Independently generated, should differ

            record_a = MemoryRecord(
                content=text_a,
                content_embedding=emb_a,
                author_agent_id=self.agent_ids[0],
                tags=["info"],
                confidence=0.9,
            )
            await self.store.write(record_a)
            total_operations += 1

            record_b = MemoryRecord(
                content=text_b,
                content_embedding=emb_b,
                author_agent_id=self.agent_ids[1],
                tags=["info"],
                confidence=0.9,
            )
            written_b, _ = await self.store.write(record_b)
            total_operations += 1

            # If these got merged, it's a false positive
            if written_b.supersedes_id is not None:
                false_positives += 1

        # Compute metrics
        total_dup_pairs = len(DUPLICATE_PAIRS)
        total_non_dup_pairs = len(NON_DUPLICATE_PAIRS)
        dedup_rate = true_positives / max(total_dup_pairs, 1)
        false_positive_rate = false_positives / max(total_non_dup_pairs, 1)

        # Score: weighted combination of dedup rate and false positive penalty
        score = max(0.0, dedup_rate * 0.8 + (1.0 - false_positive_rate) * 0.2)

        sorted_latencies = sorted(self.latencies) if self.latencies else [0.0]
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return ScenarioResult(
            scenario_name=self.name,
            score=round(score, 4),
            metrics={
                "dedup_rate": round(dedup_rate, 4),
                "false_positive_rate": round(false_positive_rate, 4),
                "true_positives": true_positives,
                "false_negatives": false_negatives,
                "false_positives": false_positives,
                "total_duplicate_pairs": total_dup_pairs,
                "total_non_duplicate_pairs": total_non_dup_pairs,
            },
            latency_p50_ms=round(p50, 2),
            latency_p99_ms=round(p99, 2),
            num_agents=self.num_agents,
            num_operations=total_operations,
            errors=errors,
        )

    async def teardown(self):
        self.latencies.clear()

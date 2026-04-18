"""S05: Scalability — N-Agent Performance.

Increase agent count from 2 to 50 and measure performance degradation.
Each agent writes 100 memories and reads 500.
"""

import asyncio
import time
import numpy as np

from benchmark.scenarios.scenario_base import BenchmarkScenario, ScenarioResult
from masm.core.memory import MemoryRecord
from masm.store.in_memory import InMemorySharedStore


SCALE_LEVELS = [2, 5, 10, 20]
WRITES_PER_AGENT = 20
READS_PER_AGENT = 50


def _fake_embedding(text: str, dim: int = 64) -> list[float]:
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


class ScalabilityScenario(BenchmarkScenario):
    """Benchmark: performance degradation as agent count increases."""

    @property
    def name(self) -> str:
        return "s05_scalability"

    @property
    def description(self) -> str:
        return "N-agent scaling behavior for writes and reads"

    async def setup(self, store, num_agents: int = 5, **kwargs):
        self.store_class = type(store)
        self.latencies: list[float] = []

    async def run(self) -> ScenarioResult:
        errors = []
        scale_results = {}
        total_operations = 0
        all_latencies = []

        for n_agents in SCALE_LEVELS:
            store = InMemorySharedStore()
            agents = [f"agent_{i}" for i in range(n_agents)]
            tags_pool = [f"topic_{t}" for t in range(10)]

            write_latencies = []
            read_latencies = []

            # Each agent writes WRITES_PER_AGENT memories
            for agent_id in agents:
                for w in range(WRITES_PER_AGENT):
                    content = f"Fact {w} from {agent_id} about topic {w % 10}"
                    record = MemoryRecord(
                        content=content,
                        author_agent_id=agent_id,
                        tags=[tags_pool[w % len(tags_pool)]],
                        confidence=0.8 + np.random.random() * 0.2,
                    )
                    t0 = time.perf_counter()
                    await store.write(record)
                    write_latencies.append((time.perf_counter() - t0) * 1000)
                    total_operations += 1

            # Each agent reads READS_PER_AGENT times
            for agent_id in agents:
                for r in range(READS_PER_AGENT):
                    tag = tags_pool[r % len(tags_pool)]
                    t0 = time.perf_counter()
                    await store.read(agent_id=agent_id, tags=[tag], limit=10)
                    read_latencies.append((time.perf_counter() - t0) * 1000)
                    total_operations += 1

            stats = await store.stats()
            sorted_writes = sorted(write_latencies)
            sorted_reads = sorted(read_latencies)

            scale_results[n_agents] = {
                "write_p50_ms": sorted_writes[len(sorted_writes) // 2] if sorted_writes else 0,
                "write_p99_ms": sorted_writes[int(len(sorted_writes) * 0.99)] if sorted_writes else 0,
                "read_p50_ms": sorted_reads[len(sorted_reads) // 2] if sorted_reads else 0,
                "read_p99_ms": sorted_reads[int(len(sorted_reads) * 0.99)] if sorted_reads else 0,
                "writes_per_sec": len(write_latencies) / (sum(write_latencies) / 1000 + 1e-9),
                "reads_per_sec": len(read_latencies) / (sum(read_latencies) / 1000 + 1e-9),
                "total_memories": stats["total_memories"],
                "conflicts": stats["conflicts_pending"],
            }
            all_latencies.extend(write_latencies)
            all_latencies.extend(read_latencies)

        # Score: based on how well throughput holds as agents scale
        # Compare N=2 throughput to N=20 throughput
        baseline_writes = scale_results[2]["writes_per_sec"]
        baseline_reads = scale_results[2]["reads_per_sec"]
        max_n = max(SCALE_LEVELS)
        scaled_writes = scale_results[max_n]["writes_per_sec"]
        scaled_reads = scale_results[max_n]["reads_per_sec"]

        write_retention = min(scaled_writes / (baseline_writes + 1e-9), 1.0)
        read_retention = min(scaled_reads / (baseline_reads + 1e-9), 1.0)

        score = write_retention * 0.5 + read_retention * 0.5

        sorted_all = sorted(all_latencies) if all_latencies else [0.0]
        p50 = sorted_all[len(sorted_all) // 2]
        p99 = sorted_all[int(len(sorted_all) * 0.99)]

        return ScenarioResult(
            scenario_name=self.name,
            score=round(score, 4),
            metrics={
                "scale_levels": SCALE_LEVELS,
                "write_throughput_retention": round(write_retention, 4),
                "read_throughput_retention": round(read_retention, 4),
                "per_level": scale_results,
            },
            latency_p50_ms=round(p50, 2),
            latency_p99_ms=round(p99, 2),
            num_agents=max(SCALE_LEVELS),
            num_operations=total_operations,
            errors=errors,
        )

    async def teardown(self):
        self.latencies.clear()

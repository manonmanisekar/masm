"""S03: Stale Read Detection.

Agent A updates a memory. Agent B reads the old version because of propagation
delay. Measures how quickly and reliably stale reads are prevented.
"""

import asyncio
import time
import numpy as np

from benchmark.scenarios.scenario_base import BenchmarkScenario, ScenarioResult
from masm.core.memory import MemoryRecord, MemoryState


class StalenessScenario(BenchmarkScenario):
    """Benchmark: stale read detection under varying write frequencies."""

    @property
    def name(self) -> str:
        return "s03_staleness"

    @property
    def description(self) -> str:
        return "Stale read detection under concurrent updates"

    async def setup(self, store, num_agents: int = 5, **kwargs):
        self.store = store
        self.num_agents = num_agents
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.latencies: list[float] = []

    async def run(self) -> ScenarioResult:
        errors = []
        total_reads = 0
        stale_reads = 0
        convergence_times: list[float] = []
        total_operations = 0

        # Scenario: writer agent updates a fact multiple times,
        # reader agents try to read the latest version.
        writer = self.agent_ids[0]
        readers = self.agent_ids[1:]

        # Write initial record
        initial = MemoryRecord(
            content="status: version_0",
            author_agent_id=writer,
            tags=["status", "versioned"],
            confidence=1.0,
        )
        await self.store.write(initial)
        total_operations += 1
        current_id = initial.id

        # Perform N updates, after each update all readers try to read
        num_updates = 20
        for version in range(1, num_updates + 1):
            # Writer updates the record
            t_write = time.perf_counter()
            new_record = await self.store.update(
                current_id,
                writer,
                {"content": f"status: version_{version}"},
                reason=f"update to v{version}",
            )
            write_latency = (time.perf_counter() - t_write) * 1000
            self.latencies.append(write_latency)
            total_operations += 1
            current_id = new_record.id

            # Each reader immediately reads
            t_convergence_start = time.perf_counter()
            all_readers_current = True

            for reader in readers:
                t0 = time.perf_counter()
                results = await self.store.read(
                    agent_id=reader,
                    tags=["status", "versioned"],
                    limit=1,
                )
                read_latency = (time.perf_counter() - t0) * 1000
                self.latencies.append(read_latency)
                total_reads += 1
                total_operations += 1

                if results:
                    # Check if the reader got the latest version
                    latest = results[0]
                    expected_content = f"status: version_{version}"
                    if latest.content != expected_content:
                        stale_reads += 1
                        all_readers_current = False
                else:
                    stale_reads += 1
                    all_readers_current = False

            convergence_time = (time.perf_counter() - t_convergence_start) * 1000
            convergence_times.append(convergence_time)

        # Metrics
        stale_read_rate = stale_reads / max(total_reads, 1)
        avg_convergence_ms = np.mean(convergence_times) if convergence_times else 0.0
        max_convergence_ms = max(convergence_times) if convergence_times else 0.0

        # Score: inversely proportional to stale read rate
        score = 1.0 - stale_read_rate

        sorted_latencies = sorted(self.latencies) if self.latencies else [0.0]
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return ScenarioResult(
            scenario_name=self.name,
            score=round(score, 4),
            metrics={
                "stale_read_rate": round(stale_read_rate, 4),
                "total_reads": total_reads,
                "stale_reads": stale_reads,
                "avg_convergence_ms": round(avg_convergence_ms, 4),
                "max_convergence_ms": round(max_convergence_ms, 4),
                "num_updates": num_updates,
            },
            latency_p50_ms=round(p50, 2),
            latency_p99_ms=round(p99, 2),
            num_agents=self.num_agents,
            num_operations=total_operations,
            errors=errors,
        )

    async def teardown(self):
        self.latencies.clear()

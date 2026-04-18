"""Performance benchmarking script — measures latency, throughput, and scalability.

Usage:
    python -m benchmark.perf_metrics
"""

import asyncio
import time
import statistics
import numpy as np
from masm.store.in_memory import InMemorySharedStore
from masm.core.memory import MemoryRecord
from benchmark.baselines.naive_shared_dict import NaiveSharedDictStore


async def measure_write_throughput(store, num_writes: int, agent_id: str = "bench") -> dict:
    """Measure write throughput and latency distribution."""
    latencies = []
    for i in range(num_writes):
        record = MemoryRecord(
            content=f"Benchmark fact number {i} for throughput testing",
            author_agent_id=agent_id,
            tags=[f"topic_{i % 10}", "benchmark"],
            confidence=0.9,
        )
        t0 = time.perf_counter()
        await store.write(record)
        latencies.append((time.perf_counter() - t0) * 1000)

    total_time_ms = sum(latencies)
    return {
        "operations": num_writes,
        "total_time_ms": round(total_time_ms, 2),
        "throughput_ops_sec": round(num_writes / (total_time_ms / 1000), 1),
        "latency_p50_ms": round(statistics.median(latencies), 4),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
        "latency_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 4),
        "latency_max_ms": round(max(latencies), 4),
    }


async def measure_read_throughput(store, num_reads: int, agent_id: str = "reader") -> dict:
    """Measure read throughput and latency distribution."""
    tags_pool = [f"topic_{i}" for i in range(10)]
    latencies = []
    for i in range(num_reads):
        tag = tags_pool[i % len(tags_pool)]
        t0 = time.perf_counter()
        await store.read(agent_id=agent_id, tags=[tag], limit=10)
        latencies.append((time.perf_counter() - t0) * 1000)

    total_time_ms = sum(latencies)
    return {
        "operations": num_reads,
        "total_time_ms": round(total_time_ms, 2),
        "throughput_ops_sec": round(num_reads / (total_time_ms / 1000), 1),
        "latency_p50_ms": round(statistics.median(latencies), 4),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
        "latency_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 4),
        "latency_max_ms": round(max(latencies), 4),
    }


async def measure_conflict_detection_overhead(store, num_writes: int) -> dict:
    """Measure overhead of conflict detection as store fills up."""
    latencies_early = []
    latencies_late = []

    for i in range(num_writes):
        record = MemoryRecord(
            content=f"Fact variant {i} about topic {i % 5}",
            author_agent_id=f"agent_{i % 3}",
            tags=[f"topic_{i % 5}"],
            confidence=0.8 + (i % 20) * 0.01,
        )
        t0 = time.perf_counter()
        await store.write(record)
        elapsed = (time.perf_counter() - t0) * 1000

        if i < num_writes // 4:
            latencies_early.append(elapsed)
        elif i >= num_writes * 3 // 4:
            latencies_late.append(elapsed)

    return {
        "total_writes": num_writes,
        "early_p50_ms": round(statistics.median(latencies_early), 4),
        "late_p50_ms": round(statistics.median(latencies_late), 4),
        "degradation_factor": round(
            statistics.median(latencies_late) / max(statistics.median(latencies_early), 0.001), 2
        ),
    }


async def measure_concurrent_agents(store_class, agent_counts: list[int]) -> dict:
    """Measure throughput as concurrent agent count increases."""
    results = {}
    writes_per_agent = 50
    reads_per_agent = 100

    for n in agent_counts:
        store = store_class()
        agents = [f"agent_{i}" for i in range(n)]

        # Writes
        t0 = time.perf_counter()
        for agent_id in agents:
            for w in range(writes_per_agent):
                await store.write(MemoryRecord(
                    content=f"Fact {w} from {agent_id}",
                    author_agent_id=agent_id,
                    tags=[f"topic_{w % 5}"],
                    confidence=0.9,
                ))
        write_time = time.perf_counter() - t0
        total_writes = n * writes_per_agent

        # Reads
        t0 = time.perf_counter()
        for agent_id in agents:
            for r in range(reads_per_agent):
                await store.read(agent_id=agent_id, tags=[f"topic_{r % 5}"], limit=10)
        read_time = time.perf_counter() - t0
        total_reads = n * reads_per_agent

        results[n] = {
            "agents": n,
            "write_throughput": round(total_writes / write_time, 1),
            "read_throughput": round(total_reads / read_time, 1),
            "write_latency_avg_ms": round((write_time / total_writes) * 1000, 4),
            "read_latency_avg_ms": round((read_time / total_reads) * 1000, 4),
        }

    return results


async def run_all():
    print("=" * 70)
    print("MASM Performance Metrics")
    print("=" * 70)

    for label, store_class in [("MASM InMemoryStore", InMemorySharedStore), ("Naive Shared Dict", NaiveSharedDictStore)]:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        store = store_class()

        # Write throughput
        write_results = await measure_write_throughput(store, num_writes=1000)
        print(f"\n  Write Throughput ({write_results['operations']} ops):")
        print(f"    Throughput:  {write_results['throughput_ops_sec']:>10,.0f} ops/sec")
        print(f"    P50 latency: {write_results['latency_p50_ms']:>10.4f} ms")
        print(f"    P99 latency: {write_results['latency_p99_ms']:>10.4f} ms")

        # Read throughput
        read_results = await measure_read_throughput(store, num_reads=2000)
        print(f"\n  Read Throughput ({read_results['operations']} ops):")
        print(f"    Throughput:  {read_results['throughput_ops_sec']:>10,.0f} ops/sec")
        print(f"    P50 latency: {read_results['latency_p50_ms']:>10.4f} ms")
        print(f"    P99 latency: {read_results['latency_p99_ms']:>10.4f} ms")

        # Conflict detection overhead
        conflict_store = store_class()
        conflict_results = await measure_conflict_detection_overhead(conflict_store, num_writes=500)
        print(f"\n  Conflict Detection Overhead ({conflict_results['total_writes']} writes):")
        print(f"    Early P50:   {conflict_results['early_p50_ms']:>10.4f} ms")
        print(f"    Late P50:    {conflict_results['late_p50_ms']:>10.4f} ms")
        print(f"    Degradation: {conflict_results['degradation_factor']:>10.2f}x")

    # Scalability comparison
    print(f"\n{'─' * 70}")
    print("  Scalability: Agent Count vs Throughput")
    print(f"{'─' * 70}")

    agent_counts = [2, 5, 10, 20]
    for label, store_class in [("MASM", InMemorySharedStore), ("Naive", NaiveSharedDictStore)]:
        scale_results = await measure_concurrent_agents(store_class, agent_counts)
        print(f"\n  {label}:")
        print(f"    {'Agents':>8} {'Writes/s':>12} {'Reads/s':>12} {'W-Lat(ms)':>12} {'R-Lat(ms)':>12}")
        for n, data in scale_results.items():
            print(f"    {data['agents']:>8} {data['write_throughput']:>12,.0f} {data['read_throughput']:>12,.0f} {data['write_latency_avg_ms']:>12.4f} {data['read_latency_avg_ms']:>12.4f}")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(run_all())

#!/usr/bin/env python3
"""Run MASM benchmarks with ASCII visualization of results."""

import asyncio

from benchmark.runner import MASMBenchmarkRunner
from masm.store.in_memory import InMemorySharedStore
from benchmark.baselines.naive_shared_dict import NaiveSharedDictStore
from masm.tools.benchmark import BenchmarkVisualizer


async def main():
    viz = BenchmarkVisualizer()

    # --- Run InMemorySharedStore ---
    print("Running benchmarks for InMemorySharedStore...")
    masm_store = InMemorySharedStore()
    masm_runner = MASMBenchmarkRunner(store=masm_store)
    masm_results = await masm_runner.run_all(num_agents=5)

    print()
    print(viz.render(masm_results))

    # --- Run NaiveSharedDictStore ---
    print("\n\nRunning benchmarks for NaiveSharedDictStore...")
    naive_store = NaiveSharedDictStore()
    naive_runner = MASMBenchmarkRunner(store=naive_store)
    naive_results = await naive_runner.run_all(num_agents=5)

    print()
    print(viz.render(naive_results))

    # --- Side-by-side comparison ---
    print("\n")
    print(viz.render_comparison([masm_results, naive_results]))


if __name__ == "__main__":
    asyncio.run(main())

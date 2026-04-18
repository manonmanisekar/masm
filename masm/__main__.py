"""CLI entry point for `python -m masm` or `masm` command."""

import argparse
import asyncio
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="masm",
        description="MASM — Multi-Agent Shared Memory Benchmark & Framework",
    )
    subparsers = parser.add_subparsers(dest="command")

    # masm benchmark
    bench = subparsers.add_parser("benchmark", help="Run benchmark scenarios")
    bench.add_argument("--store", default="masm", choices=["masm", "naive"], help="Store backend")
    bench.add_argument("--scenarios", default=None, help="Comma-separated scenario list")
    bench.add_argument("--agents", type=int, default=5, help="Number of agents")
    bench.add_argument("--output", default=None, help="Output file path")
    bench.add_argument("--format", default="markdown", choices=["markdown", "json"])

    # masm compare
    compare = subparsers.add_parser("compare", help="Compare MASM vs naive baseline")
    compare.add_argument("--agents", type=int, default=5, help="Number of agents")
    compare.add_argument("--scenarios", default=None, help="Comma-separated scenario list")
    compare.add_argument("--output", default=None, help="Output file path")

    # masm report
    report = subparsers.add_parser("report", help="Format benchmark results")
    report.add_argument("results_file", help="Path to results JSON file")
    report.add_argument("--format", default="markdown", choices=["markdown", "json"])

    # masm visualize
    viz = subparsers.add_parser("visualize", help="Visualize memory conflicts, provenance, and relevance")
    viz.add_argument("--demo", action="store_true", help="Use built-in demo data")
    viz.add_argument(
        "--type", default="all",
        choices=["conflicts", "provenance", "relevance", "all"],
        help="Which visualization to render (default: all)",
    )
    viz.add_argument(
        "--format", default="text",
        choices=["text", "dot"],
        help="Output format: text (ASCII) or dot (Graphviz DOT)",
    )
    viz.add_argument("--agent", default=None, help="Agent ID for relevance ranking")
    viz.add_argument("--query", default=None, help="Query string label for relevance view")
    viz.add_argument("--record", default=None, help="Record ID for chain/breakdown deep-dive")
    viz.add_argument("--output", default=None, help="Write output to this file instead of stdout")

    args = parser.parse_args()

    if args.command == "benchmark":
        asyncio.run(_run_benchmark(args))
    elif args.command == "compare":
        asyncio.run(_run_compare(args))
    elif args.command == "report":
        _run_report(args)
    elif args.command == "visualize":
        asyncio.run(_run_visualize(args))
    else:
        parser.print_help()


async def _run_benchmark(args) -> None:
    from masm.store.in_memory import InMemorySharedStore
    from benchmark.baselines.naive_shared_dict import NaiveSharedDictStore
    from benchmark.runner import MASMBenchmarkRunner

    if args.store == "masm":
        store = InMemorySharedStore()
    elif args.store == "naive":
        store = NaiveSharedDictStore()
    else:
        print(f"Unknown store: {args.store}", file=sys.stderr)
        sys.exit(1)

    scenario_names = args.scenarios.split(",") if args.scenarios else None

    runner = MASMBenchmarkRunner(store=store)
    print("MASM Benchmark v0.2.0")
    print(f"Store: {args.store} | Agents: {args.agents}")
    print("=" * 60)

    results = await runner.run_all(num_agents=args.agents, scenario_names=scenario_names)

    output = runner.report(results, format=args.format)
    print()
    print(output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to {args.output}")


async def _run_compare(args) -> None:
    from masm.store.in_memory import InMemorySharedStore
    from benchmark.baselines.naive_shared_dict import NaiveSharedDictStore
    from benchmark.runner import MASMBenchmarkRunner
    from benchmark.reporter import BenchmarkReporter

    scenario_names = args.scenarios.split(",") if args.scenarios else None
    all_results = []

    for label, store in [("MASM", InMemorySharedStore()), ("Naive", NaiveSharedDictStore())]:
        print(f"\n{'=' * 60}")
        print(f"Running: {label}")
        print(f"{'=' * 60}")
        runner = MASMBenchmarkRunner(store=store)
        results = await runner.run_all(num_agents=args.agents, scenario_names=scenario_names)
        all_results.append(results)

    print(f"\n{'=' * 60}")
    print(BenchmarkReporter.compare(all_results))

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"comparisons": all_results}, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


def _run_report(args) -> None:
    from benchmark.reporter import BenchmarkReporter

    with open(args.results_file) as f:
        results = json.load(f)

    if args.format == "markdown":
        print(BenchmarkReporter.to_markdown(results))
    else:
        print(BenchmarkReporter.to_json(results))


async def _run_visualize(args) -> None:
    from masm.tools.demo import build_demo_store, build_demo_agents
    from masm.tools import ConflictVisualizer, ProvenanceVisualizer, RelevanceVisualizer
    from masm.core.memory import MemoryState

    if not args.demo:
        print("Error: only --demo mode is currently supported.", file=sys.stderr)
        print("  masm visualize --demo", file=sys.stderr)
        sys.exit(1)

    store = await build_demo_store()
    agents = build_demo_agents()

    output_parts: list[str] = []

    if args.type in ("conflicts", "all"):
        conflicts = await store.list_conflicts()
        records_list = await store.list_records()
        records_map = {r.id: r for r in records_list}
        viz = ConflictVisualizer(conflicts=conflicts, records=records_map)
        if args.format == "dot":
            output_parts.append(viz.to_dot())
        else:
            output_parts.append(viz.render())

    if args.type in ("provenance", "all"):
        records = await store.list_records()
        audit_log = await store.get_audit_log()
        prov = ProvenanceVisualizer(records=records, audit_log=audit_log)
        if args.format == "dot":
            output_parts.append(prov.to_dot())
        elif args.record:
            output_parts.append(prov.render_chain(args.record))
        else:
            output_parts.append(prov.render())
            output_parts.append(prov.render_audit_trail())

    if args.type in ("relevance", "all"):
        records = await store.list_records(states=[MemoryState.ACTIVE])
        if args.agent and args.agent in agents:
            agent = agents[args.agent]
        elif args.agent:
            from masm.core.agent import Agent
            agent = Agent(id=args.agent)
        else:
            agent = agents["analyst"]
        rel = RelevanceVisualizer(records=records, agent=agent, query=args.query)
        if args.record:
            output_parts.append(rel.render_breakdown(args.record))
        else:
            output_parts.append(rel.render())

    result = "\n\n".join(output_parts)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"Visualization written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()

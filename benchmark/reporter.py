"""Results formatting for MASM benchmarks — JSON, Markdown, and leaderboard output."""

import json
from typing import Any


class BenchmarkReporter:
    """Formats benchmark results into various output formats."""

    @staticmethod
    def to_markdown(results: dict) -> str:
        """Generate a Markdown report from benchmark results."""
        lines = [
            "# MASM Benchmark Results",
            "",
            f"**Composite Score: {results['composite_score']:.2%}**",
            f"**Store: {results['store_implementation']}**",
            f"**Agents: {results['num_agents']}**",
            f"**Date: {results['timestamp']}**",
            "",
            "## Scenario Results",
            "",
            "| Scenario | Score | P50 (ms) | P99 (ms) | Operations | Errors |",
            "|----------|-------|----------|----------|------------|--------|",
        ]
        for name, data in results["scenarios"].items():
            lines.append(
                f"| {name} | {data['score']:.2%} | "
                f"{data['latency_p50_ms']:.1f} | "
                f"{data['latency_p99_ms']:.1f} | "
                f"{data['num_operations']} | "
                f"{len(data.get('errors', []))} |"
            )

        # Per-scenario details
        lines.extend(["", "## Detailed Metrics", ""])
        for name, data in results["scenarios"].items():
            lines.append(f"### {name}")
            lines.append("")
            for metric, value in data.get("metrics", {}).items():
                if isinstance(value, float):
                    lines.append(f"- **{metric}**: {value:.4f}")
                else:
                    lines.append(f"- **{metric}**: {value}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_json(results: dict) -> str:
        """Generate JSON output."""
        return json.dumps(results, indent=2, default=str)

    @staticmethod
    def to_leaderboard_entry(results: dict) -> dict:
        """Generate a leaderboard-compatible entry."""
        return {
            "store": results["store_implementation"],
            "composite_score": results["composite_score"],
            "num_agents": results["num_agents"],
            "timestamp": results["timestamp"],
            "scenario_scores": {
                name: data["score"]
                for name, data in results["scenarios"].items()
            },
        }

    @staticmethod
    def compare(results_list: list[dict]) -> str:
        """Compare multiple benchmark results side by side."""
        if not results_list:
            return "No results to compare."

        # Collect all scenario names
        all_scenarios = set()
        for r in results_list:
            all_scenarios.update(r["scenarios"].keys())

        header = "| Scenario | " + " | ".join(r["store_implementation"] for r in results_list) + " |"
        sep = "|----------|" + "|".join("--------" for _ in results_list) + "|"

        lines = [
            "# MASM Benchmark Comparison",
            "",
            header,
            sep,
        ]

        # Composite row
        scores = " | ".join(f"{r['composite_score']:.2%}" for r in results_list)
        lines.append(f"| **Composite** | {scores} |")

        for scenario in sorted(all_scenarios):
            scores = []
            for r in results_list:
                if scenario in r["scenarios"]:
                    scores.append(f"{r['scenarios'][scenario]['score']:.2%}")
                else:
                    scores.append("N/A")
            lines.append(f"| {scenario} | {' | '.join(scores)} |")

        return "\n".join(lines)

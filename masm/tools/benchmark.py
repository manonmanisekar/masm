"""Benchmark & test results visualizer — ASCII charts for scenario scores, latencies, and comparisons."""

from __future__ import annotations

from masm.tools._base import ascii_bar, format_table, section_header


class BenchmarkVisualizer:
    """
    Renders benchmark results as rich ASCII dashboards.

    Accepts the dict returned by MASMBenchmarkRunner.run_all().
    """

    def render(self, results: dict) -> str:
        """Full dashboard: header, score chart, latency table, details."""
        parts = [
            self._render_header(results),
            self._render_score_chart(results),
            self._render_latency_table(results),
            self._render_details(results),
        ]
        return "\n\n".join(parts)

    def render_score_chart(self, results: dict) -> str:
        """Standalone score bar chart."""
        return self._render_score_chart(results)

    def render_comparison(self, results_list: list[dict]) -> str:
        """Side-by-side comparison of multiple benchmark runs."""
        if not results_list:
            return "No results to compare."
        return "\n\n".join([
            self._render_comparison_header(results_list),
            self._render_comparison_table(results_list),
            self._render_comparison_chart(results_list),
        ])

    # ---- Header ----

    def _render_header(self, results: dict) -> str:
        composite = results.get("composite_score", 0)
        store = results.get("store_implementation", "Unknown")
        agents = results.get("num_agents", 0)
        ts = results.get("timestamp", "")
        bar = ascii_bar(composite, 1.0, width=40)
        grade = self._grade(composite)

        lines = [
            section_header("MASM BENCHMARK RESULTS"),
            f"  Store:      {store}",
            f"  Agents:     {agents}",
            f"  Date:       {ts}",
            "",
            f"  COMPOSITE   {bar} {composite:.1%}  {grade}",
        ]
        return "\n".join(lines)

    # ---- Score bar chart ----

    def _render_score_chart(self, results: dict) -> str:
        scenarios = results.get("scenarios", {})
        if not scenarios:
            return "  (no scenarios)"

        lines = [section_header("SCENARIO SCORES"), ""]
        max_name_len = max(len(n) for n in scenarios) if scenarios else 10

        for name, data in sorted(scenarios.items()):
            score = data.get("score", 0)
            bar = ascii_bar(score, 1.0, width=35)
            errors = len(data.get("errors", []))
            err_flag = f"  [{errors} err]" if errors else ""
            lines.append(f"  {name:<{max_name_len}}  {bar} {score:>6.1%}{err_flag}")

        return "\n".join(lines)

    # ---- Latency table ----

    def _render_latency_table(self, results: dict) -> str:
        scenarios = results.get("scenarios", {})
        if not scenarios:
            return ""

        headers = ["Scenario", "Score", "P50 ms", "P99 ms", "Ops", "Errors"]
        widths = [22, 8, 10, 10, 8, 8]
        rows = []
        for name, data in sorted(scenarios.items()):
            rows.append([
                name,
                f"{data.get('score', 0):.1%}",
                f"{data.get('latency_p50_ms', 0):.2f}",
                f"{data.get('latency_p99_ms', 0):.2f}",
                str(data.get("num_operations", 0)),
                str(len(data.get("errors", []))),
            ])

        lines = [section_header("LATENCY & OPERATIONS"), ""]
        lines.append(format_table(headers, rows, widths))
        return "\n".join(lines)

    # ---- Per-scenario details ----

    def _render_details(self, results: dict) -> str:
        scenarios = results.get("scenarios", {})
        if not scenarios:
            return ""

        lines = [section_header("DETAILED METRICS"), ""]
        for name, data in sorted(scenarios.items()):
            metrics = data.get("metrics", {})
            if not metrics:
                continue
            lines.append(f"  --- {name} ---")
            for k, v in metrics.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4f}")
                else:
                    lines.append(f"    {k}: {v}")
            lines.append("")
        return "\n".join(lines)

    # ---- Comparison ----

    def _render_comparison_header(self, results_list: list[dict]) -> str:
        lines = [section_header("BENCHMARK COMPARISON"), ""]
        for i, r in enumerate(results_list):
            lines.append(
                f"  [{i+1}] {r.get('store_implementation', '?')}"
                f"  composite={r.get('composite_score', 0):.1%}"
                f"  agents={r.get('num_agents', 0)}"
            )
        return "\n".join(lines)

    def _render_comparison_table(self, results_list: list[dict]) -> str:
        all_scenarios: set[str] = set()
        for r in results_list:
            all_scenarios.update(r.get("scenarios", {}).keys())

        store_names = [r.get("store_implementation", "?")[:16] for r in results_list]
        headers = ["Scenario"] + store_names
        widths = [22] + [max(len(n), 10) for n in store_names]

        rows = []
        # Composite row
        row = ["** COMPOSITE **"]
        for r in results_list:
            row.append(f"{r.get('composite_score', 0):.1%}")
        rows.append(row)

        for scenario in sorted(all_scenarios):
            row = [scenario]
            for r in results_list:
                s = r.get("scenarios", {}).get(scenario, {})
                if s:
                    row.append(f"{s.get('score', 0):.1%}")
                else:
                    row.append("N/A")
            rows.append(row)

        return format_table(headers, rows, widths)

    def _render_comparison_chart(self, results_list: list[dict]) -> str:
        all_scenarios: set[str] = set()
        for r in results_list:
            all_scenarios.update(r.get("scenarios", {}).keys())

        store_names = [r.get("store_implementation", "?")[:16] for r in results_list]
        markers = ["#", "=", "*", "~"]

        lines = [
            "",
            "  Legend: " + "  ".join(
                f"[{markers[i % len(markers)]}] {name}"
                for i, name in enumerate(store_names)
            ),
            "",
        ]

        for scenario in sorted(all_scenarios):
            lines.append(f"  {scenario}:")
            for i, r in enumerate(results_list):
                s = r.get("scenarios", {}).get(scenario, {})
                score = s.get("score", 0) if s else 0
                bar = ascii_bar(score, 1.0, width=30, fill=markers[i % len(markers)])
                lines.append(f"    {store_names[i]:<16} {bar} {score:.1%}")
            lines.append("")

        return "\n".join(lines)

    # ---- Helpers ----

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 0.95:
            return "A+"
        if score >= 0.90:
            return "A"
        if score >= 0.80:
            return "B"
        if score >= 0.70:
            return "C"
        if score >= 0.60:
            return "D"
        return "F"

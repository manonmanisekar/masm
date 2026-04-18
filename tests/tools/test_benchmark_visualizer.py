"""Tests for BenchmarkVisualizer."""

from masm.tools.benchmark import BenchmarkVisualizer


def _sample_results(store_name="InMemorySharedStore", composite=0.85):
    return {
        "composite_score": composite,
        "timestamp": "2026-04-15T00:00:00+00:00",
        "num_agents": 5,
        "store_implementation": store_name,
        "scenarios": {
            "s01_dedup": {
                "score": 0.95,
                "latency_p50_ms": 0.12,
                "latency_p99_ms": 0.45,
                "num_operations": 100,
                "errors": [],
                "metrics": {"dedup_rate": 0.92, "false_positives": 0},
            },
            "s02_conflict": {
                "score": 0.70,
                "latency_p50_ms": 0.30,
                "latency_p99_ms": 1.20,
                "num_operations": 50,
                "errors": ["timeout on op 42"],
                "metrics": {"conflicts_detected": 5, "conflicts_resolved": 4},
            },
        },
    }


class TestBenchmarkVisualizerRender:
    def test_render_contains_composite(self):
        viz = BenchmarkVisualizer()
        output = viz.render(_sample_results())
        assert "85.0%" in output
        assert "COMPOSITE" in output

    def test_render_contains_scenario_names(self):
        viz = BenchmarkVisualizer()
        output = viz.render(_sample_results())
        assert "s01_dedup" in output
        assert "s02_conflict" in output

    def test_render_contains_store_name(self):
        viz = BenchmarkVisualizer()
        output = viz.render(_sample_results())
        assert "InMemorySharedStore" in output

    def test_render_shows_error_count(self):
        viz = BenchmarkVisualizer()
        output = viz.render(_sample_results())
        assert "1 err" in output

    def test_render_shows_latencies(self):
        viz = BenchmarkVisualizer()
        output = viz.render(_sample_results())
        assert "0.12" in output
        assert "1.20" in output

    def test_render_shows_metrics(self):
        viz = BenchmarkVisualizer()
        output = viz.render(_sample_results())
        assert "dedup_rate" in output
        assert "conflicts_detected" in output

    def test_render_empty_scenarios(self):
        viz = BenchmarkVisualizer()
        results = _sample_results()
        results["scenarios"] = {}
        output = viz.render(results)
        assert "MASM BENCHMARK" in output

    def test_grade_mapping(self):
        viz = BenchmarkVisualizer()
        assert viz._grade(0.96) == "A+"
        assert viz._grade(0.91) == "A"
        assert viz._grade(0.82) == "B"
        assert viz._grade(0.72) == "C"
        assert viz._grade(0.62) == "D"
        assert viz._grade(0.40) == "F"


class TestBenchmarkVisualizerScoreChart:
    def test_score_chart_standalone(self):
        viz = BenchmarkVisualizer()
        output = viz.render_score_chart(_sample_results())
        assert "SCENARIO SCORES" in output
        assert "95.0%" in output
        assert "70.0%" in output


class TestBenchmarkVisualizerComparison:
    def test_comparison_two_stores(self):
        viz = BenchmarkVisualizer()
        r1 = _sample_results("InMemorySharedStore", 0.85)
        r2 = _sample_results("NaiveSharedDictStore", 0.40)
        output = viz.render_comparison([r1, r2])
        assert "InMemorySharedStore" in output
        assert "NaiveSharedDictStore" in output
        assert "COMPARISON" in output

    def test_comparison_empty(self):
        viz = BenchmarkVisualizer()
        output = viz.render_comparison([])
        assert "No results" in output

    def test_comparison_contains_composite(self):
        viz = BenchmarkVisualizer()
        r1 = _sample_results("Store1", 0.90)
        r2 = _sample_results("Store2", 0.50)
        output = viz.render_comparison([r1, r2])
        assert "COMPOSITE" in output

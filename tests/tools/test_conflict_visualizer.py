"""Tests for ConflictVisualizer."""

import pytest
from masm.core.memory import (
    ConflictEvent,
    ConflictSeverity,
    ConflictStrategy,
    MemoryRecord,
    MemoryState,
    OperationStatus,
)
from masm.tools.conflict import ConflictVisualizer
from masm.tools.demo import build_demo_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(content: str, agent: str = "agent_a", tags: tuple = ("topic",)) -> MemoryRecord:
    return MemoryRecord(content=content, author_agent_id=agent, tags=tags)


def _make_conflict(
    rec_a: MemoryRecord,
    rec_b: MemoryRecord,
    resolved: bool = False,
    severity: ConflictSeverity = ConflictSeverity.MEDIUM,
    strategy: ConflictStrategy = ConflictStrategy.LAST_WRITE_WINS,
    reasoning: str = "LWW: b is more recent",
) -> ConflictEvent:
    return ConflictEvent(
        memory_a_id=rec_a.id,
        memory_b_id=rec_b.id,
        resolved=resolved,
        severity=severity,
        strategy=strategy,
        reasoning=reasoning,
        resolved_memory_id=rec_b.id if resolved else None,
        status=OperationStatus.SUCCESS if resolved else OperationStatus.PENDING,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConflictVisualizerRender:
    def test_empty_store_renders_without_crash(self):
        viz = ConflictVisualizer(conflicts=[], records={})
        output = viz.render()
        assert "CONFLICT GRAPH" in output
        assert "0 total" in output
        assert "no conflicts" in output

    def test_renders_unresolved_conflict(self):
        a = _make_record("Sky is blue", "agent_a")
        b = _make_record("Sky is green", "agent_b")
        conflict = _make_conflict(a, b, resolved=False, severity=ConflictSeverity.HIGH)
        viz = ConflictVisualizer(conflicts=[conflict], records={a.id: a, b.id: b})
        output = viz.render()
        assert "UNRESOLVED" in output
        assert "HIGH" in output
        assert "Sky is blue" in output
        assert "Sky is green" in output

    def test_renders_resolved_conflict(self):
        a = _make_record("Budget $50k", "agent_a")
        b = _make_record("Budget $75k", "agent_b")
        conflict = _make_conflict(a, b, resolved=True, strategy=ConflictStrategy.LAST_WRITE_WINS)
        viz = ConflictVisualizer(conflicts=[conflict], records={a.id: a, b.id: b})
        output = viz.render()
        assert "RESOLVED" in output
        assert "lww" in output
        assert "LWW: b is more recent" in output
        # Winner shown
        assert b.id[:8] in output

    def test_renders_strategy_in_output(self):
        a = _make_record("Fact A", "agent_a")
        b = _make_record("Fact B", "agent_b")
        conflict = _make_conflict(a, b, strategy=ConflictStrategy.AUTHORITY_RANK)
        viz = ConflictVisualizer(conflicts=[conflict], records={a.id: a, b.id: b})
        output = viz.render()
        assert "authority_rank" in output

    def test_summary_counts_match(self):
        a = _make_record("X", "a1")
        b = _make_record("Y", "a2")
        c = _make_record("Z", "a3")
        c1 = _make_conflict(a, b, resolved=False, severity=ConflictSeverity.HIGH)
        c2 = _make_conflict(a, c, resolved=True, severity=ConflictSeverity.LOW)
        viz = ConflictVisualizer(
            conflicts=[c1, c2],
            records={a.id: a, b.id: b, c.id: c},
        )
        s = viz.summary()
        assert s["total"] == 2
        assert s["resolved"] == 1
        assert s["unresolved"] == 1
        assert s["by_severity"]["high"] == 1
        assert s["by_severity"]["low"] == 1

    def test_summary_empty(self):
        viz = ConflictVisualizer(conflicts=[], records={})
        s = viz.summary()
        assert s["total"] == 0
        assert s["resolved"] == 0
        assert s["unresolved"] == 0


class TestConflictVisualizerDot:
    def test_to_dot_contains_digraph(self):
        a = _make_record("Alpha", "a1")
        b = _make_record("Beta", "a2")
        conflict = _make_conflict(a, b)
        viz = ConflictVisualizer(conflicts=[conflict], records={a.id: a, b.id: b})
        dot = viz.to_dot()
        assert "digraph" in dot

    def test_to_dot_empty_no_crash(self):
        viz = ConflictVisualizer(conflicts=[], records={})
        dot = viz.to_dot()
        assert "digraph" in dot

    def test_to_dot_unresolved_edge_is_red(self):
        a = _make_record("P", "a1")
        b = _make_record("Q", "a2")
        conflict = _make_conflict(a, b, resolved=False)
        viz = ConflictVisualizer(conflicts=[conflict], records={a.id: a, b.id: b})
        dot = viz.to_dot()
        assert "red" in dot

    def test_to_dot_resolved_edge_is_gray(self):
        a = _make_record("P", "a1")
        b = _make_record("Q", "a2")
        conflict = _make_conflict(a, b, resolved=True)
        viz = ConflictVisualizer(conflicts=[conflict], records={a.id: a, b.id: b})
        dot = viz.to_dot()
        assert "gray" in dot


class TestConflictVisualizerAsync:
    @pytest.mark.asyncio
    async def test_render_async_uses_store(self):
        store = await build_demo_store()
        viz = ConflictVisualizer(store=store)
        output = await viz.render_async()
        assert "CONFLICT GRAPH" in output

    @pytest.mark.asyncio
    async def test_render_async_no_store_raises(self):
        viz = ConflictVisualizer()
        with pytest.raises(RuntimeError):
            await viz.render_async()

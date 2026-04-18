"""Tests for ProvenanceVisualizer."""

import pytest
from masm.core.memory import MemoryRecord, MemoryState
from masm.tools.provenance import ProvenanceVisualizer
from masm.tools.demo import build_demo_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    content: str,
    agent: str = "agent_a",
    tags: tuple = ("topic",),
    state: MemoryState = MemoryState.ACTIVE,
    supersedes_id: str | None = None,
    superseded_by_id: str | None = None,
    version: int = 1,
) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        author_agent_id=agent,
        tags=tags,
        state=state,
        supersedes_id=supersedes_id,
        superseded_by_id=superseded_by_id,
        version=version,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProvenanceVisualizerRender:
    def test_empty_renders_without_crash(self):
        viz = ProvenanceVisualizer(records=[], audit_log=[])
        output = viz.render()
        assert "PROVENANCE TREE" in output

    def test_single_record_appears_in_output(self):
        r = _make_record("Project deadline is March 15", agent="pm_agent", tags=("deadline",))
        viz = ProvenanceVisualizer(records=[r])
        output = viz.render()
        assert "Project deadline is March 15" in output
        assert "pm_agent" in output

    def test_state_icon_active(self):
        r = _make_record("Active fact", state=MemoryState.ACTIVE)
        viz = ProvenanceVisualizer(records=[r])
        output = viz.render()
        assert "+" in output

    def test_state_icon_superseded(self):
        r = _make_record("Old fact", state=MemoryState.SUPERSEDED)
        viz = ProvenanceVisualizer(records=[r])
        output = viz.render()
        assert "~" in output

    def test_version_chain_shows_superseded_by(self):
        r1 = _make_record("Budget $50k", agent="sales", version=1)
        r2 = _make_record(
            "Budget $75k",
            agent="account_mgr",
            version=2,
            supersedes_id=r1.id,
            state=MemoryState.ACTIVE,
        )
        r1_superseded = _make_record(
            "Budget $50k",
            agent="sales",
            version=1,
            state=MemoryState.SUPERSEDED,
            superseded_by_id=r2.id,
        )
        # Use the same IDs
        import dataclasses
        r1_superseded = dataclasses.replace(r1_superseded, id=r1.id)

        viz = ProvenanceVisualizer(records=[r1_superseded, r2])
        output = viz.render()
        assert "superseded by" in output

    def test_agent_attribution_in_footer(self):
        r1 = _make_record("Fact A", agent="agent_alpha")
        r2 = _make_record("Fact B", agent="agent_beta")
        viz = ProvenanceVisualizer(records=[r1, r2])
        output = viz.render()
        assert "Agent Attribution" in output
        assert "agent_alpha" in output
        assert "agent_beta" in output


class TestProvenanceVisualizerChain:
    def test_render_chain_single_record(self):
        r = _make_record("Standalone fact", agent="bot")
        viz = ProvenanceVisualizer(records=[r])
        output = viz.render_chain(r.id)
        assert "Standalone fact" in output
        assert "no further versions" in output

    def test_render_chain_two_versions(self):
        r1 = _make_record("Old version", agent="bot", version=1)
        r2 = _make_record(
            "New version",
            agent="bot",
            version=2,
            supersedes_id=r1.id,
        )
        viz = ProvenanceVisualizer(records=[r1, r2])
        output = viz.render_chain(r2.id)
        # Chain walks backward: r1 is the ancestor
        assert "Old version" in output
        assert "New version" in output
        # Indentation shows lineage
        assert "\u2514\u2500" in output  # └─

    def test_render_chain_missing_record(self):
        viz = ProvenanceVisualizer(records=[])
        output = viz.render_chain("nonexistent-id")
        assert "NOT FOUND" in output


class TestProvenanceVisualizerAudit:
    def test_audit_trail_empty(self):
        viz = ProvenanceVisualizer(records=[], audit_log=[])
        output = viz.render_audit_trail()
        assert "AUDIT TRAIL" in output
        assert "no audit entries" in output

    def test_audit_trail_shows_operations(self):
        entries = [
            {
                "id": "e1",
                "timestamp": "2026-04-15T10:00:01",
                "operation": "write",
                "agent_id": "sales_agent",
                "record_id": "abc12345",
                "details": {},
            },
            {
                "id": "e2",
                "timestamp": "2026-04-15T10:00:02",
                "operation": "read",
                "agent_id": "analyst",
                "record_id": None,
                "details": {},
            },
        ]
        viz = ProvenanceVisualizer(records=[], audit_log=entries)
        output = viz.render_audit_trail()
        assert "write" in output
        assert "sales_agent" in output
        assert "analyst" in output

    def test_audit_trail_filters_by_record_id(self):
        entries = [
            {"id": "e1", "timestamp": "2026-04-15T10:00:01", "operation": "write",
             "agent_id": "a", "record_id": "target-id", "details": {}},
            {"id": "e2", "timestamp": "2026-04-15T10:00:02", "operation": "read",
             "agent_id": "b", "record_id": "other-id", "details": {}},
        ]
        viz = ProvenanceVisualizer(records=[], audit_log=entries)
        output = viz.render_audit_trail(record_id="target-id")
        assert "write" in output
        assert "read" not in output


class TestProvenanceVisualizerDot:
    def test_to_dot_contains_digraph(self):
        r = _make_record("Fact", agent="bot")
        viz = ProvenanceVisualizer(records=[r])
        dot = viz.to_dot()
        assert "digraph" in dot

    def test_to_dot_includes_supersedes_edge(self):
        r1 = _make_record("Old", agent="bot", version=1)
        r2 = _make_record("New", agent="bot", version=2, supersedes_id=r1.id)
        viz = ProvenanceVisualizer(records=[r1, r2])
        dot = viz.to_dot()
        assert "superseded by" in dot
        assert r1.id[:8] in dot
        assert r2.id[:8] in dot


class TestProvenanceVisualizerAsync:
    @pytest.mark.asyncio
    async def test_render_async_uses_store(self):
        store = await build_demo_store()
        viz = ProvenanceVisualizer(store=store)
        output = await viz.render_async()
        assert "PROVENANCE TREE" in output

    @pytest.mark.asyncio
    async def test_render_async_no_store_raises(self):
        viz = ProvenanceVisualizer()
        with pytest.raises(RuntimeError):
            await viz.render_async()

"""Tests for RelevanceVisualizer."""

import pytest
from masm.core.agent import Agent
from masm.core.memory import MemoryRecord, MemoryState
from masm.tools.relevance import RelevanceVisualizer
from masm.tools.demo import build_demo_store, build_demo_agents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    content: str,
    agent: str = "agent_a",
    tags: tuple = ("topic",),
    confidence: float = 0.9,
    state: MemoryState = MemoryState.ACTIVE,
) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        author_agent_id=agent,
        tags=tags,
        confidence=confidence,
        state=state,
    )


def _make_agent(agent_id: str, tags: list[str] | None = None) -> Agent:
    return Agent(id=agent_id, tags_of_interest=tags or [])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRelevanceVisualizerRender:
    def test_empty_records_renders_without_crash(self):
        viz = RelevanceVisualizer(records=[], agent=_make_agent("viewer"))
        output = viz.render()
        assert "RELEVANCE RANKING" in output
        assert "no records" in output

    def test_shows_agent_id_in_header(self):
        r = _make_record("Some fact")
        viz = RelevanceVisualizer(records=[r], agent=_make_agent("my_agent"))
        output = viz.render()
        assert "my_agent" in output

    def test_shows_query_in_header(self):
        r = _make_record("Revenue data")
        viz = RelevanceVisualizer(records=[r], query="revenue growth")
        output = viz.render()
        assert "revenue growth" in output

    def test_no_query_shows_placeholder(self):
        r = _make_record("Fact")
        viz = RelevanceVisualizer(records=[r])
        output = viz.render()
        assert "no query" in output

    def test_records_ranked_by_score(self):
        low = _make_record("Unrelated fact", confidence=0.3, tags=("other",))
        high = _make_record("Budget info", confidence=0.9, tags=("budget",))
        agent = _make_agent("analyst", tags=["budget"])
        viz = RelevanceVisualizer(records=[low, high], agent=agent)
        output = viz.render()
        # High-scoring record should appear before low-scoring one
        pos_high = output.find("Budget info")
        pos_low = output.find("Unrelated fact")
        assert pos_high < pos_low

    def test_zero_score_when_no_tag_overlap_and_low_confidence(self):
        r = _make_record("Unrelated", tags=("xyz",), confidence=0.0)
        agent = _make_agent("analyst", tags=["budget"])
        viz = RelevanceVisualizer(records=[r], agent=agent)
        scores = viz.summary()
        assert scores[0]["score"] == pytest.approx(0.0, abs=1e-9)

    def test_limit_respected(self):
        records = [_make_record(f"Fact {i}", agent="bot") for i in range(20)]
        viz = RelevanceVisualizer(records=records, limit=5)
        s = viz.summary()
        assert len(s) == 5

    def test_shows_ascii_bar(self):
        r = _make_record("High confidence fact", confidence=1.0)
        viz = RelevanceVisualizer(records=[r])
        output = viz.render()
        assert "[" in output and "]" in output  # ASCII bar present

    def test_tags_of_interest_shown_in_footer(self):
        r = _make_record("Fact")
        agent = _make_agent("analyst", tags=["budget", "revenue"])
        viz = RelevanceVisualizer(records=[r], agent=agent)
        output = viz.render()
        assert "budget" in output
        assert "revenue" in output


class TestRelevanceVisualizerBreakdown:
    def test_breakdown_no_embedding_shows_na(self):
        r = _make_record("Fact without embedding")
        viz = RelevanceVisualizer(records=[r])
        output = viz.render_breakdown(r.id)
        assert "N/A" in output
        assert "no embedding" in output

    def test_breakdown_shows_composite_score(self):
        r = _make_record("Budget info", tags=("budget",), confidence=0.8)
        agent = _make_agent("analyst", tags=["budget"])
        viz = RelevanceVisualizer(records=[r], agent=agent)
        output = viz.render_breakdown(r.id)
        assert "Composite score" in output

    def test_breakdown_unknown_record_returns_not_found(self):
        viz = RelevanceVisualizer(records=[])
        output = viz.render_breakdown("nonexistent-id")
        assert "not found" in output.lower()

    def test_breakdown_by_id_prefix(self):
        r = _make_record("Fact for prefix test")
        viz = RelevanceVisualizer(records=[r])
        output = viz.render_breakdown(r.id[:8])
        assert "Composite score" in output


class TestRelevanceVisualizerSummary:
    def test_summary_sorted_by_score(self):
        r1 = _make_record("Low", confidence=0.1)
        r2 = _make_record("High", confidence=0.9)
        viz = RelevanceVisualizer(records=[r1, r2])
        s = viz.summary()
        assert s[0]["score"] >= s[1]["score"]

    def test_summary_contains_expected_keys(self):
        r = _make_record("Fact")
        viz = RelevanceVisualizer(records=[r])
        s = viz.summary()
        assert len(s) == 1
        assert "record_id" in s[0]
        assert "content_snippet" in s[0]
        assert "score" in s[0]
        assert "rank" in s[0]
        assert s[0]["rank"] == 1

    def test_summary_empty(self):
        viz = RelevanceVisualizer(records=[])
        assert viz.summary() == []


class TestRelevanceVisualizerPreScored:
    def test_pre_scored_records_used_as_is(self):
        r1 = _make_record("Fact A")
        r2 = _make_record("Fact B")
        scored = [(r1, 0.9), (r2, 0.1)]
        viz = RelevanceVisualizer(scored_records=scored)
        s = viz.summary()
        assert s[0]["record_id"] == r1.id
        assert s[1]["record_id"] == r2.id
        assert s[0]["score"] == pytest.approx(0.9)

    def test_pre_scored_sorted_descending(self):
        r1 = _make_record("Low")
        r2 = _make_record("High")
        scored = [(r1, 0.3), (r2, 0.8)]
        viz = RelevanceVisualizer(scored_records=scored)
        s = viz.summary()
        assert s[0]["score"] > s[1]["score"]


class TestRelevanceVisualizerAsync:
    @pytest.mark.asyncio
    async def test_render_async_uses_active_records_only(self):
        store = await build_demo_store()
        agents = build_demo_agents()
        viz = RelevanceVisualizer(store=store, agent=agents["analyst"])
        output = await viz.render_async()
        assert "RELEVANCE RANKING" in output
        # Forgotten record content should not appear
        assert "123-45-6789" not in output

    @pytest.mark.asyncio
    async def test_render_async_no_store_raises(self):
        viz = RelevanceVisualizer()
        with pytest.raises(RuntimeError):
            await viz.render_async()

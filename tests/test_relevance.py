"""Tests for relevance scoring and filtering."""

import pytest
import numpy as np
from masm.coordination.relevance import RelevanceScorer
from masm.core.agent import Agent
from masm.core.memory import MemoryRecord


class TestRelevanceScorer:
    def test_tag_overlap_boosts_score(self):
        scorer = RelevanceScorer()
        agent = Agent(id="coder", role="developer", tags_of_interest=["code", "bugs"])

        relevant = MemoryRecord(content="bug fix", tags=["code", "bugs"], author_agent_id="a")
        irrelevant = MemoryRecord(content="lunch menu", tags=["misc"], author_agent_id="a")

        score_relevant = scorer.score(relevant, agent)
        score_irrelevant = scorer.score(irrelevant, agent)
        assert score_relevant > score_irrelevant

    def test_high_confidence_boosts_score(self):
        scorer = RelevanceScorer()
        agent = Agent(id="a", tags_of_interest=["data"])

        high_conf = MemoryRecord(content="fact", tags=["data"], confidence=0.99, author_agent_id="b")
        low_conf = MemoryRecord(content="fact", tags=["data"], confidence=0.1, author_agent_id="b")

        assert scorer.score(high_conf, agent) > scorer.score(low_conf, agent)

    def test_filter_respects_threshold(self):
        scorer = RelevanceScorer()
        agent = Agent(id="a", tags_of_interest=["important"])

        records = [
            MemoryRecord(content="relevant", tags=["important"], confidence=0.9, author_agent_id="b"),
            MemoryRecord(content="noise", tags=["misc"], confidence=0.1, author_agent_id="b"),
        ]

        filtered = scorer.filter_relevant(records, agent, threshold=0.2)
        assert len(filtered) >= 1
        # The relevant one should be first
        assert filtered[0][0].content == "relevant"

    def test_filter_respects_limit(self):
        scorer = RelevanceScorer()
        agent = Agent(id="a", tags_of_interest=["data"])

        records = [
            MemoryRecord(content=f"fact {i}", tags=["data"], confidence=0.9, author_agent_id="b")
            for i in range(20)
        ]

        filtered = scorer.filter_relevant(records, agent, limit=5)
        assert len(filtered) == 5

    def test_embedding_similarity_ranking(self):
        scorer = RelevanceScorer()
        agent = Agent(id="a", tags_of_interest=[])

        rng = np.random.RandomState(42)
        query_emb = rng.randn(64).tolist()

        close_emb = (np.array(query_emb) + rng.randn(64) * 0.01).tolist()
        far_emb = rng.randn(64).tolist()

        close_record = MemoryRecord(
            content="close", content_embedding=close_emb, tags=["t"], confidence=0.5, author_agent_id="b"
        )
        far_record = MemoryRecord(
            content="far", content_embedding=far_emb, tags=["t"], confidence=0.5, author_agent_id="b"
        )

        score_close = scorer.score(close_record, agent, query_embedding=query_emb)
        score_far = scorer.score(far_record, agent, query_embedding=query_emb)
        assert score_close > score_far

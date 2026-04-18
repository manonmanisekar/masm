"""RelevanceVisualizer — ASCII visualization of per-agent memory relevance scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from masm.core.agent import Agent
from masm.core.memory import MemoryRecord, MemoryState
from masm.tools._base import ascii_bar, section_header, state_icon, truncate

if TYPE_CHECKING:
    from masm.store.base import SharedMemoryStore


class RelevanceVisualizer:
    """
    Visualizes which memories are most relevant for a given agent or query,
    and why (score component breakdown via ASCII bar charts).

    Usage (sync, with pre-scored data):
        # scored_records is list[tuple[MemoryRecord, float]]
        viz = RelevanceVisualizer(scored_records=scored, agent=agent, query="revenue growth")
        print(viz.render())

    Usage (sync, raw records + agent — stdlib-only scoring, no embeddings):
        records = await store.list_records(states=[MemoryState.ACTIVE])
        viz = RelevanceVisualizer(records=records, agent=agent)
        print(viz.render())

    Usage (async, with store):
        viz = RelevanceVisualizer(store=store, agent=agent, query="revenue growth")
        print(await viz.render_async())
    """

    def __init__(
        self,
        scored_records: Optional[list[tuple[MemoryRecord, float]]] = None,
        records: Optional[list[MemoryRecord]] = None,
        agent: Optional[Agent] = None,
        query: Optional[str] = None,
        store: Optional["SharedMemoryStore"] = None,
        limit: int = 10,
    ) -> None:
        self._scored: Optional[list[tuple[MemoryRecord, float]]] = scored_records
        self._records: list[MemoryRecord] = records or []
        self._agent = agent
        self._query = query
        self._store = store
        self._limit = limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self) -> str:
        """
        Render a ranked list with ASCII score bars.

        If scored_records was provided at construction, those scores are used.
        Otherwise, a lightweight stdlib-only score is computed:
            score = 0.4 * tag_overlap + 0.6 * record.confidence
        """
        scored = self._get_scored()
        scored_limited = scored[: self._limit]

        agent_label = self._agent.id if self._agent else "unknown"
        query_label = f'"{self._query}"' if self._query else "(no query)"
        title = f"RELEVANCE RANKING  agent: {agent_label}  query: {query_label}"
        lines = [section_header(title)]

        if not scored:
            lines.append("\n  (no records to rank)")
        else:
            for rank, (record, score) in enumerate(scored_limited, start=1):
                bar = ascii_bar(score, width=30)
                content = truncate(record.content, 55)
                icon = state_icon(
                    record.state.value if hasattr(record.state, "value") else str(record.state)
                )
                tags = ", ".join(record.tags) if record.tags else "(no tags)"
                lines.append("")
                lines.append(f" #{rank:<3} [{record.id[:8]}]  score: {score:.2f}  {bar}")
                lines.append(f'     "{content}"')
                lines.append(
                    f"     Author: {record.author_agent_id} | Tags: {tags} | "
                    f"conf: {record.confidence:.2f} | {icon} {record.state.value.upper() if hasattr(record.state, 'value') else record.state}"
                )

        lines.append("")
        lines.append("-" * 70)
        active_total = sum(1 for r, _ in scored if r.state == MemoryState.ACTIVE)
        shown = len(scored_limited)
        lines.append(
            f"Showing {shown} of {len(scored)} records (limit: {self._limit})"
        )
        if self._agent and self._agent.tags_of_interest:
            lines.append(f"Agent tags of interest: {', '.join(self._agent.tags_of_interest)}")

        return "\n".join(lines)

    def render_breakdown(self, record_id: str) -> str:
        """
        For a single record, render the per-component score breakdown as
        a vertical ASCII bar chart.

        When no embeddings are available, semantic similarity is shown as N/A.
        """
        scored = self._get_scored()
        target: Optional[tuple[MemoryRecord, float]] = next(
            ((r, s) for r, s in scored if r.id == record_id), None
        )
        if target is None:
            # Try searching by prefix
            target = next(
                ((r, s) for r, s in scored if r.id.startswith(record_id)), None
            )

        if target is None:
            return f"Record [{record_id[:8]}] not found in scored set."

        record, composite = target
        content = truncate(record.content, 50)
        lines = [f'Score breakdown for [{record.id[:8]}] "{content}"', ""]

        # Compute components
        tag_overlap = self._tag_overlap_score(record)
        confidence = record.confidence
        recency = self._recency_score(record)
        has_embedding = record.content_embedding is not None

        bar_w = 30
        lines.append(
            f"  tag_overlap  {ascii_bar(tag_overlap, width=bar_w)}  "
            f"{tag_overlap:.2f}  (weight: 0.40)"
        )
        lines.append(
            f"  confidence   {ascii_bar(confidence, width=bar_w)}  "
            f"{confidence:.2f}  (weight: 0.60)"
        )
        lines.append(
            f"  recency      {ascii_bar(recency, width=bar_w)}  "
            f"{recency:.2f}  (weight: 0.15)"
        )
        sem_bar = ascii_bar(0, width=bar_w, fill="-")
        sem_val = "N/A   (no embedding)" if not has_embedding else "see scorer"
        lines.append(
            f"  semantic_sim {sem_bar}  {sem_val}"
        )
        lines.append("")
        lines.append(f"  Composite score: {composite:.2f}")

        return "\n".join(lines)

    async def render_async(self) -> str:
        """Fetch active records from the store, then render."""
        if self._store is None:
            raise RuntimeError("No store provided.")
        self._records = await self._store.list_records(states=[MemoryState.ACTIVE])
        self._scored = None  # Force recompute from fresh records
        return self.render()

    def summary(self) -> list[dict]:
        """
        Return ranked list as plain dicts sorted by score descending.

        Keys: record_id, content_snippet, score, rank.
        """
        return [
            {
                "rank": i + 1,
                "record_id": r.id,
                "content_snippet": truncate(r.content, 60),
                "score": score,
            }
            for i, (r, score) in enumerate(self._get_scored()[: self._limit])
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_scored(self) -> list[tuple[MemoryRecord, float]]:
        """Return scored records, computing simple scores if not pre-supplied."""
        if self._scored is not None:
            return sorted(self._scored, key=lambda x: x[1], reverse=True)
        return sorted(
            [(r, self._simple_score(r)) for r in self._records],
            key=lambda x: x[1],
            reverse=True,
        )

    def _simple_score(self, record: MemoryRecord) -> float:
        """
        Stdlib-only relevance score (no embeddings, no LLM).

        score = 0.4 * tag_overlap + 0.6 * confidence
        """
        return 0.4 * self._tag_overlap_score(record) + 0.6 * record.confidence

    def _tag_overlap_score(self, record: MemoryRecord) -> float:
        """Tag overlap between agent's interests and record's tags, normalized 0→1."""
        if self._agent is None or not self._agent.tags_of_interest or not record.tags:
            return 0.0
        overlap = len(set(self._agent.tags_of_interest) & set(record.tags))
        return overlap / max(len(self._agent.tags_of_interest), 1)

    def _recency_score(self, record: MemoryRecord) -> float:
        """Exponential recency decay with ~1h half-life. Returns 0→1."""
        import math
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        age_seconds = (now - record.created_at).total_seconds()
        half_life = 3600.0  # 1 hour
        return math.exp(-age_seconds * math.log(2) / half_life)

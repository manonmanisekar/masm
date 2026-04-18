"""ConflictVisualizer — ASCII and DOT visualization of memory conflicts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from masm.core.memory import ConflictEvent, MemoryRecord
from masm.tools._base import ascii_bar, section_header, severity_badge, truncate

if TYPE_CHECKING:
    from masm.store.base import SharedMemoryStore


class ConflictVisualizer:
    """
    Visualizes memory conflicts, their severity, strategy, and resolution outcome.

    Usage (sync, with pre-fetched data):
        conflicts = await store.list_conflicts()
        records = {r.id: r for r in await store.list_records()}
        viz = ConflictVisualizer(conflicts=conflicts, records=records)
        print(viz.render())

    Usage (async, pass store directly):
        viz = ConflictVisualizer(store=store)
        print(await viz.render_async())
    """

    def __init__(
        self,
        conflicts: Optional[list[ConflictEvent]] = None,
        records: Optional[dict[str, MemoryRecord]] = None,
        store: Optional["SharedMemoryStore"] = None,
    ) -> None:
        self._conflicts = conflicts or []
        self._records = records or {}
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self) -> str:
        """
        Render an ASCII conflict graph.

        Raises RuntimeError if called without pre-fetched conflicts/records and
        no store was provided.
        """
        if not self._conflicts and self._store is not None:
            raise RuntimeError(
                "Data not loaded. Call render_async() to fetch from the store, "
                "or pass conflicts/records at construction time."
            )

        total = len(self._conflicts)
        resolved_count = sum(1 for c in self._conflicts if c.resolved)
        unresolved_count = total - resolved_count

        title = f"CONFLICT GRAPH  ({total} total: {unresolved_count} unresolved, {resolved_count} resolved)"
        lines = [section_header(title)]

        if not self._conflicts:
            lines.append("\n  (no conflicts recorded)")
        else:
            for c in self._conflicts:
                lines.append("")
                status = "RESOLVED  " if c.resolved else "UNRESOLVED"
                badge = severity_badge(c.severity.value if hasattr(c.severity, "value") else str(c.severity))
                lines.append(f"{badge} {status}  conflict: {c.id[:8]}")

                a = self._records.get(c.memory_a_id)
                b = self._records.get(c.memory_b_id)
                a_content = truncate(a.content, 45) if a else "?"
                b_content = truncate(b.content, 45) if b else "?"
                a_agent = a.author_agent_id if a else "?"
                b_agent = b.author_agent_id if b else "?"
                a_conf = f"{a.confidence:.2f}" if a else "?"
                b_conf = f"{b.confidence:.2f}" if b else "?"

                lines.append(f'  A: [{c.memory_a_id[:8]}] "{a_content}"')
                lines.append(f"           ({a_agent}, conf={a_conf})")
                lines.append(f'  B: [{c.memory_b_id[:8]}] "{b_content}"')
                lines.append(f"           ({b_agent}, conf={b_conf})")

                strategy = c.strategy.value if hasattr(c.strategy, "value") else str(c.strategy)
                lines.append(f"  Strategy : {strategy}")
                if c.reasoning:
                    lines.append(f"  Reasoning: {truncate(c.reasoning, 60)}")
                if c.resolved_memory_id:
                    lines.append(f"  Winner   : [{c.resolved_memory_id[:8]}]")

        lines.append("")
        lines.append("-" * 70)
        summ = self.summary()
        sev_str = "  ".join(
            f"{k.upper()}:{v}" for k, v in summ["by_severity"].items() if v > 0
        ) or "none"
        strat_str = "  ".join(
            f"{k}:{v}" for k, v in summ["by_strategy"].items() if v > 0
        ) or "none"
        lines.append(
            f"Summary: {total} conflicts | {sev_str} | {strat_str}"
        )

        return "\n".join(lines)

    async def render_async(self) -> str:
        """Fetch data from the store, then render."""
        if self._store is None:
            raise RuntimeError("No store provided.")
        self._conflicts = await self._store.list_conflicts()
        records_list = await self._store.list_records()
        self._records = {r.id: r for r in records_list}
        return self.render()

    def to_dot(self) -> str:
        """
        Return a Graphviz DOT string for the conflict graph.

        No external imports needed — pure string formatting.
        Resolved conflicts use dashed gray edges; unresolved use solid red.
        """
        lines = ["digraph masm_conflicts {"]
        lines.append("  rankdir=LR;")
        lines.append('  node [shape=box, style=filled, fontsize=10];')

        # Collect all record IDs referenced by conflicts
        seen_ids: set[str] = set()
        for c in self._conflicts:
            seen_ids.update([c.memory_a_id, c.memory_b_id])

        state_colors = {
            "active": "#90EE90",
            "superseded": "#D3D3D3",
            "disputed": "#FFD700",
            "retracted": "#FFA07A",
            "forgotten": "#FF6347",
        }

        for rid in seen_ids:
            r = self._records.get(rid)
            if r:
                color = state_colors.get(r.state.value if hasattr(r.state, "value") else str(r.state), "#FFFFFF")
                label = truncate(r.content, 35).replace('"', '\\"')
                agent = r.author_agent_id
                lines.append(f'  "{rid[:8]}" [label="{label}\\n({agent})", fillcolor="{color}"];')
            else:
                lines.append(f'  "{rid[:8]}" [label="unknown"];')

        for c in self._conflicts:
            style = "dashed" if c.resolved else "solid"
            color = "gray" if c.resolved else "red"
            lines.append(
                f'  "{c.memory_a_id[:8]}" -> "{c.memory_b_id[:8]}" '
                f'[label="conflict", style={style}, color={color}, dir=none];'
            )

        lines.append("}")
        return "\n".join(lines)

    def summary(self) -> dict:
        """
        Return summary statistics as a plain dict.

        Keys: total, resolved, unresolved, by_severity, by_strategy.
        """
        by_sev: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        by_strat: dict[str, int] = {}

        for c in self._conflicts:
            sev = c.severity.value if hasattr(c.severity, "value") else str(c.severity)
            by_sev[sev] = by_sev.get(sev, 0) + 1
            strat = c.strategy.value if hasattr(c.strategy, "value") else str(c.strategy)
            by_strat[strat] = by_strat.get(strat, 0) + 1

        resolved = sum(1 for c in self._conflicts if c.resolved)
        return {
            "total": len(self._conflicts),
            "resolved": resolved,
            "unresolved": len(self._conflicts) - resolved,
            "by_severity": by_sev,
            "by_strategy": by_strat,
        }

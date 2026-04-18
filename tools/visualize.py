"""Visualization tools for MASM memory graphs, conflicts, and provenance.

This module is a compatibility shim. The canonical implementations live in
``masm.tools``. Use those directly for the full API:

    from masm.tools import ConflictVisualizer, ProvenanceVisualizer, RelevanceVisualizer

Usage (this shim):
    python -m tools.visualize --demo
    python -m tools.visualize --export dot   # Graphviz DOT format
    python -m tools.visualize --export text  # ASCII visualization
"""

import asyncio
import argparse

from masm.store.in_memory import InMemorySharedStore
from masm.tools import ConflictVisualizer, ProvenanceVisualizer, RelevanceVisualizer
from masm.tools.demo import build_demo_store


class MemoryVisualizer:
    """
    Legacy visualizer class — wraps the new ``masm.tools`` visualizers.

    Preserved for backward compatibility. New code should use
    :class:`masm.tools.ConflictVisualizer`, :class:`masm.tools.ProvenanceVisualizer`,
    and :class:`masm.tools.RelevanceVisualizer` directly.
    """

    def __init__(self, store: InMemorySharedStore) -> None:
        self.store = store
        # Pre-fetch caches (populated lazily on first call via asyncio.run)
        self._conflicts: list = []
        self._records: dict = {}
        self._all_records: list = []
        self._audit_log: list = []

    # ------------------------------------------------------------------
    # Public methods (backward-compatible signatures)
    # ------------------------------------------------------------------

    def render_text(self) -> str:
        """Render an ASCII text visualization of the memory graph."""
        self._ensure_data()
        conflict_viz = ConflictVisualizer(
            conflicts=self._conflicts, records=self._records
        )
        prov_viz = ProvenanceVisualizer(
            records=self._all_records, audit_log=self._audit_log
        )
        return prov_viz.render() + "\n\n" + conflict_viz.render()

    def render_dot(self) -> str:
        """Render a Graphviz DOT format for the memory graph."""
        self._ensure_data()
        conflict_viz = ConflictVisualizer(
            conflicts=self._conflicts, records=self._records
        )
        prov_viz = ProvenanceVisualizer(records=self._all_records)
        # Combine both DOT outputs into a single graph
        conflict_dot = conflict_viz.to_dot()
        prov_dot = prov_viz.to_dot()
        return f"// Conflict graph\n{conflict_dot}\n\n// Provenance graph\n{prov_dot}"

    def render_provenance(self, record_id: str) -> str:
        """Trace and display the provenance chain for a specific memory."""
        self._ensure_data()
        prov_viz = ProvenanceVisualizer(
            records=self._all_records, audit_log=self._audit_log
        )
        return prov_viz.render_chain(record_id)

    def render_agent_view(self, agent_id: str) -> str:
        """Show what a specific agent can see (active records only)."""
        self._ensure_data()
        from masm.core.agent import Agent
        from masm.core.memory import MemoryState

        agent = Agent(id=agent_id)
        active_records = [
            r for r in self._all_records if r.state == MemoryState.ACTIVE
        ]
        rel_viz = RelevanceVisualizer(records=active_records, agent=agent)
        return rel_viz.render()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_data(self) -> None:
        """Fetch store data synchronously if not already loaded."""
        if not self._all_records:
            asyncio.run(self._fetch())

    async def _fetch(self) -> None:
        self._conflicts = await self.store.list_conflicts()
        self._all_records = await self.store.list_records()
        self._records = {r.id: r for r in self._all_records}
        self._audit_log = await self.store.get_audit_log()


async def demo() -> None:
    """Run a demo visualization with sample data."""
    store = await build_demo_store()

    # Fetch data once and share across visualizers
    conflicts = await store.list_conflicts()
    all_records = await store.list_records()
    records_map = {r.id: r for r in all_records}
    audit_log = await store.get_audit_log()

    prov_viz = ProvenanceVisualizer(records=all_records, audit_log=audit_log)
    conflict_viz = ConflictVisualizer(conflicts=conflicts, records=records_map)

    print(prov_viz.render())
    print()
    print(conflict_viz.render())
    print()

    # Show provenance chain for the most recent budget record
    budget_records = [r for r in all_records if "budget" in r.tags]
    if budget_records:
        newest = sorted(budget_records, key=lambda r: r.created_at)[-1]
        print(prov_viz.render_chain(newest.id))
        print()

    from masm.core.agent import Agent
    from masm.core.memory import MemoryState
    active = [r for r in all_records if r.state == MemoryState.ACTIVE]
    rel_viz = RelevanceVisualizer(records=active, agent=Agent(id="analyst"))
    print(rel_viz.render())


def main() -> None:
    parser = argparse.ArgumentParser(description="MASM Memory Visualization")
    parser.add_argument("--demo", action="store_true", help="Run demo visualization")
    parser.add_argument("--export", choices=["text", "dot"], default="text", help="Export format")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(demo())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

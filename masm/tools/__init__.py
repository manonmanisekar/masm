"""
masm.tools — Visualization utilities for MASM memory stores.

All output is pure ASCII (stdlib only). Optional Graphviz DOT export via .to_dot().

Example::

    import asyncio
    from masm.tools import ConflictVisualizer, ProvenanceVisualizer, RelevanceVisualizer
    from masm.tools.demo import build_demo_store, build_demo_agents

    async def main():
        store = await build_demo_store()
        agents = build_demo_agents()

        # Conflict graph
        viz = ConflictVisualizer(store=store)
        print(await viz.render_async())

        # Provenance tree
        prov = ProvenanceVisualizer(store=store)
        print(await prov.render_async())

        # Relevance ranking
        records = await store.list_records()
        rel = RelevanceVisualizer(records=records, agent=agents["analyst"])
        print(rel.render())

    asyncio.run(main())
"""

from masm.tools.benchmark import BenchmarkVisualizer
from masm.tools.conflict import ConflictVisualizer
from masm.tools.provenance import ProvenanceVisualizer
from masm.tools.relevance import RelevanceVisualizer

__all__ = [
    "BenchmarkVisualizer",
    "ConflictVisualizer",
    "ProvenanceVisualizer",
    "RelevanceVisualizer",
]

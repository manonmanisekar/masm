"""ProvenanceVisualizer — ASCII and DOT visualization of memory lineage and audit trails."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Optional

from masm.core.memory import MemoryRecord
from masm.tools._base import format_table, section_header, state_icon, truncate

if TYPE_CHECKING:
    from masm.store.base import SharedMemoryStore


class ProvenanceVisualizer:
    """
    Visualizes memory lineage: version chains, agent attribution, audit trails.

    Usage (sync, with pre-fetched data):
        records = await store.list_records()
        audit_log = await store.get_audit_log()
        viz = ProvenanceVisualizer(records=records, audit_log=audit_log)
        print(viz.render())
        print(viz.render_chain(record_id="abc12345"))

    Usage (async, pass store directly):
        viz = ProvenanceVisualizer(store=store)
        print(await viz.render_async())
    """

    def __init__(
        self,
        records: Optional[list[MemoryRecord]] = None,
        audit_log: Optional[list[dict]] = None,
        store: Optional["SharedMemoryStore"] = None,
    ) -> None:
        self._records: list[MemoryRecord] = records or []
        self._audit_log: list[dict] = audit_log or []
        self._store = store
        # Build id → record index for fast lookup
        self._by_id: dict[str, MemoryRecord] = {r.id: r for r in self._records}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self) -> str:
        """
        Render all version chains as an ASCII tree, grouped by tags.

        Root nodes (no supersedes_id) appear first; version children are indented.
        Footer shows per-agent attribution counts.
        """
        agent_count = len({r.author_agent_id for r in self._records})
        title = f"PROVENANCE TREE  ({len(self._records)} records across {agent_count} agents)"
        lines = [section_header(title)]

        # Group root records by their tag signature
        roots = [r for r in self._records if not r.supersedes_id]
        # Also include records that supersede something but whose root isn't in the store
        all_child_ids = {r.supersedes_id for r in self._records if r.supersedes_id}
        orphaned_children = [
            r for r in self._records
            if r.supersedes_id and r.supersedes_id not in self._by_id
        ]
        roots += orphaned_children

        # Group roots by tags tuple (sorted for stability)
        by_tags: dict[str, list[MemoryRecord]] = defaultdict(list)
        for r in roots:
            key = ", ".join(sorted(r.tags)) if r.tags else "(no tags)"
            by_tags[key].append(r)

        for tag_key, tag_roots in sorted(by_tags.items()):
            lines.append("")
            lines.append(f"[{tag_key}]")
            for root in tag_roots:
                self._render_chain_lines(root, lines, depth=1)

        # Catch records not in any version chain (no tags, no supersedes)
        untagged_roots = [r for r in roots if not r.tags]
        if not roots and not self._records:
            lines.append("\n  (no records in store)")

        lines.append("")
        lines.append("-" * 70)
        lines.append(self._agent_summary_inline())

        return "\n".join(lines)

    def render_chain(self, record_id: str) -> str:
        """
        Render the full ancestry chain for one specific record.

        Walks supersedes_id links backward to the root, then shows the
        forward link (superseded_by_id) if present.
        """
        lines = [f"Provenance chain for memory [{record_id[:8]}]:"]
        lines.append("")

        # Walk backward to root
        chain: list[MemoryRecord] = []
        current_id: Optional[str] = record_id
        visited: set[str] = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            r = self._by_id.get(current_id)
            if r is None:
                lines.append(f"  [{current_id[:8]}]  NOT FOUND IN STORE")
                break
            chain.append(r)
            current_id = r.supersedes_id

        # Display oldest first
        chain.reverse()
        for i, r in enumerate(chain):
            icon = state_icon(r.state.value if hasattr(r.state, "value") else str(r.state))
            prefix = "  " + ("└─ " if i > 0 else "")
            content = truncate(r.content, 50)
            lines.append(
                f"{prefix}{icon} [{r.id[:8]}] v{r.version}  {r.author_agent_id}"
                f'    "{content}"    [{r.state.value.upper() if hasattr(r.state, "value") else r.state}]'
            )

        # Show forward link
        target = self._by_id.get(record_id)
        if target and target.superseded_by_id:
            fwd = self._by_id.get(target.superseded_by_id)
            fwd_content = truncate(fwd.content, 50) if fwd else "?"
            lines.append(
                f'  └─ superseded by [{target.superseded_by_id[:8]}]: "{fwd_content}"'
            )
        elif target and not target.superseded_by_id:
            lines.append("     (no further versions)")

        return "\n".join(lines)

    def render_audit_trail(self, record_id: Optional[str] = None) -> str:
        """
        Render a time-sorted audit trail table.

        If record_id is provided, only show entries for that record.
        """
        entries = self._audit_log
        if record_id:
            entries = [e for e in entries if e.get("record_id") == record_id]

        title = f"AUDIT TRAIL ({len(entries)} entries" + (f" for {record_id[:8]}" if record_id else "") + ")"
        lines = [section_header(title), ""]

        if not entries:
            lines.append("  (no audit entries)")
            return "\n".join(lines)

        headers = ["Timestamp", "Operation", "Agent", "Record"]
        col_widths = [24, 13, 18, 10]
        rows = []
        for e in sorted(entries, key=lambda x: x.get("timestamp", "")):
            ts = str(e.get("timestamp", ""))[:23]
            op = str(e.get("operation", ""))[:13]
            agent = str(e.get("agent_id", ""))[:18]
            rid = str(e.get("record_id") or "\u2014")[:10]
            rows.append([ts, op, agent, rid])

        lines.append(format_table(headers, rows, col_widths))
        return "\n".join(lines)

    async def render_async(self) -> str:
        """Fetch data from the store, then render."""
        if self._store is None:
            raise RuntimeError("No store provided.")
        self._records = await self._store.list_records()
        self._by_id = {r.id: r for r in self._records}
        self._audit_log = await self._store.get_audit_log()
        return self.render()

    def to_dot(self) -> str:
        """
        Return a Graphviz DOT string for the provenance graph.

        Nodes = MemoryRecord (labeled with content snippet, agent, version).
        Edges = supersedes_id (directed: old → new, labeled "superseded by").
        Node shape: box=ACTIVE, ellipse=SUPERSEDED, diamond=DISPUTED.
        """
        state_shapes = {
            "active": "box",
            "superseded": "ellipse",
            "disputed": "diamond",
            "retracted": "parallelogram",
            "forgotten": "octagon",
        }
        state_colors = {
            "active": "#90EE90",
            "superseded": "#D3D3D3",
            "disputed": "#FFD700",
            "retracted": "#FFA07A",
            "forgotten": "#FF6347",
        }

        lines = ["digraph masm_provenance {"]
        lines.append("  rankdir=LR;")
        lines.append("  fontsize=10;")

        for r in self._records:
            sv = r.state.value if hasattr(r.state, "value") else str(r.state)
            shape = state_shapes.get(sv, "box")
            color = state_colors.get(sv, "#FFFFFF")
            label = truncate(r.content, 30).replace('"', '\\"')
            lines.append(
                f'  "{r.id[:8]}" [label="{label}\\nv{r.version} ({r.author_agent_id})", '
                f'shape={shape}, style=filled, fillcolor="{color}"];'
            )

        for r in self._records:
            if r.supersedes_id:
                lines.append(
                    f'  "{r.supersedes_id[:8]}" -> "{r.id[:8]}" [label="superseded by"];'
                )

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_chain_lines(
        self, root: MemoryRecord, lines: list[str], depth: int
    ) -> None:
        """Recursively render a record and its version descendants."""
        indent = "  " * depth
        icon = state_icon(root.state.value if hasattr(root.state, "value") else str(root.state))
        content = truncate(root.content, 50)
        lines.append(
            f"{indent}{icon} [{root.id[:8]}] v{root.version}  {root.author_agent_id:<14}  \"{content}\""
        )
        if root.superseded_by_id:
            lines.append(f"{indent}    \u2514\u2500 superseded by: [{root.superseded_by_id[:8]}]")
        # Find children (records that supersede this one)
        children = [r for r in self._records if r.supersedes_id == root.id]
        for child in children:
            self._render_chain_lines(child, lines, depth + 1)

    def _agent_summary_inline(self) -> str:
        """Return a one-line-per-agent attribution summary."""
        from masm.core.memory import MemoryState
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"written": 0, "active": 0, "superseded": 0})
        for r in self._records:
            agent = r.author_agent_id
            counts[agent]["written"] += 1
            sv = r.state.value if hasattr(r.state, "value") else str(r.state)
            if sv == MemoryState.ACTIVE.value:
                counts[agent]["active"] += 1
            elif sv == MemoryState.SUPERSEDED.value:
                counts[agent]["superseded"] += 1

        lines = ["Agent Attribution:"]
        for agent, c in sorted(counts.items()):
            lines.append(
                f"  {agent:<20}: {c['written']} written  "
                f"({c['active']} active, {c['superseded']} superseded)"
            )
        return "\n".join(lines)

"""Demo data factory for MASM visualization tools.

Used by `masm visualize --demo` and by the test suite to get a pre-populated store.
"""

from __future__ import annotations

from masm.core.agent import Agent
from masm.core.memory import MemoryRecord, MemoryType
from masm.store.in_memory import InMemorySharedStore


async def build_demo_store() -> InMemorySharedStore:
    """
    Populate an InMemorySharedStore with representative demo data:

    - 6 memory records across 4 agents and 3 tag domains
    - 1 conflict (budget disagreement, resolved by LWW)
    - 1 version chain (budget record updated to v2)
    - 1 forgotten record (GDPR compliance demo)

    Returns the populated store.
    """
    store = InMemorySharedStore()

    # --- Budget domain: two agents disagree, then one updates ---
    r1, _ = await store.write(MemoryRecord(
        content="Customer budget is $50,000",
        content_embedding=[1.0, 0.0, 0.0, 0.0],
        author_agent_id="sales_agent",
        tags=("budget", "customer"),
        confidence=0.90,
        memory_type=MemoryType.SEMANTIC,
    ))

    # This conflicts with r1 (same tags, different embedding) — LWW picks r2 as winner
    r2, _ = await store.write(MemoryRecord(
        content="Budget increased to $75,000 after renewal",
        content_embedding=[0.0, 1.0, 0.0, 0.0],
        author_agent_id="account_mgr",
        tags=("budget", "customer"),
        confidence=0.95,
        memory_type=MemoryType.SEMANTIC,
    ))

    # Update r2 to v2 (creates a version chain)
    await store.update(
        record_id=r2.id,
        agent_id="account_mgr",
        updates={"content": "Budget confirmed at $75,000 — signed by VP", "confidence": 0.99},
        reason="Verbal confirmation received",
    )

    # --- Preferences domain ---
    await store.write(MemoryRecord(
        content="Customer prefers email over phone for follow-ups",
        author_agent_id="support_agent",
        tags=("preferences", "customer"),
        confidence=0.80,
        memory_type=MemoryType.EPISODIC,
    ))

    # --- Project domain ---
    r5, _ = await store.write(MemoryRecord(
        content="Project deadline is March 15, 2026",
        author_agent_id="pm_agent",
        tags=("deadline", "project"),
        confidence=0.99,
        memory_type=MemoryType.SEMANTIC,
    ))

    # --- Forgotten record (GDPR demo) ---
    r6, _ = await store.write(MemoryRecord(
        content="Customer SSN: 123-45-6789",
        author_agent_id="sales_agent",
        tags=("pii", "customer"),
        confidence=1.0,
        memory_type=MemoryType.SEMANTIC,
    ))
    await store.forget(
        record_id=r6.id,
        agent_id="compliance_bot",
        reason="gdpr_request — PII removal",
        cascade=True,
    )

    return store


def build_demo_agents() -> dict[str, Agent]:
    """Return the demo agents used in build_demo_store()."""
    return {
        "sales_agent": Agent(
            id="sales_agent",
            role="Manages customer deals and budget tracking",
            authority_rank=1,
            tags_of_interest=["budget", "customer", "deadline"],
        ),
        "account_mgr": Agent(
            id="account_mgr",
            role="Owns customer accounts and renewal negotiations",
            authority_rank=3,
            tags_of_interest=["budget", "customer"],
        ),
        "support_agent": Agent(
            id="support_agent",
            role="Handles customer support interactions",
            authority_rank=1,
            tags_of_interest=["preferences", "customer"],
        ),
        "pm_agent": Agent(
            id="pm_agent",
            role="Tracks project timelines and deliverables",
            authority_rank=2,
            tags_of_interest=["deadline", "project"],
        ),
        "analyst": Agent(
            id="analyst",
            role="Analyzes data across all domains",
            authority_rank=2,
            tags_of_interest=["budget", "deadline", "project"],
        ),
    }

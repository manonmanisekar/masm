"""
Research agent vs Fact-checker — where naive memory silently lies.

Two agents answer the same question:
  "When was 'Attention Is All You Need' published?"

  * ResearchAgent grabs a plausible but wrong date from a stale source.
  * FactChecker verifies against the primary source with higher confidence.

They both write to shared memory under the same topic tags. We run the
same interaction twice: once against a naive dict-backed store (what most
multi-agent tutorials use), once against MASM. The punchline prints at
the bottom.

Run:
    python examples/research_vs_factchecker.py
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

from masm import InMemorySharedStore, MemoryRecord
from masm.cognitive.trust import TrustEngine
from masm.explain.conflict_explainer import ConflictExplainer


# ---------- toy embedding (deterministic, no API key needed) ----------

def embed(text: str, dim: int = 32) -> list[float]:
    """Hash-based pseudo-embedding. Real code would use OpenAI / sentence-transformers."""
    h = hashlib.sha256(text.lower().encode()).digest()
    vec = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


# ---------- the two agents ----------

@dataclass
class Claim:
    content: str
    agent: str
    confidence: float
    tags: tuple[str, ...]


RESEARCH_CLAIM = Claim(
    content="'Attention Is All You Need' was published in 2016.",  # wrong — the arXiv preprint is 2017
    agent="research_agent",
    confidence=0.60,                                                # hedged — pulled from a stale blog
    tags=("paper:attention", "publication_year"),
)

FACTCHECK_CLAIM = Claim(
    content="'Attention Is All You Need' was published in 2017 (arXiv:1706.03762).",
    agent="fact_checker",
    confidence=0.98,                                                # verified against arXiv
    tags=("paper:attention", "publication_year"),
)


# ---------- Baseline: naive dict-backed "shared memory" ----------

class NaiveStore:
    """What a typical multi-agent tutorial does: dict keyed by topic."""
    def __init__(self):
        self.by_topic: dict[tuple[str, ...], str] = {}
        self.audit: list[str] = []

    def write(self, claim: Claim):
        # Last writer wins. No conflict signal. No provenance trail.
        self.by_topic[claim.tags] = claim.content
        self.audit.append(f"{claim.agent} wrote to {claim.tags}")

    def read(self, tags: tuple[str, ...]) -> str | None:
        return self.by_topic.get(tags)


async def run_naive():
    store = NaiveStore()
    # Research agent writes first (as it usually does — it's the "fast" agent).
    store.write(RESEARCH_CLAIM)
    # Fact-checker's verified answer overwrites silently.
    store.write(FACTCHECK_CLAIM)
    # ...but imagine a slower fact-checker, or a retry, or a race where research
    # runs second and clobbers the verified answer:
    store.write(RESEARCH_CLAIM)

    answer = store.read(("paper:attention", "publication_year"))
    return answer, store.audit


# ---------- MASM: conflict-aware shared memory ----------

async def run_masm():
    trust = TrustEngine(prior=0.5)
    explainer = ConflictExplainer()
    store = InMemorySharedStore(trust_engine=trust, conflict_explainer=explainer)

    async def write(claim: Claim):
        return await store.write(MemoryRecord(
            content=claim.content,
            content_embedding=embed(claim.content),
            author_agent_id=claim.agent,
            tags=list(claim.tags),
            confidence=claim.confidence,
        ))

    await write(RESEARCH_CLAIM)
    await write(FACTCHECK_CLAIM)
    # Same "research clobbers fact-check" race as the naive case:
    _, late_conflicts = await write(RESEARCH_CLAIM)

    active = [r for r in await store.read(agent_id="reader", tags=["paper:attention"])
              if r.state.value == "active"]
    return active, late_conflicts, store, trust


# ---------- pretty output ----------

def banner(title: str):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


async def main():
    banner("NAIVE dict-backed shared memory")
    naive_answer, naive_audit = await run_naive()
    print(f"  Final answer stored:  {naive_answer}")
    print(f"  Conflicts surfaced:   0  (the store has no concept of one)")
    print(f"  Audit trail length:   {len(naive_audit)}  (who said what is lost)")
    print("  → A downstream agent reading this memory is told the paper is from 2016.")

    banner("MASM — conflict-aware shared memory")
    active, late_conflicts, store, trust = await run_masm()
    winner = max(active, key=lambda r: r.confidence) if active else None
    print(f"  Final answer stored:  {winner.content if winner else '(none)'}")
    print(f"  Conflicts surfaced:   {len(store._conflicts)}")
    print(f"  Trust scores:         "
          f"fact_checker={trust.score('fact_checker'):.2f}  "
          f"research_agent={trust.score('research_agent'):.2f}")
    if store._explanations:
        exp = store._explanations[-1]
        print(f"  Why fact-checker won: {exp.summary}")
    print("  → The verified 2017 answer survives even when research writes last.")

    banner("The diff")
    print("  Naive:  silently returns the wrong year, no way to detect it.")
    print("  MASM:   detects the conflict, resolves by confidence+trust,")
    print("          keeps full audit trail and a structured explanation.")
    print()


if __name__ == "__main__":
    asyncio.run(main())

# MASM — Multi-Agent Shared Memory

**The first benchmark and framework for multi-agent memory coordination.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Python](https://img.shields.io/badge/python-3.11+-green)]()
[![Version](https://img.shields.io/badge/version-0.2.0-orange)]()

---

## The Problem

AI memory tools (Mem0, Zep, Letta) solve single-agent memory. But production AI deploys 5-20 agents working together.

**No one solves:**
- Memory duplication across agents
- Conflicting facts between agents
- Stale reads from propagation delays
- Memory provenance and attribution
- GDPR-compliant coordinated forgetting
- Efficient per-agent relevance filtering
- Adversarial/unreliable agent isolation

MASM solves these problems and provides the first standardized benchmark to measure multi-agent memory quality.

## See It In 90 Seconds

Two agents answer *"When was 'Attention Is All You Need' published?"*. A research
agent grabs a wrong date (2016) from a stale blog; a fact-checker verifies the
correct answer (2017) against arXiv. Both write to shared memory under the same
tags. Run the same interaction against a naive dict and against MASM:

```bash
python examples/research_vs_factchecker.py
```

```text
================================================================
  NAIVE dict-backed shared memory
================================================================
  Final answer stored:  'Attention Is All You Need' was published in 2016.
  Conflicts surfaced:   0  (the store has no concept of one)
  → A downstream agent reading this memory is told the paper is from 2016.

================================================================
  MASM — conflict-aware shared memory
================================================================
  Final answer stored:  'Attention Is All You Need' was published in 2017 (arXiv:1706.03762).
  Conflicts surfaced:   2
  Trust scores:         fact_checker=0.60  research_agent=0.30
  Why fact-checker won: Memory A won via authority_rank:
                        stronger confidence (Author-reported confidence on each record).
  → The verified 2017 answer survives even when research writes last.
```

The naive store silently ships the wrong answer downstream. MASM detects the
conflict, resolves it by confidence and dynamic trust, keeps a full audit
trail, and returns a structured explanation. Same two agents, same sequence
of writes — one store just lies quietly.

### Scaled Up: 13-agent supply chain

Two agents is the tutorial case. The real test is what happens when 10+
specialised agents each own a slice of the truth and some of them disagree.
A supply-chain "war room" with 13 agents — suppliers, warehouses, logistics,
customs, demand, procurement, QC, finance, compliance, and a planner — asks
one question: *"Can we hit the Q2 delivery commitment for SKU-A47?"*

```bash
python examples/supply_chain_war_room.py
```

```text
========================================================================
  NAIVE dict-backed shared memory
========================================================================
  Conflicts surfaced:     0
  Lead time planner saw:  SKU-A47 lead time remains 14 days (re-confirmed).
  Shipment planner saw:   Container MSCU-7742 held at Rotterdam customs...
  QC status planner saw:  SKU-A47 batch B-2026-Q2-07 on hold (defect 2.1%).
  → Decision:             GO     (wrong)

========================================================================
  MASM — conflict-aware shared memory
========================================================================
  Active records:         8     (forgotten: 1)
  Conflicts surfaced:     5
  Lead time planner saw:  SKU-A47 raw material ships in 7 days via air.
                          (via supplier_orion)
  Trust scores:           supplier_apex=0.30   supplier_orion=0.60
  → Decision:             NO-GO  (correct)
```

Same 13 agents, same writes, same stale-supplier race. The naive store
returns **GO** — which would ship a QC-held batch through a customs-held
container, because the stale supplier re-read clobbered the verified 7-day
lead time and no contradictions were ever surfaced. MASM returns **NO-GO**,
with 5 conflicts surfaced and each one explained, the correct lead time
preserved against the race, and the PII contact cleanly forgotten. The
composite failure mode of a naive shared dict isn't subtle at this scale.

## Quick Start

```bash
pip install -e .
```

```python
import asyncio
from masm import InMemorySharedStore, MemoryRecord

async def main():
    store = InMemorySharedStore()

    # Agent writes a memory
    record, conflicts = await store.write(MemoryRecord(
        content="Revenue grew 23% YoY in Q3 2025",
        author_agent_id="researcher",
        tags=["revenue", "q3"],
        confidence=0.95,
    ))

    # Another agent reads relevant context
    context = await store.read(agent_id="analyst", tags=["revenue"])

    # Conflicts are detected automatically
    record2, conflicts = await store.write(MemoryRecord(
        content="Revenue grew 21% YoY in Q3 2025",
        author_agent_id="analyst",
        tags=["revenue", "q3"],
        confidence=0.85,
    ))
    print(f"Conflicts: {len(conflicts)}")

asyncio.run(main())
```

## Run the Benchmark

```bash
# Full 10-scenario suite
python -m masm benchmark

# Specific scenarios
python -m masm benchmark --scenarios s01_dedup,s02_conflict

# Compare MASM vs Naive baseline
python -m masm compare

# JSON output for CI
python -m masm benchmark --format json --output results.json

# Performance metrics
python -m benchmark.perf_metrics
```

## Benchmark Scenarios (v0.2.0)

| # | Scenario | What It Tests | Weight |
|---|----------|---------------|--------|
| S01 | Deduplication | Semantic duplicate detection across concurrent writes | 1.0 |
| S02 | Conflict Resolution | Detecting and resolving contradictory agent memories | 1.5 |
| S03 | Stale Reads | Ensuring agents see latest data after updates | 1.0 |
| S04 | Attribution | Provenance tracking through multi-agent chains | 1.0 |
| S05 | Scalability | Performance from 2 to 50 agents | 1.2 |
| S06 | Relevance | Per-agent filtering by role and context | 1.0 |
| S07 | Forgetting | GDPR-compliant cascading deletion | 1.3 |
| S08 | Handoff | Context fidelity in agent-to-agent transfers | 1.2 |
| S09 | Adversarial | Isolation of unreliable agent data | 1.0 |
| S10 | E2E Workflow | Complete multi-agent task quality | 1.5 |

### MASM vs Naive Shared Dict

Run `python -m masm compare` to reproduce. Both backends run the same 10
scenarios with the same 5 agents; the only variable is the store.

| Scenario        | MASM      | Naive dict | Δ         |
|-----------------|----------:|-----------:|----------:|
| **Composite**   | **85.04%**| **62.94%** | **+22.10**|
| s01_dedup       |   100.00% |     20.00% |    +80.00 |
| s02_conflict    |    86.67% |     20.00% |    +66.67 |
| s03_staleness   |   100.00% |    100.00% |      0.00 |
| s04_attribution |   100.00% |    100.00% |      0.00 |
| s05_scalability |    17.82% |     20.54% |     −2.72 |
| s06_relevance   |    90.00% |     90.00% |      0.00 |
| s07_forgetting  |   100.00% |     20.00% |    +80.00 |
| s08_handoff     |    70.91% |     92.73% |    −21.82 |
| s09_adversarial |   100.00% |    100.00% |      0.00 |
| s10_e2e         |    92.33% |     89.67% |     +2.66 |

Read these honestly: MASM dominates where coordination matters (dedup,
conflict, forgetting — the scenarios where a naive store *silently loses
data*). It's roughly a wash on single-writer scenarios. It loses on
scalability (S05 — the in-memory backend has a known throughput ceiling)
and on handoff (S08 — the naive store's flat dict makes serialization
trivial). Those are the two open work items.

## Architecture

```
masm/
├── core/           # MemoryRecord, Agent, VectorClock, ConflictDetection
├── store/          # Abstract interface + InMemory backend
├── coordination/   # Merge strategies, relevance scoring, protocols
├── semantic/       # Embedding-based dedup, LLM conflict resolution
└── integrations/   # LangChain, CrewAI, AutoGen, Swarm adapters

benchmark/
├── scenarios/      # S01-S10 benchmark scenarios
├── datasets/       # Ground truth data (4 domains)
├── baselines/      # Naive dict, Mem0 adapter
└── perf_metrics.py # Throughput/latency benchmarking

tools/
└── visualize.py    # Memory graph visualization (ASCII + Graphviz)
```

## Key Design Decisions

- **Vector clocks** for causal ordering (not wall clocks)
- **Immutable records** — updates create new versions, never mutate
- **Conflicts are first-class** — detected, tracked, and resolved explicitly
- **Async throughout** — multi-agent systems are inherently concurrent
- **Zero required ML deps** — semantic features are optional extras

## Running Tests

```bash
pip install -e ".[dev]"
pytest                          # All tests
pytest tests/test_stress.py     # Stress tests
pytest tests/test_integration.py # Integration tests
```

## Visualization

```bash
python -m tools.visualize --demo
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Priority areas:
- Storage backends (Redis, PostgreSQL)
- Framework integrations
- New benchmark scenarios
- Ground truth datasets

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan.

## License

Apache 2.0

# Changelog

All notable changes to MASM are documented in this file.

## [0.2.0] - 2026-04-15

### Added
- **Full Benchmark Suite**: All 10 scenarios (S01-S10) now implemented
  - S03: Stale read detection
  - S04: Attribution and provenance tracking
  - S05: N-agent scalability
  - S06: Per-agent relevance filtering
  - S07: Coordinated forgetting (GDPR compliance)
  - S09: Adversarial agent input isolation
  - S10: End-to-end multi-agent workflows
- **Baseline Comparisons**: Naive shared dict baseline and Mem0 adapter
- **Performance Metrics**: Dedicated throughput/latency benchmarking script
- **Comparison Mode**: `masm compare` CLI command for side-by-side results
- **Visualization Tools**: ASCII and Graphviz DOT memory graph rendering
- **Ground Truth Datasets**: customer_support, research_team, code_review, financial_analysis
- **Framework Integration Stubs**: LangChain, CrewAI, AutoGen, OpenAI Swarm adapters
- **Expanded Documentation**:
  - Conflict resolution guide with examples
  - GDPR-compliant forgetting guide
  - Architecture overview
  - Framework integration guide
  - Extension guide for custom backends
- **Stress Tests**: High-volume writes, concurrent operations, cascade chains
- **Integration Tests**: Full lifecycle, handoff, subscription, all-scenario validation
- **Community Files**: CONTRIBUTING.md, issue templates, ROADMAP.md

### Changed
- Benchmark runner now loads all 10 scenarios by default
- CLI updated to v0.2.0 with `compare` subcommand
- VectorClock gains `reset()` and `equals()` methods

## [0.1.0] - 2026-04-14

### Added
- Core data model: MemoryRecord, MemoryType, MemoryState, ConflictStrategy
- Agent identity with read/write permissions
- VectorClock for causal ordering
- Conflict detection and resolution (LWW, Authority Rank, Manual)
- InMemorySharedStore with full SharedMemoryStore interface
- Semantic deduplication (embedding cosine similarity)
- LLM-based conflict resolution stub (with heuristic fallback)
- Relevance scoring (tag + semantic + recency + confidence)
- Memory merge strategies
- Pub/sub memory change notifications
- GDPR-compliant forgetting with cascade
- Full audit trail
- Benchmark framework with runner and reporter
- 3 benchmark scenarios: S01 (Dedup), S02 (Conflict), S08 (Handoff)
- Quickstart example
- CLI interface (`python -m masm benchmark`)
- 33 unit tests
- README with quick start guide

# MASM Roadmap

## v0.1.0 — Foundation (Released)
- [x] Core data model (MemoryRecord, Agent, VectorClock)
- [x] InMemorySharedStore with full SharedMemoryStore interface
- [x] Conflict detection and resolution (LWW, Authority, Manual)
- [x] Semantic deduplication
- [x] 3 benchmark scenarios (S01, S02, S08)
- [x] Quickstart example
- [x] Test suite

## v0.2.0 — Full Benchmark Suite (Current)
- [x] All 10 benchmark scenarios (S01-S10)
- [x] Naive baseline implementation
- [x] Mem0 adapter (baseline comparison)
- [x] Ground truth datasets (4 domains)
- [x] Performance metrics tooling
- [x] Comparison benchmarking (MASM vs Naive)
- [x] Expanded test coverage (unit, integration, stress)
- [x] Documentation: conflict resolution, GDPR, architecture
- [x] CONTRIBUTING.md and issue templates
- [x] Visualization tools

## v0.3.0 — Production Features (Planned)
- [ ] Redis-backed SharedMemoryStore
- [ ] SQLite-backed SharedMemoryStore
- [ ] Pub/sub memory change notifications (real async)
- [ ] GDPR governance module with compliance reporting
- [ ] LLM-based semantic conflict resolution (OpenAI integration)
- [ ] LangChain adapter (full implementation)
- [ ] CrewAI adapter (full implementation)
- [ ] Memory compression/summarization
- [ ] GitHub Actions CI pipeline

## v0.4.0 — Ecosystem (Planned)
- [ ] AutoGen adapter
- [ ] OpenAI Swarm adapter
- [ ] PostgreSQL backend
- [ ] DynamoDB backend
- [ ] Web dashboard for memory visualization
- [ ] REST API server (`masm serve`)
- [ ] Docker compose setup
- [ ] Published benchmark results on website

## v1.0.0 — Stable Release (Future)
- [ ] API stability guarantee
- [ ] Comprehensive documentation site
- [ ] Published arXiv paper
- [ ] PyPI package with semantic versioning
- [ ] Community-contributed benchmark scenarios
- [ ] Leaderboard website

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved. Priority areas are tagged in the issue tracker.

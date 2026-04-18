# Contributing to MASM

Thank you for your interest in contributing to MASM! This project aims to become the definitive benchmark and framework for multi-agent shared memory.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/masm.git
cd masm

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run benchmarks
python -m masm benchmark

# Run performance metrics
python -m benchmark.perf_metrics
```

## What We Need Help With

### High Priority
- **Storage backends**: Redis, PostgreSQL, SQLite implementations
- **Framework integrations**: Complete LangChain, CrewAI, AutoGen adapters
- **Ground truth datasets**: More labeled scenarios for benchmarks

### Medium Priority
- **Benchmark scenarios**: New scenarios beyond the core 10
- **Semantic features**: Better embedding-based dedup and conflict resolution
- **Performance optimization**: Faster in-memory operations, better indexing

### Good First Issues
- Add type hints to any module missing them
- Improve docstrings and inline documentation
- Add more unit tests for edge cases
- Fix any `TODO` comments in the code

## Development Guidelines

### Code Style
- Python 3.11+ with type hints
- Format with `ruff` (configuration in `pyproject.toml`)
- Docstrings on every public method
- `async/await` for all store operations

### Testing
- Write tests for every new feature
- Test at three levels: unit, integration, benchmark validation
- Use `pytest-asyncio` for async tests
- Run `pytest` before submitting PRs

### Architecture Rules
- **Never mutate MemoryRecord in place** — always create new versions
- **Store operations must be thread-safe** — use asyncio locks
- **Core must be dependency-free** — no OpenAI/ML requirements in `masm/core/`
- **Benchmarks must be reproducible** — seed all randomness

### Git Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests first, then implementation
4. Run `pytest` and `ruff check .`
5. Submit a PR with a clear description

### PR Checklist
- [ ] Tests pass (`pytest`)
- [ ] Linting passes (`ruff check .`)
- [ ] New public APIs have docstrings
- [ ] Benchmark scenarios still run (`python -m masm benchmark`)
- [ ] Updated relevant docs if needed

## Adding a New Benchmark Scenario

1. Create `benchmark/scenarios/s_XX_name.py`
2. Subclass `BenchmarkScenario` from `scenario_base.py`
3. Implement `name`, `description`, `setup()`, `run()`, `teardown()`
4. `run()` must return a `ScenarioResult` with score 0.0-1.0
5. Add a test in `tests/test_benchmark.py`
6. Register in `benchmark/runner.py`

## Adding a Storage Backend

1. Create `masm/store/my_backend.py`
2. Implement all methods from `SharedMemoryStore` (see `base.py`)
3. Add integration tests
4. Run the full benchmark suite against your backend

## Questions?

Open an issue on GitHub with the `question` label.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

"""Base class for all benchmark scenarios."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ScenarioResult:
    """Standardized result from a benchmark scenario."""

    scenario_name: str
    score: float  # 0.0 to 1.0 (higher is better)
    metrics: dict[str, Any] = field(default_factory=dict)
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    num_agents: int = 0
    num_operations: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None


class BenchmarkScenario(ABC):
    """Base class for all benchmark scenarios."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable scenario name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """What this scenario tests."""
        ...

    @abstractmethod
    async def setup(self, store, num_agents: int = 5, **kwargs):
        """Initialize scenario state."""
        ...

    @abstractmethod
    async def run(self) -> ScenarioResult:
        """Execute the scenario and return results."""
        ...

    @abstractmethod
    async def teardown(self):
        """Clean up."""
        ...

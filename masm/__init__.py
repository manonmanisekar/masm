"""MASM — Multi-Agent Shared Memory: Benchmark and framework for AI agent memory coordination."""

from masm.core.memory import MemoryRecord, MemoryType, MemoryState, ConflictStrategy
from masm.core.agent import Agent
from masm.core.clock import VectorClock
from masm.store.base import SharedMemoryStore
from masm.store.in_memory import InMemorySharedStore
from masm.tools import ConflictVisualizer, ProvenanceVisualizer, RelevanceVisualizer

__version__ = "0.2.0"

__all__ = [
    "MemoryRecord",
    "MemoryType",
    "MemoryState",
    "ConflictStrategy",
    "Agent",
    "VectorClock",
    "SharedMemoryStore",
    "InMemorySharedStore",
    "ConflictVisualizer",
    "ProvenanceVisualizer",
    "RelevanceVisualizer",
]

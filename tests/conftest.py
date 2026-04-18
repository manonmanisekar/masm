"""Shared test fixtures for MASM tests."""

import pytest
from masm.store.in_memory import InMemorySharedStore
from masm.core.agent import Agent


@pytest.fixture
def store():
    """Fresh in-memory store for each test."""
    return InMemorySharedStore()


@pytest.fixture
def agents():
    """Standard set of test agents."""
    return {
        "alice": Agent(id="alice", role="researcher", tags_of_interest=["research", "data"]),
        "bob": Agent(id="bob", role="analyst", tags_of_interest=["analysis", "data"]),
        "carol": Agent(id="carol", role="writer", tags_of_interest=["writing", "report"]),
    }

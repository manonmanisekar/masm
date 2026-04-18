"""Agent identity, roles, and permissions for multi-agent memory systems."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from masm.core.memory import MemoryRecord

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """
    Represents an AI agent participating in shared memory.

    Each agent has an identity, a role description (used for relevance filtering),
    and an optional authority rank (used in authority-based conflict resolution).
    """

    id: str
    role: str = ""
    authority_rank: int = 0  # Higher = more authoritative in conflict resolution
    tags_of_interest: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def can_read(self, record: "MemoryRecord") -> bool:
        """Check if this agent has read access to a memory record.

        Currently all agents can read all records. Future versions may add
        per-record read_agents restrictions.
        """
        return True

    def can_write(self, record: "MemoryRecord") -> bool:
        """Check if this agent has write access to a memory record.

        Only the original author can write (update) a record by default.
        """
        return record.author_agent_id == self.id

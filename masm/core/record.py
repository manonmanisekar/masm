"""Record class for multi-agent memory systems."""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Record:
    """
    Represents a memory record in the system.

    Attributes:
        id: Unique identifier for the record.
        read_agents: List of agent IDs that can read this record.
        write_agents: List of agent IDs that can write to this record.
        author_agent_id: ID of the agent that created this record.
    """
    id: str
    read_agents: Optional[List[str]] = field(default_factory=list)
    write_agents: Optional[List[str]] = field(default_factory=list)
    author_agent_id: Optional[str] = None
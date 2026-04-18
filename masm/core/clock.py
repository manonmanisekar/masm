""""Vector clock implementation for causal ordering across agents."""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class VectorClock:
    """
    Tracks causal ordering across multiple agents.

    Each agent maintains a counter. On write, increment own counter.
    On read, merge with sender's clock (take max per agent).
    """

    def __init__(self):
        self._clock: Dict[str, int] = {}

    @property
    def clock(self) -> Dict[str, int]:
        return self._clock.copy()

    def increment(self, agent_id: str) -> Dict[str, int]:
        """Increment this agent's logical timestamp and return the new clock."""
        self._clock[agent_id] = self._clock.get(agent_id, 0) + 1
        logger.info(f"Incremented clock for agent '{agent_id}': {self._clock}")
        return self._clock.copy()

    def merge(self, other: Dict[str, int]) -> Dict[str, int]:
        """Merge another clock into this one (element-wise max)."""
        logger.info(f"Merging current clock {self._clock} with {other}")
        for agent_id, counter in other.items():
            self._clock[agent_id] = max(self._clock.get(agent_id, 0), counter)
        logger.info(f"Resulting clock after merge: {self._clock}")
        return self._clock.copy()

    def happens_before(self, a: Dict[str, int], b: Dict[str, int]) -> bool:
        """Returns True if clock A causally precedes clock B."""
        all_keys = set(a) | set(b)
        return all(a.get(k, 0) <= b.get(k, 0) for k in all_keys) and any(
            a.get(k, 0) < b.get(k, 0) for k in all_keys
        )

    def concurrent(self, a: dict[str, int], b: dict[str, int]) -> bool:
        """
        Returns True if neither clock precedes the other (potential conflict).

        Two clocks are concurrent if:
        - A does not causally precede B.
        - B does not causally precede A.
        - A and B are not identical.
        """
        if a == b:
            return False  # A clock cannot be concurrent with itself
        return not (self.happens_before(a, b) or self.happens_before(b, a))

    def reset(self) -> None:
        """Reset the vector clock to its initial state."""
        self._clock.clear()
        logger.info("Vector clock has been reset.")

    def equals(self, a: Dict[str, int], b: Dict[str, int]) -> bool:
        """Check if two clocks are equal."""
        return a == b
    
    def difference(self, a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        """
        Calculate the difference between two clocks.

        Args:
            a: The first vector clock.
            b: The second vector clock.

        Returns:
            A dictionary showing the difference for each agent.
        """
        all_keys = set(a) | set(b)
        return {k: a.get(k, 0) - b.get(k, 0) for k in all_keys}

    def max_clock(self) -> Optional[str]:
        """
        Find the agent with the highest logical timestamp.

        Returns:
            The agent ID with the highest timestamp, or None if the clock is empty.
        """
        if not self._clock:
            return None
        return max(self._clock, key=self._clock.get)
    
    def from_dict(self, clock_dict: Dict[str, int]) -> None:
        """Initialize the vector clock from an existing dictionary. """
        self._clock = clock_dict.copy()
        logger.info(f"Vector clock initialized from dictionary: {self._clock}")
        
    def to_dict(self) -> Dict[str, int]:
        """Convert the vector clock to a dictionary."""
        return self._clock.copy()   

    def compare(self, a: Dict[str, int], b: Dict[str, int]) -> str:
        """
        Compare two vector clocks.

        Args:
            a: The first vector clock.
            b: The second vector clock.

        Returns:
            A string indicating the relationship:
            - "happens-before": If A causally precedes B.
            - "happens-after": If B causally precedes A.
            - "concurrent": If A and B are concurrent.
            - "equal": If A and B are identical.
        """
        if self.equals(a, b):
            return "equal"
        elif self.happens_before(a, b):
            return "happens-before"
        elif self.happens_before(b, a):
            return "happens-after"
        else:
            return "concurrent"
"""Dynamic Trust Engine — per-agent authority computed from observed behavior.

The existing `ConflictResolver` consults a static `authority_rank` dict. The
trust engine replaces that with a rolling score driven by the edge graph:

* Winning a CONTRADICTS edge (agent's record was kept)            → +w_win
* Losing a CONTRADICTS edge (agent's record was superseded)       → −w_loss
* Receiving a SUPPORTS edge from another agent                    → +w_support
* Authoring a FORGOTTEN / RETRACTED record                        → −w_retract

Scores are clamped to [0, 1]. Start-of-life score is `prior` (default 0.5).
Callers feed in events via `record_event`; the engine is pure bookkeeping
and does not touch the store directly, so it's trivial to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TrustEvent = Literal[
    "conflict_win",
    "conflict_loss",
    "supported",
    "retracted",
    "forgotten",
]


@dataclass
class TrustEngine:
    prior: float = 0.5
    w_win: float = 0.05
    w_loss: float = 0.10
    w_support: float = 0.03
    w_retract: float = 0.15

    _scores: dict[str, float] = field(default_factory=dict)
    _events: dict[str, list[TrustEvent]] = field(default_factory=dict)

    def score(self, agent_id: str) -> float:
        """Return the current trust score for `agent_id` (defaulting to prior)."""
        return self._scores.get(agent_id, self.prior)

    def record_event(self, agent_id: str, event: TrustEvent) -> float:
        """Apply an event for `agent_id` and return the resulting score."""
        delta = {
            "conflict_win": self.w_win,
            "conflict_loss": -self.w_loss,
            "supported": self.w_support,
            "retracted": -self.w_retract,
            "forgotten": -self.w_retract,
        }[event]
        new = max(0.0, min(1.0, self.score(agent_id) + delta))
        self._scores[agent_id] = new
        self._events.setdefault(agent_id, []).append(event)
        return new

    def ranking(self, agents: list[str]) -> list[tuple[str, float]]:
        """Return agents sorted by trust desc. Ties broken by id for stability."""
        return sorted(
            ((a, self.score(a)) for a in agents),
            key=lambda x: (-x[1], x[0]),
        )

    def as_authority_dict(self) -> dict[str, dict]:
        """Shape compatible with `ConflictResolver._resolve_authority(agents=...)`."""
        return {aid: {"authority_rank": s} for aid, s in self._scores.items()}

    def history(self, agent_id: str) -> list[TrustEvent]:
        return list(self._events.get(agent_id, []))

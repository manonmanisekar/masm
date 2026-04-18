"""Abstract vector-index interface — all backends implement this."""

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence


class VectorIndex(ABC):
    """Cosine-similarity vector index keyed by record id.

    Implementations may be exact (brute force) or approximate (HNSW, IVF, …).
    All methods are synchronous — concurrency is the caller's responsibility.
    """

    dim: int

    @abstractmethod
    def add(self, record_id: str, embedding: Sequence[float]) -> None:
        """Insert or overwrite the embedding for `record_id`."""

    @abstractmethod
    def remove(self, record_id: str) -> None:
        """Best-effort removal. No-op if `record_id` is unknown."""

    @abstractmethod
    def search(
        self,
        embedding: Sequence[float],
        k: int = 10,
        threshold: float = 0.0,
        exclude_ids: Optional[Iterable[str]] = None,
    ) -> list[tuple[str, float]]:
        """Return up to `k` `(record_id, cosine_similarity)` pairs, sorted desc.

        Only returns candidates with similarity >= `threshold`. `exclude_ids`
        are filtered after the search (useful for "find duplicates of X but
        not X itself").
        """

    @abstractmethod
    def __len__(self) -> int: ...

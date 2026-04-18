"""Exact cosine-similarity index via numpy. Default backend, no extra deps."""

from typing import Iterable, Optional, Sequence

import numpy as np

from masm.index.base import VectorIndex


class BruteForceIndex(VectorIndex):
    """Linear-scan cosine index.

    Stores unit-normalized embeddings in a dense numpy matrix so `search` is
    a single matmul. Good up to ~1e5 records; beyond that use `HNSWIndex`.
    """

    def __init__(self, dim: int = 0):
        self.dim = dim
        self._ids: list[str] = []
        self._id_to_row: dict[str, int] = {}
        # Lazily allocated so callers don't have to know `dim` up front.
        self._matrix: Optional[np.ndarray] = None

    def _ensure_matrix(self, dim: int) -> None:
        if self._matrix is None:
            self.dim = dim
            self._matrix = np.zeros((0, dim), dtype=np.float32)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec
        return vec / norm

    def add(self, record_id: str, embedding: Sequence[float]) -> None:
        v = np.asarray(embedding, dtype=np.float32)
        self._ensure_matrix(v.shape[0])
        if v.shape[0] != self.dim:
            raise ValueError(
                f"Embedding dim {v.shape[0]} does not match index dim {self.dim}"
            )
        v = self._normalize(v)

        if record_id in self._id_to_row:
            row = self._id_to_row[record_id]
            self._matrix[row] = v
            return

        assert self._matrix is not None
        self._id_to_row[record_id] = len(self._ids)
        self._ids.append(record_id)
        self._matrix = np.vstack([self._matrix, v[None, :]])

    def remove(self, record_id: str) -> None:
        row = self._id_to_row.pop(record_id, None)
        if row is None or self._matrix is None:
            return
        self._ids.pop(row)
        self._matrix = np.delete(self._matrix, row, axis=0)
        # Re-index rows for everyone shifted up.
        for rid, r in self._id_to_row.items():
            if r > row:
                self._id_to_row[rid] = r - 1

    def search(
        self,
        embedding: Sequence[float],
        k: int = 10,
        threshold: float = 0.0,
        exclude_ids: Optional[Iterable[str]] = None,
    ) -> list[tuple[str, float]]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        q = self._normalize(np.asarray(embedding, dtype=np.float32))
        if q.shape[0] != self.dim:
            return []
        sims = self._matrix @ q  # cosine since rows are unit-normalized
        excl = set(exclude_ids or ())
        scored = [
            (self._ids[i], float(sims[i]))
            for i in range(len(self._ids))
            if self._ids[i] not in excl and sims[i] >= threshold
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self._ids)

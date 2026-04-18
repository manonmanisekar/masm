"""Approximate nearest-neighbor index via hnswlib.

Optional — only importable when `hnswlib` is installed (`pip install hnswlib`).
"""

from typing import Iterable, Optional, Sequence

import numpy as np

try:
    import hnswlib
except ImportError as exc:  # pragma: no cover - optional dep
    raise ImportError(
        "HNSWIndex requires hnswlib. Install with: pip install hnswlib"
    ) from exc

from masm.index.base import VectorIndex


class HNSWIndex(VectorIndex):
    """HNSW cosine-space index backed by hnswlib.

    Use when the record count grows past ~1e5 or when dedup/read latency
    starts to dominate benchmark S08 (scalability). Returns *approximate*
    nearest neighbors — exact recall is not guaranteed.
    """

    def __init__(
        self,
        dim: int,
        max_elements: int = 10_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ):
        self.dim = dim
        self._max_elements = max_elements
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=max_elements, ef_construction=ef_construction, M=M
        )
        self._index.set_ef(ef_search)
        self._id_to_label: dict[str, int] = {}
        self._label_to_id: dict[int, str] = {}
        self._removed: set[int] = set()
        self._next_label = 0

    def _ensure_capacity(self) -> None:
        if len(self._id_to_label) >= self._max_elements:
            self._max_elements *= 2
            self._index.resize_index(self._max_elements)

    def add(self, record_id: str, embedding: Sequence[float]) -> None:
        v = np.asarray(embedding, dtype=np.float32)
        if v.shape[0] != self.dim:
            raise ValueError(
                f"Embedding dim {v.shape[0]} does not match index dim {self.dim}"
            )
        self._ensure_capacity()
        if record_id in self._id_to_label:
            label = self._id_to_label[record_id]
        else:
            label = self._next_label
            self._next_label += 1
            self._id_to_label[record_id] = label
            self._label_to_id[label] = record_id
        self._index.add_items(v[None, :], np.asarray([label]))
        self._removed.discard(label)

    def remove(self, record_id: str) -> None:
        label = self._id_to_label.pop(record_id, None)
        if label is None:
            return
        self._label_to_id.pop(label, None)
        try:
            self._index.mark_deleted(label)
        except RuntimeError:
            pass
        self._removed.add(label)

    def search(
        self,
        embedding: Sequence[float],
        k: int = 10,
        threshold: float = 0.0,
        exclude_ids: Optional[Iterable[str]] = None,
    ) -> list[tuple[str, float]]:
        if len(self._id_to_label) == 0:
            return []
        v = np.asarray(embedding, dtype=np.float32)
        if v.shape[0] != self.dim:
            return []
        want = min(k + len(exclude_ids or ()) + len(self._removed), len(self._id_to_label))
        labels, distances = self._index.knn_query(v[None, :], k=want)
        excl = set(exclude_ids or ())
        out: list[tuple[str, float]] = []
        for label, dist in zip(labels[0], distances[0]):
            rid = self._label_to_id.get(int(label))
            if rid is None or rid in excl:
                continue
            # hnswlib returns cosine *distance* = 1 - cosine similarity
            sim = float(1.0 - dist)
            if sim < threshold:
                continue
            out.append((rid, sim))
            if len(out) >= k:
                break
        return out

    def __len__(self) -> int:
        return len(self._id_to_label)

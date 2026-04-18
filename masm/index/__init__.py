"""Vector index backends for MASM.

Expose a common `VectorIndex` interface with two backends:

* `BruteForceIndex` — linear scan over numpy arrays. Zero-dependency default.
* `HNSWIndex`       — approximate nearest-neighbor via hnswlib (optional).

Both implement the same contract so `InMemorySharedStore` can swap them
without changing call sites.
"""

from masm.index.base import VectorIndex
from masm.index.brute_force import BruteForceIndex

try:
    from masm.index.hnsw import HNSWIndex  # noqa: F401
except ImportError:  # hnswlib not installed
    HNSWIndex = None  # type: ignore[assignment]

__all__ = ["VectorIndex", "BruteForceIndex", "HNSWIndex"]

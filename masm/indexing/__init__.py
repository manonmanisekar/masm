"""Structural (non-vector) indexes for MASM.

The vector-similarity index lives in `masm.index`. This package is for
*structured* retrieval — entity/attribute lookups, etc. — that a cosine
search can't answer efficiently.

Available indexes:

* `EntityAttributeIndex` — map `(entity, attribute)` pairs to record IDs so
  callers can answer "what do we know about entity X?" or "who has an
  attribute Y?" in O(1) without scanning the whole store.
"""

from masm.indexing.entity_attribute import (
    EntityAttributeIndex,
    EntityAttributeExtractor,
    DefaultEntityAttributeExtractor,
)

__all__ = [
    "EntityAttributeIndex",
    "EntityAttributeExtractor",
    "DefaultEntityAttributeExtractor",
]

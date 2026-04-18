"""Memory merge strategies for resolving duplicates and combining information."""

from dataclasses import replace

from masm.core.memory import MemoryRecord, MemoryState


class MemoryMerger:
    """
    Merges two memory records that represent the same fact
    (duplicates or complementary information).
    """

    def merge(self, a: MemoryRecord, b: MemoryRecord) -> tuple[MemoryRecord, MemoryRecord]:
        """
        Merge two records into one. Takes the higher-confidence version as the base,
        combines tags, and preserves the richer metadata.

        Returns:
            (merged_record, superseded_record) — the winner and the loser
            marked as SUPERSEDED.
        """
        if b.confidence >= a.confidence:
            base, other = b, a
        else:
            base, other = a, b

        merged = replace(
            base,
            tags=tuple(set(base.tags) | set(other.tags)),
            confidence=max(a.confidence, b.confidence),
            metadata={
                **(base.metadata if isinstance(base.metadata, dict) else {}),
                "merged_from": [a.id, b.id],
                "merge_strategy": "confidence_based",
            },
        )

        superseded = replace(
            other,
            state=MemoryState.SUPERSEDED,
            superseded_by_id=merged.id,
        )

        return merged, superseded

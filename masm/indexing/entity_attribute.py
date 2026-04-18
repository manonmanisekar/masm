"""Entity-Attribute Index — structured (entity, attribute) → record_id lookups.

Vector similarity is great for fuzzy queries ("memories like this one") but
terrible for crisp structured queries ("what role does Alice play?" or
"which records mention Acme Corp?"). The `EntityAttributeIndex` fills that
gap: an inverted index keyed by two dimensions.

Extraction is pluggable. By default the index reads from:

  record.metadata["entities"]   -> Iterable[str]
  record.metadata["attributes"] -> Iterable[str]  or
                                   Mapping[str, Any]  (keys are attribute names)

Callers with richer schemas (NER output, domain-specific extractors) can
subclass `EntityAttributeExtractor` and pass it in.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Protocol

from masm.core.memory import MemoryRecord


class EntityAttributeExtractor(Protocol):
    """Return the `(entities, attributes)` pair for a record."""

    def extract(self, record: MemoryRecord) -> tuple[Iterable[str], Iterable[str]]: ...


class DefaultEntityAttributeExtractor:
    """Read entities/attributes from `record.metadata`.

    The metadata schema is intentionally permissive:

      * `metadata["entities"]`    — any iterable of strings.
      * `metadata["attributes"]`  — iterable of strings *or* a dict (keys are
        used as attribute names). Dict values are ignored by the index but
        remain available on the record for callers that want them.
    """

    def __init__(self, entity_key: str = "entities", attribute_key: str = "attributes"):
        self.entity_key = entity_key
        self.attribute_key = attribute_key

    def extract(self, record: MemoryRecord) -> tuple[list[str], list[str]]:
        md: Mapping[str, Any] = record.metadata or {}
        entities = self._coerce(md.get(self.entity_key))
        raw_attrs = md.get(self.attribute_key)
        if isinstance(raw_attrs, Mapping):
            attributes = [str(k) for k in raw_attrs.keys()]
        else:
            attributes = self._coerce(raw_attrs)
        return entities, attributes

    @staticmethod
    def _coerce(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        try:
            return [str(v) for v in value]
        except TypeError:
            return []


@dataclass
class EntityAttributeIndex:
    """Inverted index of `(entity, attribute) → {record_id}`.

    All operations are O(1) amortized. The index does not own the records —
    it just keeps ID references, so callers should still resolve IDs
    against the store to get the full `MemoryRecord`.
    """

    extractor: EntityAttributeExtractor = field(default_factory=DefaultEntityAttributeExtractor)
    _by_entity: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _by_attribute: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _by_pair: dict[tuple[str, str], set[str]] = field(default_factory=lambda: defaultdict(set))
    _per_record: dict[str, tuple[list[str], list[str]]] = field(default_factory=dict)

    # ---- Mutation ----

    def add(self, record: MemoryRecord) -> None:
        """Insert `record` (or replace its existing entry)."""
        if record.id in self._per_record:
            self.remove(record.id)
        entities, attributes = self.extractor.extract(record)
        entities = [e for e in (str(e).strip() for e in entities) if e]
        attributes = [a for a in (str(a).strip() for a in attributes) if a]
        for e in entities:
            self._by_entity[e].add(record.id)
        for a in attributes:
            self._by_attribute[a].add(record.id)
        for e in entities:
            for a in attributes:
                self._by_pair[(e, a)].add(record.id)
        self._per_record[record.id] = (entities, attributes)

    def remove(self, record_id: str) -> None:
        entry = self._per_record.pop(record_id, None)
        if entry is None:
            return
        entities, attributes = entry
        for e in entities:
            self._by_entity[e].discard(record_id)
            if not self._by_entity[e]:
                self._by_entity.pop(e, None)
        for a in attributes:
            self._by_attribute[a].discard(record_id)
            if not self._by_attribute[a]:
                self._by_attribute.pop(a, None)
        for e in entities:
            for a in attributes:
                key = (e, a)
                self._by_pair[key].discard(record_id)
                if not self._by_pair[key]:
                    self._by_pair.pop(key, None)

    # ---- Query ----

    def lookup(
        self,
        entity: Optional[str] = None,
        attribute: Optional[str] = None,
    ) -> list[str]:
        """Return record IDs matching the given filter.

        Supplying both `entity` and `attribute` is an AND; supplying neither
        raises `ValueError` (use `all_entities()` / `all_attributes()` for
        enumeration instead).
        """
        if entity is None and attribute is None:
            raise ValueError("lookup() requires at least one of entity/attribute")
        if entity is not None and attribute is not None:
            return sorted(self._by_pair.get((entity, attribute), ()))
        if entity is not None:
            return sorted(self._by_entity.get(entity, ()))
        return sorted(self._by_attribute.get(attribute, ()))  # type: ignore[arg-type]

    def entities_of(self, record_id: str) -> list[str]:
        entry = self._per_record.get(record_id)
        return list(entry[0]) if entry else []

    def attributes_of(self, record_id: str) -> list[str]:
        entry = self._per_record.get(record_id)
        return list(entry[1]) if entry else []

    def all_entities(self) -> list[str]:
        return sorted(self._by_entity.keys())

    def all_attributes(self) -> list[str]:
        return sorted(self._by_attribute.keys())

    def __len__(self) -> int:
        return len(self._per_record)

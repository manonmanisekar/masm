"""Memory Type Classifier — infer MemoryType from content heuristics.

Deterministic rule-based classifier that runs before storage when the caller
hasn't supplied an explicit `memory_type`. An LLM-backed classifier can be
plugged in later by subclassing and overriding `classify`.

Rules (checked in order, first match wins):

1. Starts with a procedural verb ("always ...", "never ...", "if X then Y",
   "to do X, first Y") → PROCEDURAL.
2. Contains explicit past-tense event markers ("yesterday", "today I", dates,
   "happened") → EPISODIC.
3. Hedged or derived ("probably", "seems", "suggests", "implies") → IMPLICIT.
4. Default → SEMANTIC.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from masm.core.memory import MemoryType


_PROCEDURAL_PATTERNS = [
    re.compile(r"^\s*(always|never|if\b|when\b|to\b|first,?\b|step\s*\d+)", re.I),
    re.compile(r"\b(you should|you must|don['’]t|do not)\b", re.I),
]
_EPISODIC_PATTERNS = [
    re.compile(r"\b(yesterday|today|last (week|month|year)|on \w+day)\b", re.I),
    re.compile(r"\b(happened|occurred|took place|met with|spoke to)\b", re.I),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]
_IMPLICIT_PATTERNS = [
    re.compile(r"\b(probably|likely|seems|appears|suggests?|implies|might|may)\b", re.I),
    re.compile(r"\b(based on|i (think|believe|guess))\b", re.I),
]


@dataclass
class MemoryTypeClassifier:
    """Rule-based classifier producing a `MemoryType` from raw content."""

    def classify(self, content: str) -> MemoryType:
        if not content:
            return MemoryType.SEMANTIC
        for p in _PROCEDURAL_PATTERNS:
            if p.search(content):
                return MemoryType.PROCEDURAL
        for p in _EPISODIC_PATTERNS:
            if p.search(content):
                return MemoryType.EPISODIC
        for p in _IMPLICIT_PATTERNS:
            if p.search(content):
                return MemoryType.IMPLICIT
        return MemoryType.SEMANTIC

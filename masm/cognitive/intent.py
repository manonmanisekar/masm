"""Intent Classifier — coarse labels for incoming queries.

Downstream ranking can use the intent to shift weights (e.g. FACT_LOOKUP
favors high confidence + semantic similarity; RECENT_ACTIVITY favors
recency; PROCEDURE_RECALL favors PROCEDURAL memories). Rule-based and
deterministic by default; override `classify` for LLM-backed variants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, unique


@unique
class QueryIntent(Enum):
    FACT_LOOKUP = "fact_lookup"
    RECENT_ACTIVITY = "recent_activity"
    PROCEDURE_RECALL = "procedure_recall"
    CONFLICT_CHECK = "conflict_check"
    UNKNOWN = "unknown"


_RECENT_PAT = re.compile(r"\b(recent|latest|last|yesterday|today|now)\b", re.I)
_PROCEDURE_PAT = re.compile(r"\b(how (do|to)|what is the (process|procedure)|steps? to)\b", re.I)
_CONFLICT_PAT = re.compile(r"\b(disagree|contradict|conflict|inconsist)\b", re.I)
_FACT_PAT = re.compile(r"^(who|what|where|when|which)\b", re.I)


@dataclass
class IntentClassifier:
    """Return a `QueryIntent` for a natural-language query string."""

    def classify(self, query: str) -> QueryIntent:
        if not query:
            return QueryIntent.UNKNOWN
        if _CONFLICT_PAT.search(query):
            return QueryIntent.CONFLICT_CHECK
        if _PROCEDURE_PAT.search(query):
            return QueryIntent.PROCEDURE_RECALL
        if _RECENT_PAT.search(query):
            return QueryIntent.RECENT_ACTIVITY
        if _FACT_PAT.search(query):
            return QueryIntent.FACT_LOOKUP
        return QueryIntent.UNKNOWN

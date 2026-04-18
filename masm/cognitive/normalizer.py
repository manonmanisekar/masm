"""Semantic Normalizer — canonicalize content before it hits the store.

The normalizer is deliberately small and deterministic:

1. Unicode NFC normalize, strip, collapse whitespace, lowercase.
2. Optional synonym expansion (caller-supplied mapping).
3. Optional stopword stripping for *tag-like* phrases.

It is not a full NLP pipeline. The goal is to make trivially-equivalent
content collapse into the same string so embeddings and tag sets line up.
Heavier normalization (coreference, entity canonicalization) should live in
a separate module.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Mapping, Optional


_DEFAULT_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "is", "are", "to", "in", "on", "at", "for",
})


@dataclass
class SemanticNormalizer:
    """Canonicalize raw content for deterministic dedup/tag comparison."""

    lowercase: bool = True
    strip_punctuation: bool = False
    collapse_whitespace: bool = True
    synonyms: Mapping[str, str] = field(default_factory=dict)
    stopwords: frozenset = field(default_factory=lambda: _DEFAULT_STOPWORDS)

    _punct_re = re.compile(r"[^\w\s-]")
    _ws_re = re.compile(r"\s+")

    def normalize_text(self, text: str) -> str:
        """Return a canonical form of `text` suitable for dedup/tag matching."""
        if not text:
            return ""
        s = unicodedata.normalize("NFC", text).strip()
        if self.lowercase:
            s = s.lower()
        if self.strip_punctuation:
            s = self._punct_re.sub(" ", s)
        if self.collapse_whitespace:
            s = self._ws_re.sub(" ", s).strip()
        if self.synonyms:
            # Whole-word replacement only — avoids mangling substrings.
            for src, dst in self.synonyms.items():
                s = re.sub(rf"\b{re.escape(src.lower())}\b", dst.lower(), s)
        return s

    def normalize_tags(self, tags: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Lowercase, dedupe, strip stopwords and blanks. Stable ordering."""
        seen: dict[str, None] = {}
        for t in tags:
            t = self.normalize_text(t)
            if not t or t in self.stopwords:
                continue
            seen.setdefault(t, None)
        return tuple(seen.keys())

    def normalize_record_fields(
        self,
        content: str,
        tags: Optional[tuple[str, ...]] = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Convenience helper for applying both normalizations together."""
        return self.normalize_text(content), self.normalize_tags(tags or ())

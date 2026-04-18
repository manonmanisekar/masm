"""Cognitive layer — pre/post-processing hooks for MASM.

These components implement the "Cognitive Memory Orchestrator" box in the
architecture diagram. They are all optional and composable:

* `SemanticNormalizer`    — canonicalize content before storage.
* `MemoryTypeClassifier`  — infer `MemoryType` from content when unset.
* `IntentClassifier`      — tag queries with coarse intent labels.
* `TrustEngine`           — maintain dynamic per-agent authority scores.
"""

from masm.cognitive.normalizer import SemanticNormalizer
from masm.cognitive.type_classifier import MemoryTypeClassifier
from masm.cognitive.intent import IntentClassifier, QueryIntent
from masm.cognitive.trust import TrustEngine

__all__ = [
    "SemanticNormalizer",
    "MemoryTypeClassifier",
    "IntentClassifier",
    "QueryIntent",
    "TrustEngine",
]

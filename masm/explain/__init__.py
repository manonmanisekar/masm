"""Human-readable and machine-readable explanations for MASM decisions.

Currently exposes `ConflictExplainer` for turning a `ConflictEvent` plus the
two records it references into a structured `ConflictExplanation` (factor
breakdown + one-line summary). Future additions — write-path explainers,
retrieval explainers — will live in this package.
"""

from masm.explain.conflict_explainer import (
    ConflictExplainer,
    ConflictExplanation,
    ExplanationFactor,
)

__all__ = [
    "ConflictExplainer",
    "ConflictExplanation",
    "ExplanationFactor",
]

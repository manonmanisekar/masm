"""LLM-based semantic conflict resolution for memories that can't be resolved by simple rules."""

CONFLICT_RESOLUTION_PROMPT = """
You are a memory conflict resolver for a multi-agent AI system.

Two agents have recorded conflicting memories about the same topic.

Memory A:
- Content: {memory_a_content}
- Author: Agent {memory_a_author}
- Confidence: {memory_a_confidence}
- Recorded at: {memory_a_timestamp}
- Evidence: {memory_a_evidence}
- Tags: {memory_a_tags}

Memory B:
- Content: {memory_b_content}
- Author: Agent {memory_b_author}
- Confidence: {memory_b_confidence}
- Recorded at: {memory_b_timestamp}
- Evidence: {memory_b_evidence}
- Tags: {memory_b_tags}

Determine the correct resolution:

1. If one memory is simply a more detailed version of the other, MERGE them.
2. If they contradict each other and one has stronger evidence or is more recent, pick the WINNER.
3. If the contradiction cannot be resolved, mark as UNRESOLVED.

Respond in JSON:
{{
  "resolution": "merge" | "winner_a" | "winner_b" | "unresolved",
  "merged_content": "...",
  "reasoning": "...",
  "confidence": 0.0-1.0
}}
"""


class SemanticConflictResolver:
    """
    Uses an LLM to resolve conflicts that can't be handled by simple strategies.

    This is an optional component — requires the 'semantic' extra (openai).
    Falls back to LWW if no LLM client is provided.
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o-mini"):
        self.llm_client = llm_client
        self.model = model

    async def resolve(self, memory_a, memory_b) -> dict:
        """
        Use LLM to resolve a conflict between two memories.

        Returns dict with: resolution, merged_content, reasoning, confidence
        """
        if self.llm_client is None:
            # Fallback to simple heuristic
            return self._heuristic_resolve(memory_a, memory_b)

        prompt = CONFLICT_RESOLUTION_PROMPT.format(
            memory_a_content=memory_a.content,
            memory_a_author=memory_a.author_agent_id,
            memory_a_confidence=memory_a.confidence,
            memory_a_timestamp=memory_a.created_at.isoformat(),
            memory_a_evidence=memory_a.metadata.get("evidence", "None"),
            memory_a_tags=", ".join(memory_a.tags) if memory_a.tags else "None",
            memory_b_content=memory_b.content,
            memory_b_author=memory_b.author_agent_id,
            memory_b_confidence=memory_b.confidence,
            memory_b_timestamp=memory_b.created_at.isoformat(),
            memory_b_evidence=memory_b.metadata.get("evidence", "None"),
            memory_b_tags=", ".join(memory_b.tags) if memory_b.tags else "None",
        )

        import json

        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def _heuristic_resolve(self, memory_a, memory_b) -> dict:
        """Simple heuristic fallback: prefer higher confidence, then more recent."""
        if memory_b.confidence > memory_a.confidence:
            return {
                "resolution": "winner_b",
                "merged_content": None,
                "reasoning": "Higher confidence",
                "confidence": memory_b.confidence,
            }
        elif memory_a.confidence > memory_b.confidence:
            return {
                "resolution": "winner_a",
                "merged_content": None,
                "reasoning": "Higher confidence",
                "confidence": memory_a.confidence,
            }
        elif memory_b.created_at >= memory_a.created_at:
            return {
                "resolution": "winner_b",
                "merged_content": None,
                "reasoning": "More recent (tie-breaker)",
                "confidence": memory_b.confidence,
            }
        else:
            return {
                "resolution": "winner_a",
                "merged_content": None,
                "reasoning": "More recent (tie-breaker)",
                "confidence": memory_a.confidence,
            }

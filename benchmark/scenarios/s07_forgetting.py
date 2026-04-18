"""S07: Coordinated Forgetting (GDPR Compliance).

A 'forget request' targets a specific memory. Verify it's removed from all
agents' views, derived memories are cascaded, and audit trail remains.
"""

import time

from benchmark.scenarios.scenario_base import BenchmarkScenario, ScenarioResult
from masm.core.memory import MemoryRecord, MemoryState


FORGET_SCENARIOS = [
    {
        "id": "fg_001",
        "title": "Forget customer PII",
        "original": {
            "content": "Customer email is john.doe@example.com",
            "tags": ["pii", "contact"],
        },
        "derived": [
            {
                "content": "Contacted john.doe@example.com about renewal",
                "tags": ["pii", "contact", "sales"],
            },
            {
                "content": "john.doe@example.com opened support ticket #789",
                "tags": ["pii", "support"],
            },
        ],
        "unrelated": {
            "content": "Support ticket #789 was resolved",
            "tags": ["support"],
        },
    },
    {
        "id": "fg_002",
        "title": "Forget payment details",
        "original": {
            "content": "Customer credit card ending in 4242",
            "tags": ["pii", "payment"],
        },
        "derived": [
            {
                "content": "Charged card 4242 for $99 subscription",
                "tags": ["pii", "payment", "billing"],
            },
        ],
        "unrelated": {
            "content": "Subscription plan is Pro tier",
            "tags": ["billing", "plan"],
        },
    },
    {
        "id": "fg_003",
        "title": "Forget personal preferences",
        "original": {
            "content": "Customer has medical condition requiring accessibility features",
            "tags": ["pii", "health", "accessibility"],
        },
        "derived": [
            {
                "content": "Enabled high-contrast mode due to customer's medical needs",
                "tags": ["pii", "health", "settings"],
            },
        ],
        "unrelated": {
            "content": "High-contrast mode is available in settings",
            "tags": ["settings", "features"],
        },
    },
]


class ForgettingScenario(BenchmarkScenario):
    """Benchmark: GDPR-compliant coordinated forgetting."""

    @property
    def name(self) -> str:
        return "s07_forgetting"

    @property
    def description(self) -> str:
        return "Coordinated forgetting with cascade and audit preservation"

    async def setup(self, store, num_agents: int = 5, **kwargs):
        self.store = store
        self.num_agents = num_agents
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.latencies: list[float] = []

    async def run(self) -> ScenarioResult:
        errors = []
        total_forget_targets = 0
        correctly_forgotten = 0
        cascade_correct = 0
        total_cascades = 0
        unrelated_preserved = 0
        total_unrelated = 0
        audit_preserved = 0
        total_audit_checks = 0
        total_operations = 0

        for scenario in FORGET_SCENARIOS:
            # Write original memory
            original = MemoryRecord(
                content=scenario["original"]["content"],
                author_agent_id=self.agent_ids[0],
                tags=scenario["original"]["tags"],
                confidence=0.95,
            )
            await self.store.write(original)
            total_operations += 1

            # Write derived memories (superseding the original)
            derived_ids = []
            prev_id = original.id
            for derived_def in scenario["derived"]:
                derived = MemoryRecord(
                    content=derived_def["content"],
                    author_agent_id=self.agent_ids[1],
                    tags=derived_def["tags"],
                    confidence=0.9,
                    supersedes_id=prev_id,
                )
                await self.store.write(derived)
                total_operations += 1
                derived_ids.append(derived.id)
                prev_id = derived.id

            # Write unrelated memory (should NOT be forgotten)
            unrelated = MemoryRecord(
                content=scenario["unrelated"]["content"],
                author_agent_id=self.agent_ids[min(2, self.num_agents - 1)],
                tags=scenario["unrelated"]["tags"],
                confidence=0.9,
            )
            await self.store.write(unrelated)
            total_operations += 1

            # Issue forget request on original
            t0 = time.perf_counter()
            success = await self.store.forget(
                original.id,
                agent_id="admin",
                reason="gdpr_request",
                cascade=True,
            )
            self.latencies.append((time.perf_counter() - t0) * 1000)
            total_operations += 1

            total_forget_targets += 1

            # Check 1: Original is forgotten
            if success and hasattr(self.store, "_records"):
                orig_record = self.store._records.get(original.id)
                if orig_record and orig_record.state == MemoryState.FORGOTTEN:
                    correctly_forgotten += 1
                    if orig_record.content == "[FORGOTTEN]":
                        correctly_forgotten += 0  # Already counted

            # Check 2: Derived memories are cascaded
            for did in derived_ids:
                total_cascades += 1
                if hasattr(self.store, "_records"):
                    derived_rec = self.store._records.get(did)
                    if derived_rec and derived_rec.state == MemoryState.FORGOTTEN:
                        cascade_correct += 1

            # Check 3: Unrelated memory is preserved
            total_unrelated += 1
            unrelated_results = await self.store.read(
                agent_id=self.agent_ids[min(3, self.num_agents - 1)],
                tags=scenario["unrelated"]["tags"],
            )
            total_operations += 1
            if any(r.id == unrelated.id for r in unrelated_results):
                unrelated_preserved += 1

            # Check 4: Audit trail preserved
            total_audit_checks += 1
            audit = await self.store.get_audit_log(record_id=original.id)
            total_operations += 1
            if any(e["operation"] == "forget" for e in audit):
                audit_preserved += 1

            # Check 5: Forgotten records invisible to agents
            for agent_id in self.agent_ids:
                pii_results = await self.store.read(
                    agent_id=agent_id,
                    tags=["pii"],
                )
                total_operations += 1
                for r in pii_results:
                    if r.id == original.id or r.id in derived_ids:
                        errors.append(f"Agent {agent_id} can still see forgotten record")

        # Metrics
        forget_completeness = correctly_forgotten / max(total_forget_targets, 1)
        cascade_accuracy = cascade_correct / max(total_cascades, 1)
        unrelated_preservation = unrelated_preserved / max(total_unrelated, 1)
        audit_preservation = audit_preserved / max(total_audit_checks, 1)

        score = (
            forget_completeness * 0.3
            + cascade_accuracy * 0.3
            + unrelated_preservation * 0.2
            + audit_preservation * 0.2
        )

        sorted_latencies = sorted(self.latencies) if self.latencies else [0.0]
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return ScenarioResult(
            scenario_name=self.name,
            score=round(score, 4),
            metrics={
                "forget_completeness": round(forget_completeness, 4),
                "cascade_accuracy": round(cascade_accuracy, 4),
                "unrelated_preservation": round(unrelated_preservation, 4),
                "audit_preservation": round(audit_preservation, 4),
                "total_forget_targets": total_forget_targets,
                "total_cascades": total_cascades,
                "errors_count": len(errors),
            },
            latency_p50_ms=round(p50, 2),
            latency_p99_ms=round(p99, 2),
            num_agents=self.num_agents,
            num_operations=total_operations,
            errors=errors,
        )

    async def teardown(self):
        self.latencies.clear()

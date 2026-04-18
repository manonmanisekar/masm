"""S10: End-to-End Multi-Agent Task Completion.

Realistic multi-agent workflows where shared memory directly affects
task outcome quality. Tests customer support, research, and code review flows.
"""

import time
import numpy as np

from benchmark.scenarios.scenario_base import BenchmarkScenario, ScenarioResult
from masm.core.memory import MemoryRecord, MemoryType


# Workflow 1: Customer support — greeter → specialist → resolver
SUPPORT_WORKFLOW = {
    "id": "e2e_support",
    "title": "Customer support escalation",
    "steps": [
        {
            "agent": "greeter",
            "writes": [
                {"content": "Customer Jane reports billing discrepancy of $49.99", "tags": ["billing", "issue", "customer"], "type": "episodic", "critical": True},
                {"content": "Customer has been with us for 2 years", "tags": ["customer", "tenure"], "type": "semantic", "critical": True},
                {"content": "Customer tone: frustrated but polite", "tags": ["sentiment"], "type": "episodic", "critical": False},
            ],
            "reads": [],
        },
        {
            "agent": "specialist",
            "writes": [
                {"content": "Found duplicate charge on invoice #INV-2025-1234", "tags": ["billing", "investigation"], "type": "semantic", "critical": True},
                {"content": "Duplicate caused by payment retry after timeout", "tags": ["billing", "root_cause"], "type": "semantic", "critical": True},
            ],
            "reads": ["billing", "customer"],
            "must_know": ["billing discrepancy", "2 years"],
        },
        {
            "agent": "resolver",
            "writes": [
                {"content": "Refund of $49.99 issued to original payment method", "tags": ["billing", "resolution"], "type": "episodic", "critical": True},
                {"content": "Applied 10% loyalty discount for next billing cycle", "tags": ["customer", "retention"], "type": "procedural", "critical": True},
            ],
            "reads": ["billing", "customer", "investigation"],
            "must_know": ["duplicate charge", "root_cause", "2 years"],
        },
    ],
    "expected_outcome_facts": [
        "billing discrepancy",
        "duplicate charge",
        "refund issued",
        "loyalty discount",
    ],
}

# Workflow 2: Research team — searcher → reader → synthesizer → writer
RESEARCH_WORKFLOW = {
    "id": "e2e_research",
    "title": "Multi-agent research synthesis",
    "steps": [
        {
            "agent": "searcher",
            "writes": [
                {"content": "Found 3 relevant papers on transformer efficiency", "tags": ["research", "sources"], "type": "episodic", "critical": True},
                {"content": "Key paper: 'FlashAttention 3' achieves 2x speedup", "tags": ["research", "finding"], "type": "semantic", "critical": True},
                {"content": "Secondary paper: 'Sparse transformers reduce compute by 40%'", "tags": ["research", "finding"], "type": "semantic", "critical": True},
            ],
            "reads": [],
        },
        {
            "agent": "reader",
            "writes": [
                {"content": "FlashAttention 3 uses tiled computation to reduce memory access", "tags": ["research", "analysis"], "type": "semantic", "critical": True},
                {"content": "Sparse transformers trade 5% accuracy for 40% compute savings", "tags": ["research", "analysis", "tradeoff"], "type": "semantic", "critical": True},
            ],
            "reads": ["research", "finding"],
            "must_know": ["FlashAttention", "Sparse transformers"],
        },
        {
            "agent": "synthesizer",
            "writes": [
                {"content": "Two complementary approaches: memory optimization (FlashAttention) and compute reduction (sparsity)", "tags": ["research", "synthesis"], "type": "semantic", "critical": True},
            ],
            "reads": ["research", "analysis"],
            "must_know": ["tiled computation", "compute savings"],
        },
        {
            "agent": "writer",
            "writes": [
                {"content": "Report: Transformer efficiency gains through FlashAttention (2x memory) and sparsity (40% compute)", "tags": ["research", "output"], "type": "procedural", "critical": True},
            ],
            "reads": ["research", "synthesis"],
            "must_know": ["complementary approaches"],
        },
    ],
    "expected_outcome_facts": [
        "FlashAttention",
        "sparse",
        "memory optimization",
        "compute reduction",
    ],
}

# Workflow 3: Code review — style → logic → security → unified feedback
CODE_REVIEW_WORKFLOW = {
    "id": "e2e_codereview",
    "title": "Multi-agent code review",
    "steps": [
        {
            "agent": "style_reviewer",
            "writes": [
                {"content": "Style: 3 functions exceed 50-line limit", "tags": ["review", "style"], "type": "semantic", "critical": True},
                {"content": "Style: inconsistent naming (camelCase mixed with snake_case)", "tags": ["review", "style"], "type": "semantic", "critical": False},
            ],
            "reads": [],
        },
        {
            "agent": "logic_reviewer",
            "writes": [
                {"content": "Logic: off-by-one error in pagination (line 142)", "tags": ["review", "logic", "bug"], "type": "semantic", "critical": True},
                {"content": "Logic: race condition in cache invalidation", "tags": ["review", "logic", "bug"], "type": "semantic", "critical": True},
            ],
            "reads": [],
        },
        {
            "agent": "security_reviewer",
            "writes": [
                {"content": "Security: SQL injection vulnerability in search endpoint", "tags": ["review", "security", "critical_bug"], "type": "semantic", "critical": True},
                {"content": "Security: missing rate limiting on auth endpoint", "tags": ["review", "security"], "type": "semantic", "critical": True},
            ],
            "reads": [],
        },
        {
            "agent": "review_aggregator",
            "writes": [
                {"content": "Unified review: 1 critical (SQL injection), 2 high (race condition, rate limit), 1 medium (off-by-one), 2 low (style)", "tags": ["review", "summary"], "type": "procedural", "critical": True},
            ],
            "reads": ["review"],
            "must_know": ["SQL injection", "race condition", "off-by-one", "rate limiting"],
        },
    ],
    "expected_outcome_facts": [
        "SQL injection",
        "race condition",
        "off-by-one",
        "rate limiting",
    ],
}

ALL_WORKFLOWS = [SUPPORT_WORKFLOW, RESEARCH_WORKFLOW, CODE_REVIEW_WORKFLOW]


class E2EWorkflowScenario(BenchmarkScenario):
    """Benchmark: end-to-end multi-agent task completion quality."""

    @property
    def name(self) -> str:
        return "s10_e2e_workflow"

    @property
    def description(self) -> str:
        return "End-to-end multi-agent workflow task completion"

    async def setup(self, store, num_agents: int = 5, **kwargs):
        self.store = store
        self.num_agents = num_agents
        self.latencies: list[float] = []

    async def run(self) -> ScenarioResult:
        errors = []
        total_operations = 0
        workflow_scores = []

        for workflow in ALL_WORKFLOWS:
            wf_memories_used = 0
            wf_total_memories = 0
            wf_context_hits = 0
            wf_context_checks = 0

            for step in workflow["steps"]:
                # Agent reads context from previous steps
                if step.get("reads"):
                    t0 = time.perf_counter()
                    context = await self.store.read(
                        agent_id=step["agent"],
                        tags=step["reads"],
                        limit=20,
                    )
                    self.latencies.append((time.perf_counter() - t0) * 1000)
                    total_operations += 1

                    # Check if required knowledge was available
                    if "must_know" in step:
                        context_text = " ".join(r.content.lower() for r in context)
                        for fact in step["must_know"]:
                            wf_context_checks += 1
                            if fact.lower() in context_text:
                                wf_context_hits += 1

                # Agent writes its findings
                for mem_def in step["writes"]:
                    record = MemoryRecord(
                        content=mem_def["content"],
                        author_agent_id=step["agent"],
                        tags=mem_def["tags"],
                        memory_type=MemoryType(mem_def["type"]),
                        confidence=0.9,
                        metadata={"workflow": workflow["id"]},
                    )
                    t0 = time.perf_counter()
                    await self.store.write(record)
                    self.latencies.append((time.perf_counter() - t0) * 1000)
                    total_operations += 1
                    wf_total_memories += 1

            # Check final outcome quality
            final_results = await self.store.read(
                agent_id="evaluator",
                tags=list({t for s in workflow["steps"] for m in s["writes"] for t in m["tags"]}),
                limit=50,
            )
            total_operations += 1

            all_content = " ".join(r.content.lower() for r in final_results)
            outcome_hits = 0
            for fact in workflow["expected_outcome_facts"]:
                if fact.lower() in all_content:
                    outcome_hits += 1
                    wf_memories_used += 1

            outcome_quality = outcome_hits / max(len(workflow["expected_outcome_facts"]), 1)
            context_quality = wf_context_hits / max(wf_context_checks, 1)

            wf_score = outcome_quality * 0.6 + context_quality * 0.4
            workflow_scores.append({
                "workflow": workflow["id"],
                "outcome_quality": round(outcome_quality, 4),
                "context_quality": round(context_quality, 4),
                "score": round(wf_score, 4),
            })

        # Aggregate
        avg_score = np.mean([ws["score"] for ws in workflow_scores])
        avg_outcome = np.mean([ws["outcome_quality"] for ws in workflow_scores])
        avg_context = np.mean([ws["context_quality"] for ws in workflow_scores])

        sorted_latencies = sorted(self.latencies) if self.latencies else [0.0]
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return ScenarioResult(
            scenario_name=self.name,
            score=round(float(avg_score), 4),
            metrics={
                "avg_outcome_quality": round(float(avg_outcome), 4),
                "avg_context_quality": round(float(avg_context), 4),
                "per_workflow": workflow_scores,
            },
            latency_p50_ms=round(p50, 2),
            latency_p99_ms=round(p99, 2),
            num_agents=self.num_agents,
            num_operations=total_operations,
            errors=errors,
        )

    async def teardown(self):
        self.latencies.clear()

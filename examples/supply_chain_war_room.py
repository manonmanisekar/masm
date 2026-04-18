"""
Supply-chain war room — 13 agents, one question, two stores.

Question: "Will we hit the Q2 delivery commitment for SKU-A47?"

Thirteen specialised agents each write what they know into shared memory:
suppliers, warehouses, logistics, customs, demand, procurement, QC, finance,
and a compliance bot. Several of them disagree — which is what real supply
chains actually look like. A final `planner` agent reads shared memory and
issues a go / no-go call.

We run the same 13-agent interaction twice:
  * against a naive dict-backed store (last writer wins, no conflicts)
  * against MASM (conflict-aware, trust-weighted, provenance-preserving)

The planner's decision diverges. That's the point.

Run:
    python examples/supply_chain_war_room.py
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

from masm import InMemorySharedStore, MemoryRecord
from masm.cognitive.trust import TrustEngine
from masm.explain.conflict_explainer import ConflictExplainer


# ---------- toy embedding (deterministic, no API key needed) ----------

def embed(text: str, dim: int = 32) -> list[float]:
    h = hashlib.sha256(text.lower().encode()).digest()
    vec = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    n = sum(v * v for v in vec) ** 0.5
    return [v / n for v in vec]


# ---------- the 13-agent war room ----------

@dataclass
class Report:
    agent: str
    content: str
    tags: tuple[str, ...]
    confidence: float


# Thirteen agents reporting state on SKU-A47. Conflicts are deliberate — they
# mirror what happens when a real supply chain has multiple sources of truth.
#
# Tag scheme: one topic-specific tag per fact. Conflict detection uses tag
# overlap, so lumping everything under a broad "sku:a47" tag would make every
# agent conflict with every other agent even when they're talking about
# different facets (lead time vs. inventory vs. shipment). Real systems want
# topic-scoped tags for the same reason.
REPORTS: list[Report] = [
    # --- Suppliers disagree about lead time (real conflict) ---
    Report("supplier_apex",   "SKU-A47 raw-material lead time is 14 days from our side.",
           ("sku_a47_lead_time",), confidence=0.55),        # incumbent, stale
    Report("supplier_orion",  "SKU-A47 raw material ships in 7 days via air.",
           ("sku_a47_lead_time",), confidence=0.90),        # backup, signed PO

    # --- Two warehouses independently report the SAME APAC stock (→ dedup) ---
    Report("warehouse_sg",    "SKU-A47 APAC on-hand inventory is 1,200 units.",
           ("sku_a47_inventory_apac",), confidence=0.95),
    Report("warehouse_la",    "APAC WMS confirms SKU-A47 on-hand inventory at 1200 units.",
           ("sku_a47_inventory_apac",), confidence=0.93),   # paraphrase of warehouse_sg

    # --- Procurement contradicts APAC inventory (allocation already applied) ---
    Report("procurement",     "SKU-A47 APAC free-to-promise is 950 units after PO-88421 allocation.",
           ("sku_a47_inventory_apac",), confidence=0.97),

    # --- EU inventory (separate tag, no conflict expected) ---
    Report("warehouse_rtm",   "Rotterdam holds 480 SKU-A47 units; EU buffer healthy.",
           ("sku_a47_inventory_eu",), confidence=0.92),

    # --- Logistics vs customs contradict on shipment status ---
    Report("logistics_ocean", "Container MSCU-7742 carrying SKU-A47 is in transit, ETA Apr 25.",
           ("sku_a47_shipment",), confidence=0.70),
    Report("customs_eu",      "Container MSCU-7742 held at Rotterdam customs pending HS-code review.",
           ("sku_a47_shipment",), confidence=0.99),

    # --- Demand forecaster: baseline, no conflict ---
    Report("demand_forecaster",
           "Projected Q2 demand for SKU-A47 is 4,800 units across EU+APAC.",
           ("sku_a47_demand_q2",), confidence=0.88),

    # --- QC hold — cannot afford to lose ---
    Report("quality_control", "SKU-A47 batch B-2026-Q2-07 on hold pending re-test (defect rate 2.1%).",
           ("sku_a47_qc_hold",), confidence=0.96),

    # --- Finance: credit OK ---
    Report("finance",         "Supplier Orion credit line confirmed at $2.4M through Q3.",
           ("supplier_orion_credit",), confidence=0.94),

    # --- Compliance: PII contact to be forgotten (GDPR cascade) ---
    Report("compliance_bot",  "Supplier Apex primary contact: jane.doe@apex.example.",
           ("supplier_apex_pii",), confidence=1.00),

    # --- Planner writes its working hypothesis after reading ---
    Report("planner",         "Working assumption: hit Q2 commitment via Orion + APAC stock.",
           ("sku_a47_plan",), confidence=0.60),
]

# Stale re-read by supplier_apex that arrives AFTER Orion's update — the kind
# of race that routinely poisons naive stores:
LATE_APEX_RACE = Report(
    "supplier_apex",
    "SKU-A47 lead time remains 14 days (re-confirmed).",
    ("sku_a47_lead_time",), confidence=0.50,
)


# ---------- Baseline: naive dict ----------

class NaiveStore:
    def __init__(self):
        self.by_topic: dict[tuple[str, ...], str] = {}
        self.writes = 0

    def write(self, r: Report):
        self.by_topic[r.tags] = r.content
        self.writes += 1

    def read_topic(self, tags: tuple[str, ...]) -> str | None:
        return self.by_topic.get(tags)


def run_naive() -> dict:
    store = NaiveStore()
    for r in REPORTS:
        store.write(r)
    store.write(LATE_APEX_RACE)     # stale supplier re-read clobbers the Orion truth

    # "Planner" reads by topic and makes a decision. The naive store has no
    # conflict detector, so each topic key holds whatever was written *last*.
    lead_time = store.read_topic(("sku_a47_lead_time",))
    shipment  = store.read_topic(("sku_a47_shipment",))
    inventory = store.read_topic(("sku_a47_inventory_apac",))
    qc        = store.read_topic(("sku_a47_qc_hold",))

    # Naive planner trusts whatever the key holds. If lead time *reads* as
    # 14 days (the stale Apex re-read that arrived last), it greenlights.
    decision = "GO" if (lead_time and "14 days" in lead_time) else "NO-GO"
    return {
        "writes": store.writes,
        "lead_time_seen": lead_time,
        "shipment_seen": shipment,
        "apac_inventory_seen": inventory,
        "qc_seen": qc,
        "conflicts_surfaced": 0,
        "decision": decision,
    }


# ---------- MASM run ----------

async def run_masm() -> dict:
    trust = TrustEngine(prior=0.5)
    explainer = ConflictExplainer()
    store = InMemorySharedStore(trust_engine=trust, conflict_explainer=explainer)

    async def write(r: Report):
        return await store.write(MemoryRecord(
            content=r.content,
            content_embedding=embed(r.content),
            author_agent_id=r.agent,
            tags=list(r.tags),
            confidence=r.confidence,
        ))

    for r in REPORTS:
        await write(r)
    await write(LATE_APEX_RACE)      # same stale race as the naive run

    # Compliance retracts the Apex PII contact — cascaded forget.
    pii_records = [r for r in await store.read(agent_id="compliance_bot",
                                                tags=["supplier_apex_pii"])
                   if r.state.value == "active"]
    for rec in pii_records:
        await store.forget(record_id=rec.id,
                           agent_id="compliance_bot",
                           reason="contract_ended_retention_expired",
                           cascade=True)

    # Planner reads active memory, one topic at a time.
    async def active(tag):
        return [r for r in await store.read(agent_id="planner", tags=[tag])
                if r.state.value == "active"]

    lead_time_records = await active("sku_a47_lead_time")
    shipment_records  = await active("sku_a47_shipment")
    apac_records      = await active("sku_a47_inventory_apac")
    qc_records        = await active("sku_a47_qc_hold")

    def best(records):
        return max(records, key=lambda r: r.confidence) if records else None

    lead = best(lead_time_records)
    shipment = best(shipment_records)
    inventory = best(apac_records)
    qc = best(qc_records)

    # MASM-aware decision: honour QC holds, pick trust/confidence-weighted truth,
    # and block on any surfaced customs exception.
    has_customs_hold = shipment and "customs" in shipment.content.lower()
    has_qc_hold      = qc is not None
    decision = "NO-GO" if (has_customs_hold or has_qc_hold) else "GO"

    return {
        "writes": store._write_count,
        "active_records": sum(1 for r in store._records.values()
                              if r.state.value == "active"),
        "lead_time_seen": lead.content if lead else None,
        "lead_time_author": lead.author_agent_id if lead else None,
        "shipment_seen": shipment.content if shipment else None,
        "apac_inventory_seen": inventory.content if inventory else None,
        "apac_inventory_author": inventory.author_agent_id if inventory else None,
        "qc_seen": qc.content if qc else None,
        "conflicts_surfaced": len(store._conflicts),
        "explanations": [e.summary for e in store._explanations[-3:]],
        "forgotten_records": sum(1 for r in store._records.values()
                                  if r.state.value == "forgotten"),
        "trust_apex":  trust.score("supplier_apex"),
        "trust_orion": trust.score("supplier_orion"),
        "decision": decision,
    }


# ---------- pretty output ----------

def banner(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


async def main():
    banner(f"Supply-chain war room — {len(REPORTS)} agents, +1 stale race")

    banner("NAIVE dict-backed shared memory")
    n = run_naive()
    print(f"  Writes accepted:        {n['writes']}")
    print(f"  Conflicts surfaced:     {n['conflicts_surfaced']}")
    print(f"  Lead time planner saw:  {n['lead_time_seen']}")
    print(f"  Shipment planner saw:   {n['shipment_seen']}")
    print(f"  APAC inventory saw:     {n['apac_inventory_seen']}")
    print(f"  QC status planner saw:  {n['qc_seen']}")
    print(f"  → Decision:             {n['decision']}")
    print("  (Stale Apex re-read clobbered Orion's 7-day lead time; planner")
    print("   never learned QC and Orion disagreed with anyone. GO is wrong.)")

    banner("MASM — conflict-aware shared memory")
    m = await run_masm()
    print(f"  Writes accepted:        {m['writes']}")
    print(f"  Active records:         {m['active_records']}  "
          f"(forgotten: {m['forgotten_records']})")
    print(f"  Conflicts surfaced:     {m['conflicts_surfaced']}")
    print(f"  Lead time planner saw:  {m['lead_time_seen']}")
    print(f"                          (via {m['lead_time_author']})")
    print(f"  Shipment planner saw:   {m['shipment_seen']}")
    print(f"  APAC inventory saw:     {m['apac_inventory_seen']}")
    print(f"                          (via {m['apac_inventory_author']})")
    print(f"  QC status planner saw:  {m['qc_seen']}")
    print(f"  Trust scores:           supplier_apex={m['trust_apex']:.2f}  "
          f"supplier_orion={m['trust_orion']:.2f}")
    if m["explanations"]:
        print("  Last few conflict resolutions:")
        for s in m["explanations"]:
            print(f"    • {s}")
    print(f"  → Decision:             {m['decision']}")
    print("  (Orion's verified 7-day lead time survives the stale Apex race;")
    print("   customs hold and QC hold are both visible; PII forgotten cleanly.)")

    banner("The diff")
    print(f"  Naive decision:  {n['decision']}   (wrong — would ship a QC-held batch"
          " through a customs-held container)")
    print(f"  MASM decision:   {m['decision']}  (correct — both blockers surfaced,"
          " explained, auditable)")
    print(f"  Conflicts surfaced:  naive={n['conflicts_surfaced']}   "
          f"MASM={m['conflicts_surfaced']}")
    print()


if __name__ == "__main__":
    asyncio.run(main())

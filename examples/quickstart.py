"""MASM Quickstart — 3 agents sharing memory in under 50 lines."""

import asyncio
from masm import InMemorySharedStore, MemoryRecord, Agent


async def main():
    # 1. Create a shared memory store
    store = InMemorySharedStore()

    # 2. Define agents
    researcher = Agent(id="researcher", role="finds information")
    analyst = Agent(id="analyst", role="analyzes data")
    writer = Agent(id="writer", role="writes reports")

    # 3. Researcher discovers a fact and writes to shared memory.
    #    Embeddings let MASM distinguish contradictions from paraphrases;
    #    in real code use OpenAI / sentence-transformers, here stubbed.
    record, conflicts = await store.write(MemoryRecord(
        content="Revenue grew 23% YoY in Q3 2025",
        content_embedding=[1.0, 0.0, 0.0],
        author_agent_id="researcher",
        tags=["revenue", "q3", "growth"],
        confidence=0.95,
    ))
    print(f"Researcher wrote: {record.content}")

    # 4. Analyst reads relevant memories
    context = await store.read(agent_id="analyst", tags=["revenue"], limit=5)
    print(f"Analyst sees {len(context)} memories about revenue")

    # 5. Analyst writes a conflicting fact — conflict detected automatically!
    record2, conflicts = await store.write(MemoryRecord(
        content="Revenue grew 21% YoY in Q3 2025",
        content_embedding=[0.0, 1.0, 0.0],
        author_agent_id="analyst",
        tags=["revenue", "q3", "growth"],
        confidence=0.85,
    ))
    print(f"Analyst wrote: {record2.content}")
    print(f"Conflicts detected: {len(conflicts)}")

    # 6. Writer reads shared memory to draft a report
    all_context = await store.read(agent_id="writer", tags=["revenue", "growth"])
    print(f"Writer has {len(all_context)} memories to work with")

    # 7. Check store stats
    stats = await store.stats()
    print(f"\nStore stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())

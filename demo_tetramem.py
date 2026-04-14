"""
TetraMem-XL Demonstration — Full System Walkthrough

Showcases all core capabilities:
  1. Pure geometric memory storage on tetrahedra
  2. Topological retrieval (no vector embeddings)
  3. Secondary memory attachment and abstract reorganization
  4. Dream cycle with quality tracking
  5. Self-organization
  6. Statistics and introspection
"""

import time

import numpy as np

from tetrahedron_memory.tetra_mesh import TetraMesh
from tetrahedron_memory.tetra_dream import TetraDreamCycle, DreamStore
from tetrahedron_memory.tetra_self_org import TetraSelfOrganizer


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    mesh = TetraMesh(time_lambda=0.001)

    # ── 1. Store memories ──────────────────────────────────────
    section("1. Storing Memories (Pure Geometric)")

    memories = [
        ("Machine learning is a subset of artificial intelligence", ["ai", "ml"], [0.0, 0.0, 0.0]),
        ("Deep learning uses neural networks with many layers", ["ai", "dl"], [0.1, 0.0, 0.0]),
        ("Python is the most popular language for data science", ["python", "data"], [0.2, 0.0, 0.0]),
        ("Natural language processing enables machines to understand text", ["ai", "nlp"], [0.3, 0.0, 0.0]),
        ("Computer vision allows machines to interpret images", ["ai", "cv"], [0.4, 0.0, 0.0]),
        ("Reinforcement learning trains agents through rewards", ["ai", "rl"], [0.5, 0.0, 0.0]),
        ("Transfer learning reuses knowledge across domains", ["ai", "tl"], [0.6, 0.0, 0.0]),
        ("Generative models create new data from learned distributions", ["ai", "gen"], [0.7, 0.0, 0.0]),
        ("Graph neural networks operate on graph-structured data", ["ai", "gnn"], [0.8, 0.0, 0.0]),
        ("Federated learning preserves privacy in distributed training", ["ai", "fl"], [0.9, 0.0, 0.0]),
        ("Bayesian methods quantify uncertainty in predictions", ["ai", "bayes"], [1.0, 0.0, 0.0]),
        ("Attention mechanisms power transformer architectures", ["ai", "nlp"], [1.1, 0.0, 0.0]),
    ]

    ids = []
    for content, labels, point in memories:
        tid = mesh.store(content, seed_point=np.array(point, dtype=np.float32), labels=labels, weight=1.0)
        ids.append(tid)
        print(f"  Stored: {content[:50]:50s} -> {tid[:8]}...")

    stats = mesh.get_statistics()
    print(f"\n  Mesh: {stats['total_tetrahedra']} tetrahedra, {stats['total_vertices']} vertices, {stats['total_faces']} faces")

    # ── 2. Topological Query ───────────────────────────────────
    section("2. Topological Query (No Vector Embeddings)")

    results = mesh.query_topological(np.array([0.5, 0.0, 0.0]), k=5, labels=["ai"])
    print("  Query: point=[0.5, 0, 0], labels=['ai'], k=5")
    for rank, (tid, score) in enumerate(results, 1):
        tetra = mesh.get_tetrahedron(tid)
        conn_type = "seed"
        print(f"    #{rank}: score={score:.3f} | {tetra.content[:55]} | labels={tetra.labels}")

    # ── 3. Topological Association ─────────────────────────────
    section("3. Topological Association")

    assoc = mesh.associate_topological(ids[0], max_depth=2)
    print(f"  Associations from '{memories[0][0][:40]}':")
    for tid, score, conn in assoc[:5]:
        t = mesh.get_tetrahedron(tid)
        print(f"    {conn:12s} score={score:.3f} | {t.content[:45]}")

    # ── 4. Secondary Memories + Abstract Reorganization ────────
    section("4. Secondary Memories & Abstract Reorganization")

    t_ai = ids[0]
    mesh.store_secondary(t_ai, "Supervised learning uses labeled data", labels=["ml", "supervised"], weight=1.2)
    mesh.store_secondary(t_ai, "Unsupervised learning finds hidden patterns", labels=["ml", "unsupervised"], weight=1.1)
    mesh.store_secondary(t_ai, "Feature engineering is crucial for ML models", labels=["ml", "features"], weight=0.9)

    tetra = mesh.get_tetrahedron(t_ai)
    print(f"  Attached {len(tetra.secondary_memories)} secondary memories to '{memories[0][0][:40]}'")

    reorg_stats = mesh.abstract_reorganize(min_density=2)
    print(f"  Reorganization: {reorg_stats['integrated_count']} integrated, {reorg_stats['cross_fusions']} cross-fusions")

    tetra = mesh.get_tetrahedron(t_ai)
    print(f"  After reorganization:")
    print(f"    Content: {tetra.content[:80]}")
    print(f"    Labels: {tetra.labels}")
    print(f"    Weight: {tetra.weight:.2f}")
    if "reorg_history" in tetra.metadata:
        for entry in tetra.metadata["reorg_history"]:
            print(f"    Themes extracted: {entry['themes_extracted']}")

    # ── 5. Dream Cycle ─────────────────────────────────────────
    section("5. Dream Cycle (Self-Emergent Synthesis)")

    def organizer_callback(m):
        org = TetraSelfOrganizer(m, max_iterations=3)
        org.run()

    dream = TetraDreamCycle(
        mesh,
        organizer=organizer_callback,
        walk_steps=10,
        dream_weight=0.5,
    )

    dream_stats = dream.trigger_now()
    print(f"  Dream phase: {dream_stats['phase']}")
    print(f"  Walk visited: {dream_stats['walk_visited']} tetrahedra")
    print(f"  Clusters found: {dream_stats['clusters_found']}")
    print(f"  Dreams created: {dream_stats['dreams_created']}")
    print(f"  Entropy: {dream_stats['entropy_before']:.4f} -> {dream_stats['entropy_after']:.4f} (delta={dream_stats['entropy_delta']:.4f})")

    dream_store = dream.get_dream_store()
    store_stats = dream_store.quality_stats()
    print(f"\n  Dream Store: {store_stats['count']} records")
    print(f"  Avg fusion quality: {store_stats['avg_quality']:.3f}")
    print(f"  Avg entropy delta: {store_stats['avg_entropy_delta']:.4f}")

    if dream_store.size > 0:
        print("\n  Recent dreams:")
        for rec in dream_store.get_recent(5):
            print(f"    [{rec.dream_id[:8]}] quality={rec.fusion_quality:.2f} | {rec.synthesis_content[:60]}")
            print(f"      sources={len(rec.source_tetra_ids)} labels={rec.labels}")

    # ── 6. Self-Organization ───────────────────────────────────
    section("6. Self-Organization")

    organizer = TetraSelfOrganizer(mesh, max_iterations=5)
    org_stats = organizer.run()
    print(f"  Iterations: {org_stats['iterations']}")
    print(f"  Converged: {org_stats['converged']}")
    if org_stats.get('convergence_reason'):
        print(f"  Reason: {org_stats['convergence_reason']}")
    print(f"  Actions: integrations={org_stats['integrations']}, merges={org_stats['merges']}, cave_growths={org_stats['cave_growths']}")

    # ── 7. Final Statistics ────────────────────────────────────
    section("7. Final Mesh Statistics")

    final = mesh.get_statistics()
    print(f"  Total tetrahedra:  {final['total_tetrahedra']}")
    print(f"  Total vertices:    {final['total_vertices']}")
    print(f"  Total faces:       {final['total_faces']}")
    print(f"  Boundary faces:    {final['boundary_faces']}")
    print(f"  Labels:            {final['total_labels']}")
    print(f"  Avg weight:        {final['avg_weight']:.3f}")
    print(f"  Total integrations:{final['total_integrations']}")
    print(f"  Total accesses:    {final['total_accesses']}")

    status = dream.get_status()
    print(f"\n  Dream cycles:      {status['cycle_count']}")
    print(f"  Dreams created:    {status['dreams_created']}")
    print(f"  Dreams reintegrated: {status['dreams_reintegrated']}")

    # ── 8. Traceability Demo ───────────────────────────────────
    section("8. Dream Traceability")

    if ids:
        traces = dream.get_dream_trace(ids[0])
        print(f"  Traces from '{memories[0][0][:40]}': {len(traces)} dreams")
        for trace in traces[:3]:
            print(f"    dream_id={trace['dream_id'][:8]} quality={trace['fusion_quality']:.2f}")
            print(f"      sources={trace['source_tetra_ids'][:3]}")

    print("\n" + "=" * 60)
    print("  TetraMem-XL Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

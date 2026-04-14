#!/usr/bin/env python3
"""
TetraMem-XL Complete Demo — Eternal Memory + Self-Emergent Closed Loop

Showcases the full cognitive cycle:
  1. Pure geometric memory storage on tetrahedra
  2. Topological retrieval (no vector embeddings)
  3. Secondary memory attachment and abstract reorganization
  4. Dream cycle with DreamProtocol (THINK → EXECUTE → REFLECT)
  5. Self-organization
  6. DreamStore traceability and quality statistics
  7. Distributed architecture
  8. Eternal audit — all memories preserved forever
"""

import time

import numpy as np

from tetrahedron_memory.tetra_mesh import TetraMesh
from tetrahedron_memory.tetra_dream import (
    TetraDreamCycle,
    DreamProtocol,
    DreamStore,
    fusion_quality_score,
)
from tetrahedron_memory.tetra_self_org import TetraSelfOrganizer
from tetrahedron_memory.tetra_distributed import TetraDistributedController


def section(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


def main():
    print("TetraMem-XL — Eternal Memory + Self-Emergent Closed Loop Demo")
    print("=" * 60)

    # ── 1. Store Memories ────────────────────────────────────
    section("1. Storing Memories (Pure Geometric, No Vectors)")

    mesh = TetraMesh(time_lambda=0.001)

    memories = [
        ("Machine learning is a subset of artificial intelligence", ["ai", "ml"], [0.0, 0.0, 0.0]),
        ("Deep learning uses neural networks with many layers", ["ai", "dl"], [0.1, 0.0, 0.0]),
        ("Python is the most popular language for data science", ["python", "data"], [0.2, 0.0, 0.0]),
        ("Natural language processing enables text understanding", ["ai", "nlp"], [0.3, 0.0, 0.0]),
        ("Computer vision allows machines to interpret images", ["ai", "cv"], [0.4, 0.0, 0.0]),
        ("Reinforcement learning trains agents through rewards", ["ai", "rl"], [0.5, 0.0, 0.0]),
        ("Transfer learning reuses knowledge across domains", ["ai", "tl"], [0.6, 0.0, 0.0]),
        ("Generative models create new data from distributions", ["ai", "gen"], [0.7, 0.0, 0.0]),
        ("Graph neural networks operate on graph structures", ["ai", "gnn"], [0.8, 0.0, 0.0]),
        ("Federated learning preserves privacy in training", ["ai", "fl"], [0.9, 0.0, 0.0]),
        ("Bayesian methods quantify prediction uncertainty", ["ai", "bayes"], [1.0, 0.0, 0.0]),
        ("Attention mechanisms power transformer architectures", ["ai", "nlp"], [1.1, 0.0, 0.0]),
    ]

    ids = []
    for content, labels, point in memories:
        tid = mesh.store(content, seed_point=np.array(point, dtype=np.float32), labels=labels)
        ids.append(tid)
        print(f"  Stored: {content[:50]:50s} [{', '.join(labels)}]")

    stats = mesh.get_statistics()
    print(f"\n  Mesh: {stats['total_tetrahedra']} tetrahedra, {stats['total_vertices']} vertices, {stats['total_faces']} faces")

    # ── 2. Pure Topological Query ────────────────────────────
    section("2. Pure Topological Query (BFS Along Faces/Edges/Vertices)")

    results = mesh.query_topological(np.array([0.5, 0.0, 0.0]), k=5, labels=["ai"])
    print("  Query: point=[0.5,0,0], labels=['ai'], k=5")
    for rank, (tid, score) in enumerate(results, 1):
        t = mesh.get_tetrahedron(tid)
        print(f"    #{rank}: score={score:.3f} | {t.content[:50]} | labels={t.labels}")

    # ── 3. Abstract Reorganization ──────────────────────────
    section("3. Abstract Reorganization (Secondary Memory Fusion)")

    t_ai = ids[0]
    mesh.store_secondary(t_ai, "Supervised learning uses labeled data", labels=["ml", "supervised"], weight=1.2)
    mesh.store_secondary(t_ai, "Unsupervised learning finds hidden patterns", labels=["ml", "unsupervised"], weight=1.1)
    mesh.store_secondary(t_ai, "Feature engineering is crucial for ML", labels=["ml", "features"], weight=0.9)

    t = mesh.get_tetrahedron(t_ai)
    print(f"  Attached {len(t.secondary_memories)} secondary memories to '{memories[0][0][:40]}'")

    reorg = mesh.abstract_reorganize(min_density=2)
    print(f"  Reorganization: {reorg['integrated_count']} integrated, {reorg['cross_fusions']} cross-fusions")

    t = mesh.get_tetrahedron(t_ai)
    print(f"  After reorg: content={t.content[:70]}")
    print(f"    Labels: {t.labels}")
    print(f"    Weight: {t.weight:.2f}")
    if "reorg_history" in t.metadata:
        for entry in t.metadata["reorg_history"]:
            print(f"    Themes: {entry['themes_extracted']}")

    # ── 4. Dream Cycle with DreamProtocol ────────────────────
    section("4. Dream Cycle (THINK -> EXECUTE -> REFLECT)")

    def organizer_cb(m):
        TetraSelfOrganizer(m, max_iterations=3).run()

    dream = TetraDreamCycle(
        mesh,
        organizer=organizer_cb,
        walk_steps=10,
        dream_weight=0.5,
    )

    ds = dream.trigger_now()
    print(f"  Phase: {ds['phase']}")
    print(f"  Walk visited: {ds['walk_visited']} | Clusters: {ds['clusters_found']}")
    print(f"  Dreams created: {ds['dreams_created']}")
    print(f"  Entropy: {ds['entropy_before']:.4f} -> {ds['entropy_after']:.4f} (delta={ds['entropy_delta']:.4f})")

    dream_store = dream.get_dream_store()
    store_stats = dream_store.quality_stats()
    print(f"\n  DreamStore: {store_stats['count']} records")
    print(f"  Avg quality: {store_stats['avg_quality']:.3f}")

    if dream_store.size > 0:
        print("\n  Recent dreams:")
        for rec in dream_store.get_recent(5):
            print(f"    [{rec.dream_id[:8]}] q={rec.fusion_quality:.2f} | {rec.synthesis_content[:55]}")

    # ── 5. DreamProtocol (explicit three-phase) ──────────────
    section("5. DreamProtocol: THINK -> EXECUTE -> REFLECT")

    protocol = DreamProtocol(quality_threshold=0.3)
    inputs = [
        {"labels": ["ai", "ml"], "weight": 2.0, "integration_count": 3,
         "content": "deep learning fundamentals", "centroid": [0.0, 0.0, 0.0]},
        {"labels": ["ai", "nlp"], "weight": 1.5, "integration_count": 1,
         "content": "language model architectures", "centroid": [1.0, 0.0, 0.0]},
        {"labels": ["ml", "stats"], "weight": 1.0, "integration_count": 5,
         "content": "statistical learning theory", "centroid": [0.5, 1.0, 0.0]},
    ]

    result = protocol.run(inputs)
    print(f"  THINK strategy: {result['analysis']['strategy']}")
    print(f"  EXECUTE content: {result['content']}")
    print(f"  REFLECT quality: {result['quality']:.3f} | accepted: {result['accepted']}")

    proto_stats = protocol.get_statistics()
    print(f"  Protocol stats: {proto_stats}")

    # ── 6. Self-Organization ────────────────────────────────
    section("6. Self-Organization (PH Geometric Surgery)")

    organizer = TetraSelfOrganizer(mesh, max_iterations=5)
    org_stats = organizer.run()
    print(f"  Iterations: {org_stats['iterations']}")
    print(f"  Converged: {org_stats['converged']}")
    if org_stats.get("convergence_reason"):
        print(f"  Reason: {org_stats['convergence_reason']}")
    print(f"  Actions: integrations={org_stats['integrations']}, merges={org_stats['merges']}, cave_growths={org_stats['cave_growths']}")

    # ── 7. Distributed Architecture ─────────────────────────
    section("7. Distributed Architecture")

    ctrl = TetraDistributedController(num_buckets=2, max_tetra_per_bucket=500, use_ray=False)
    ctrl.initialize()

    for i in range(30):
        ctrl.store(
            "distributed memory " + str(i),
            np.array([float(i) * 0.05, 0.0, 0.0]),
            labels=["topic", "pipeline"],
            weight=1.0 + float(i % 5) * 0.2,
        )

    dist_dream = ctrl.run_dream_cycle(walk_steps=8)
    dist_org = ctrl.run_self_organization(max_iterations=3)

    print(f"  Dream: {dist_dream['phase']}, created={dist_dream['local_dreams_created']}")
    print(f"  Self-org: {dist_org['phase']}, actions={dist_org['total_actions']}")

    dist_stats = ctrl.get_statistics()
    print(f"  Total: {dist_stats['total_tetrahedra']} tetrahedra in {dist_stats['total_buckets']} buckets")
    print(f"  Ghost cells: {dist_stats['total_ghost_cells']}")

    # ── 8. Eternal Audit ────────────────────────────────────
    section("8. Eternal Audit — All Memories Preserved")

    final = mesh.get_statistics()
    print(f"  Tetrahedra: {final['total_tetrahedra']} (all preserved, none deleted)")
    print(f"  Total integrations: {final['total_integrations']}")
    print(f"  Total accesses: {final['total_accesses']}")
    print(f"  Labels indexed: {final['total_labels']}")

    dream_status = dream.get_status()
    print(f"\n  Dream cycles run: {dream_status['cycle_count']}")
    print(f"  Dreams created: {dream_status['dreams_created']}")
    print(f"  Dreams reintegrated: {dream_status['dreams_reintegrated']}")

    status = dream.get_status()
    if "dream_store" in status:
        ds = status["dream_store"]
        print(f"  DreamStore: {ds['count']} records, avg quality={ds['avg_quality']:.3f}")

    print("\n" + "=" * 60)
    print("  TetraMem-XL Demo Complete")
    print("  Eternal Memory + Self-Emergent Closed Loop")
    print("=" * 60)


if __name__ == "__main__":
    main()

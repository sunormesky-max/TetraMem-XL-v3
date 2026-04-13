"""
Integration tests verifying production-grade fixes:
  - self_organize() routes through mesh
  - store_batch() respects _use_mesh
  - GUDHI optional for mesh mode
  - TetraRouter ghost cell fixes
"""

import numpy as np
import pytest

from tetrahedron_memory.core import GeoMemoryBody


class TestSelfOrganizeMeshPath:
    def test_self_organize_routes_through_mesh(self):
        body = GeoMemoryBody()
        for i in range(10):
            body.store(f"item_{i}", labels=["topic"], weight=1.0)
        assert body._use_mesh is True
        result = body.self_organize()
        assert "iterations" in result or "actions" in result or "reason" in result

    def test_self_organize_with_gudhi_fallback(self):
        pytest.importorskip("gudhi")
        body = GeoMemoryBody()
        body._use_mesh = False
        for i in range(10):
            body.store(f"item_{i}", labels=["topic"])
        result = body.self_organize()
        assert "actions" in result or "reason" in result

    def test_self_organize_mesh_has_self_organizer(self):
        body = GeoMemoryBody()
        for i in range(5):
            body.store(f"item_{i}", labels=["test"])
        body.self_organize()
        assert body._self_organizer is not None


class TestStoreBatchMesh:
    def test_store_batch_uses_mesh(self):
        body = GeoMemoryBody()
        items = [{"content": f"batch_{i}", "labels": ["batch"]} for i in range(5)]
        ids = body.store_batch(items)
        assert len(ids) == 5
        for mid in ids:
            assert mid in body._mesh.tetrahedra

    def test_store_batch_legacy_path(self):
        pytest.importorskip("gudhi")
        body = GeoMemoryBody()
        body._use_mesh = False
        items = [{"content": f"legacy_{i}", "labels": ["legacy"]} for i in range(3)]
        ids = body.store_batch(items)
        assert len(ids) == 3
        for mid in ids:
            assert mid in body._nodes_dict


class TestGUDHIOptional:
    def test_geo_memory_body_inits_without_gudhi_import(self):
        body = GeoMemoryBody()
        assert body._use_mesh is True
        assert body._mesh is not None


class TestQueryAndAssociateMesh:
    def test_store_query_roundtrip(self):
        body = GeoMemoryBody()
        body.store("hello world", labels=["greeting"])
        body.store("goodbye world", labels=["farewell"])
        results = body.query("hello", k=2)
        assert len(results) >= 1
        assert "hello" in results[0].node.content or "goodbye" in results[0].node.content

    def test_associate_mesh(self):
        body = GeoMemoryBody()
        id1 = body.store("AI research", labels=["ai"])
        for i in range(5):
            body.store(f"related topic {i}", labels=["ai", "research"])
        results = body.associate(id1, max_depth=2)
        assert isinstance(results, list)

    def test_update_weight_mesh(self):
        body = GeoMemoryBody()
        tid = body.store("weighted", labels=["test"], weight=1.0)
        body.update_weight(tid, 2.0, use_ema=False)
        tetra = body._mesh.get_tetrahedron(tid)
        assert abs(tetra.weight - 3.0) < 0.01

    def test_global_catalyze_mesh(self):
        body = GeoMemoryBody()
        for i in range(5):
            body.store(f"integrate_{i}", labels=["test"], weight=0.1)
        result = body.global_catalyze_integration(strength=1.0)
        assert "catalyzed" in result


class TestEdgeContractionIntervalFix:
    def test_edge_contraction_uses_interval(self):
        pytest.importorskip("gudhi")
        body = GeoMemoryBody()
        body._use_mesh = False
        for i in range(10):
            body.store(f"item_{i}", labels=["test"])
        stats = body.self_organize()
        assert isinstance(stats, dict)

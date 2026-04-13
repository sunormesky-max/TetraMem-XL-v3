"""
Tests for TetraMesh core module.
"""

import numpy as np
import pytest

from tetrahedron_memory.tetra_mesh import FaceRecord, MemoryTetrahedron, TetraMesh


class TestMemoryTetrahedron:
    def test_creation(self):
        t = MemoryTetrahedron(
            id="test",
            content="hello",
            vertex_indices=(0, 1, 2, 3),
            centroid=np.array([0.0, 0.0, 0.0]),
            labels=["a"],
            weight=1.5,
            creation_time=1000.0,
            last_access_time=1000.0,
            init_weight=1.5,
        )
        assert t.id == "test"
        assert t.content == "hello"
        assert t.weight == 1.5

    def test_filtration(self):
        t = MemoryTetrahedron(
            id="f",
            content="x",
            vertex_indices=(0, 1, 2, 3),
            centroid=np.zeros(3),
            creation_time=1000.0,
            init_weight=1.0,
            _spatial_alpha=0.5,
        )
        fil = t.filtration(time_lambda=0.001)
        assert fil >= 0.5

    def test_touch(self):
        t = MemoryTetrahedron(
            id="t",
            content="x",
            vertex_indices=(0, 1, 2, 3),
            centroid=np.zeros(3),
            last_access_time=0.0,
        )
        old_time = t.last_access_time
        t.touch()
        assert t.last_access_time >= old_time


class TestFaceRecord:
    def test_is_boundary_single(self):
        fr = FaceRecord(vertex_indices=(0, 1, 2), tetrahedra={"a"})
        assert fr.is_boundary is True

    def test_is_boundary_shared(self):
        fr = FaceRecord(vertex_indices=(0, 1, 2), tetrahedra={"a", "b"})
        assert fr.is_boundary is False

    def test_is_boundary_empty(self):
        fr = FaceRecord(vertex_indices=(0, 1, 2))
        assert fr.is_boundary is True


class TestTetraMeshStore:
    def test_store_first_creates_seed(self):
        mesh = TetraMesh()
        tid = mesh.store("hello", seed_point=np.array([1.0, 0.0, 0.0]))
        assert tid is not None
        assert len(mesh.tetrahedra) == 1
        assert len(mesh.vertices) == 4

    def test_store_multiple_grows_mesh(self):
        mesh = TetraMesh()
        ids = []
        for i in range(10):
            pt = np.array([float(i) * 0.1, 0.0, 0.0])
            ids.append(mesh.store(f"item_{i}", seed_point=pt))
        assert len(set(ids)) == 10
        assert len(mesh.tetrahedra) == 10

    def test_store_with_labels(self):
        mesh = TetraMesh()
        tid = mesh.store("text", seed_point=np.zeros(3), labels=["ai", "ml"])
        tetra = mesh.get_tetrahedron(tid)
        assert tetra is not None
        assert "ai" in tetra.labels
        assert "ml" in tetra.labels

    def test_store_with_metadata(self):
        mesh = TetraMesh()
        tid = mesh.store("text", seed_point=np.zeros(3), metadata={"key": "val"})
        tetra = mesh.get_tetrahedron(tid)
        assert tetra.metadata["key"] == "val"

    def test_label_index(self):
        mesh = TetraMesh()
        mesh.store("a", seed_point=np.array([0.0, 0.0, 0.0]), labels=["x"])
        mesh.store("b", seed_point=np.array([1.0, 0.0, 0.0]), labels=["x", "y"])
        assert len(mesh.label_index["x"]) == 2
        assert len(mesh.label_index["y"]) == 1


class TestTetraMeshQuery:
    def test_query_empty_mesh(self):
        mesh = TetraMesh()
        results = mesh.query_topological(np.zeros(3))
        assert results == []

    def test_query_single_tetra(self):
        mesh = TetraMesh()
        mesh.store("hello", seed_point=np.array([0.0, 0.0, 0.0]))
        results = mesh.query_topological(np.array([0.0, 0.0, 0.0]), k=5)
        assert len(results) >= 1
        assert results[0][0] is not None

    def test_query_returns_sorted_by_score(self):
        mesh = TetraMesh()
        for i in range(20):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.05, 0.0, 0.0]))
        results = mesh.query_topological(np.array([0.0, 0.0, 0.0]), k=5)
        assert len(results) <= 5
        scores = [s for _, s in results]
        assert scores == sorted(scores)


class TestTetraMeshAssociate:
    def test_associate_nonexistent(self):
        mesh = TetraMesh()
        result = mesh.associate_topological("nonexistent")
        assert result == []

    def test_associate_single_tetra(self):
        mesh = TetraMesh()
        tid = mesh.store("only", seed_point=np.zeros(3))
        result = mesh.associate_topological(tid)
        assert isinstance(result, list)

    def test_associate_connected_mesh(self):
        mesh = TetraMesh()
        ids = []
        for i in range(5):
            ids.append(mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.1, 0.0, 0.0])))
        result = mesh.associate_topological(ids[0], max_depth=2)
        assert len(result) > 0
        assoc_ids = {tid for tid, _, _ in result}
        assert ids[0] not in assoc_ids


class TestTetraMeshEdgeContraction:
    def test_edge_contraction_nonexistent(self):
        mesh = TetraMesh()
        result = mesh.edge_contraction("a", "b")
        assert result is None

    def test_edge_contraction_works(self):
        mesh = TetraMesh()
        t1 = mesh.store("a", seed_point=np.array([0.0, 0.0, 0.0]))
        t2 = mesh.store("b", seed_point=np.array([0.1, 0.0, 0.0]))
        shared = set(mesh.get_tetrahedron(t1).vertex_indices) & set(
            mesh.get_tetrahedron(t2).vertex_indices
        )
        if len(shared) >= 2:
            new_id = mesh.edge_contraction(t1, t2)
            assert new_id is not None
            assert t1 not in mesh.tetrahedra
            assert t2 not in mesh.tetrahedra
            assert new_id in mesh.tetrahedra


class TestTetraMeshIntegration:
    def test_catalyze_integration_batch(self):
        mesh = TetraMesh()
        ids = []
        for i in range(5):
            ids.append(mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.1, 0.0, 0.0])))
        result = mesh.catalyze_integration_batch(ids, strength=1.0)
        assert result["catalyzed"] == 5
        for tid in ids:
            tetra = mesh.get_tetrahedron(tid)
            assert tetra.integration_count >= 1


class TestTetraMeshStatistics:
    def test_statistics(self):
        mesh = TetraMesh()
        mesh.store("a", seed_point=np.zeros(3), labels=["x"])
        stats = mesh.get_statistics()
        assert stats["total_tetrahedra"] == 1
        assert stats["total_vertices"] == 4
        assert stats["total_faces"] == 4
        assert stats["boundary_faces"] >= 0
        assert "avg_weight" in stats

    def test_boundary_faces(self):
        mesh = TetraMesh()
        mesh.store("a", seed_point=np.zeros(3))
        assert len(mesh.boundary_faces) == 4


class TestTetraMeshComputePH:
    def test_compute_ph_insufficient_vertices(self):
        mesh = TetraMesh()
        result = mesh.compute_ph()
        assert result is None

    def test_compute_ph_with_data(self):
        pytest.importorskip("gudhi")
        mesh = TetraMesh()
        for i in range(5):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.5, 0.0, 0.0]))
        st = mesh.compute_ph()
        if st is not None:
            h0 = st.persistence_intervals_in_dimension(0)
            assert len(h0) > 0


class TestTetraMeshSpatialIndex:
    def test_centroid_index_rebuild(self):
        mesh = TetraMesh()
        for i in range(60):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.01, 0.0, 0.0]))
        mesh._nearest_tetrahedron(np.zeros(3))
        assert mesh._centroid_matrix is not None
        assert len(mesh._centroid_ids) == 60

    def test_nearest_uses_index_for_large_mesh(self):
        mesh = TetraMesh()
        for i in range(60):
            mesh.store(f"item_{i}", seed_point=np.array([float(i), 0.0, 0.0]))
        query = np.array([30.0, 0.0, 0.0])
        tid, dist = mesh._nearest_tetrahedron(query)
        assert tid is not None
        assert dist < 1.0

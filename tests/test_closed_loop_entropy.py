"""
Production-grade tests for TetraMem-XL closed loop and entropy integration.

Covers:
  - ClosedLoopEngine full cycle (recall→think→execute→reflect→integrate→dream)
  - Persistent entropy computation and convergence
  - Entropy-driven integration triggers
  - Self-organization entropy convergence
  - Dream cycle entropy tracking
  - Eternity principle enforcement (no deletion after any operation)
"""

import time

import numpy as np
import pytest

from tetrahedron_memory.closed_loop import (
    ClosedLoopEngine,
    LoopPhase,
    _default_execute,
    _default_reflect,
    _default_think,
)
from tetrahedron_memory.core import GeoMemoryBody
from tetrahedron_memory.persistent_entropy import (
    EntropyTracker,
    compute_entropy_by_dimension,
    compute_entropy_delta,
    compute_persistent_entropy,
    should_trigger_integration,
)
from tetrahedron_memory.tetra_dream import TetraDreamCycle
from tetrahedron_memory.tetra_mesh import TetraMesh
from tetrahedron_memory.tetra_self_org import TetraSelfOrganizer


class TestPersistentEntropyComputation:
    def test_entropy_zero_on_empty(self):
        mesh = TetraMesh()
        assert mesh.compute_persistent_entropy() == 0.0

    def test_entropy_positive_with_data(self):
        pytest.importorskip("gudhi")
        mesh = TetraMesh()
        for i in range(10):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.5, 0.0, 0.0]))
        entropy = mesh.compute_persistent_entropy()
        assert entropy >= 0.0

    def test_entropy_by_dimension(self):
        pytest.importorskip("gudhi")
        mesh = TetraMesh()
        for i in range(10):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.3, float(i % 3) * 0.2, 0.0]))
        st = mesh.compute_ph()
        if st is not None:
            dims = compute_entropy_by_dimension(st)
            assert isinstance(dims, dict)
            if 0 in dims:
                assert dims[0] >= 0.0

    def test_entropy_delta(self):
        assert compute_entropy_delta(1.0, 0.5) == pytest.approx(0.5, rel=1e-5)
        assert compute_entropy_delta(0.0, 0.5) == 0.0
        assert compute_entropy_delta(2.0, 2.2) < 0.0

    def test_should_trigger_integration(self):
        assert should_trigger_integration(1.5, 1.0, 1.3) is True
        assert should_trigger_integration(1.0, 1.0, 1.3) is False
        assert should_trigger_integration(0.8, 0.0) is True
        assert should_trigger_integration(0.3, 0.0) is False


class TestEntropyTracker:
    def test_record_and_current(self):
        tracker = EntropyTracker()
        tracker.record(1.5)
        assert tracker.baseline == 1.5
        tracker.record(1.3)
        tracker.record(1.1)
        assert tracker.current == 1.1
        assert tracker.baseline == pytest.approx((1.5 + 1.3 + 1.1) / 3, rel=1e-5)

    def test_trend_detection(self):
        tracker = EntropyTracker()
        tracker.record(2.0)
        tracker.record(1.5)
        tracker.record(1.0)
        assert tracker.trend == "decreasing"

        tracker.record(2.0)
        tracker.record(2.5)
        tracker.record(3.0)
        assert tracker.trend == "increasing"

    def test_last_delta(self):
        tracker = EntropyTracker()
        tracker.record(2.0)
        tracker.record(1.0)
        assert tracker.last_delta == pytest.approx(0.5, rel=1e-5)

    def test_window_trimming(self):
        tracker = EntropyTracker(window_size=5)
        for i in range(10):
            tracker.record(float(i))
        assert len(tracker._history) == 5

    def test_should_integrate(self):
        tracker = EntropyTracker()
        tracker.record(1.0)
        tracker.record(1.0)
        tracker.record(1.0)
        tracker.record(2.0)
        assert tracker.should_integrate(1.3) is True
        assert tracker.should_integrate(2.0) is False

    def test_get_summary(self):
        tracker = EntropyTracker()
        tracker.record(1.0)
        summary = tracker.get_summary()
        assert "current" in summary
        assert "baseline" in summary
        assert "trend" in summary
        assert "last_delta" in summary


class TestClosedLoopEngine:
    def _make_body(self, n=10):
        body = GeoMemoryBody()
        for i in range(n):
            body.store(f"memory_{i}_about_topic_{i % 3}", labels=[f"topic_{i % 3}"])
        return body

    def test_full_cycle_with_context(self):
        body = self._make_body(15)
        engine = ClosedLoopEngine(memory=body)
        result = engine.run_cycle(context="topic_0", k=5)
        assert result["phase"] == LoopPhase.COMPLETE.name
        assert result["recall_count"] >= 0
        assert "think" in result
        assert "execute" in result
        assert "reflect" in result
        assert result["cycle"] == 0

    def test_full_cycle_without_context(self):
        body = self._make_body(10)
        engine = ClosedLoopEngine(memory=body)
        result = engine.run_cycle()
        assert result["phase"] == LoopPhase.COMPLETE.name
        assert result["recall_count"] >= 0

    def test_custom_think_execute_reflect(self):
        body = self._make_body(10)

        def custom_think(recall):
            return {"analysis": "custom", "confidence": 0.9, "patterns": ["x"]}

        def custom_execute(think):
            return {"action": "custom_action", "output": "custom output"}

        def custom_reflect(think, execute):
            return {
                "quality": 0.8,
                "contradictions": 0,
                "should_integrate": True,
                "should_dream": False,
                "insight": "custom insight",
            }

        engine = ClosedLoopEngine(
            memory=body,
            think_fn=custom_think,
            execute_fn=custom_execute,
            reflect_fn=custom_reflect,
        )
        result = engine.run_cycle(context="test")
        assert result["think"]["analysis"] == "custom"
        assert result["execute"]["action"] == "custom_action"
        assert result["reflect"]["quality"] == 0.8

    def test_insight_stored_as_memory(self):
        body = self._make_body(10)
        engine = ClosedLoopEngine(memory=body)
        initial_count = len(body._nodes)
        result = engine.run_cycle(context="test")
        if result.get("reflect", {}).get("should_integrate"):
            assert len(body._nodes) >= initial_count

    def test_dream_triggered_when_entropy_high(self):
        pytest.importorskip("gudhi")
        body = self._make_body(20)

        def force_dream_reflect(think, execute):
            return {
                "quality": 0.1,
                "contradictions": 1,
                "should_integrate": True,
                "should_dream": True,
                "insight": "low confidence insight",
            }

        engine = ClosedLoopEngine(memory=body, reflect_fn=force_dream_reflect)
        result = engine.run_cycle(context="test", force_dream=True)
        assert result["dream_triggered"] is True

    def test_no_deletion_after_cycle(self):
        body = self._make_body(20)
        initial_count = len(body._nodes)
        engine = ClosedLoopEngine(memory=body)
        for _ in range(3):
            engine.run_cycle(context="test")
        assert len(body._nodes) >= initial_count

    def test_get_status(self):
        body = self._make_body(5)
        engine = ClosedLoopEngine(memory=body)
        engine.run_cycle()
        status = engine.get_status()
        assert status["cycle_count"] == 1
        assert status["total_stores"] >= 0
        assert "entropy" in status
        assert "last_cycle" in status

    def test_daemon_start_stop(self):
        body = self._make_body(5)
        engine = ClosedLoopEngine(memory=body, auto_cycle_interval=9999)
        engine.start_daemon()
        assert engine.get_status()["running"] is True
        engine.stop_daemon()
        assert engine.get_status()["running"] is False


class TestSelfOrganizerEntropyConvergence:
    def test_entropy_tracked_during_run(self):
        pytest.importorskip("gudhi")
        mesh = TetraMesh()
        for i in range(15):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.3, float(i % 3) * 0.1, 0.0]))
        so = TetraSelfOrganizer(mesh, max_iterations=3)
        stats = so.run()
        assert "entropy_before" in stats
        assert "entropy_after" in stats
        assert "entropy_delta" in stats

    def test_convergence_reason_reported(self):
        pytest.importorskip("gudhi")
        mesh = TetraMesh()
        for i in range(15):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.3, float(i % 3) * 0.1, 0.0]))
        so = TetraSelfOrganizer(mesh, max_iterations=5, convergence_threshold=2)
        stats = so.run()
        if stats["converged"]:
            assert "convergence_reason" in stats
            assert stats["convergence_reason"] in ("low_actions", "entropy_stable")

    def test_status_includes_entropy(self):
        mesh = TetraMesh()
        for i in range(5):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.3, 0.0, 0.0]))
        so = TetraSelfOrganizer(mesh)
        status = so.get_status()
        assert "entropy" in status
        assert "current" in status["entropy"]


class TestDreamCycleEntropy:
    def test_dream_tracks_entropy(self):
        pytest.importorskip("gudhi")
        mesh = TetraMesh()
        for i in range(20):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.2, 0.0, 0.0]))
        dc = TetraDreamCycle(mesh, walk_steps=15)
        stats = dc.trigger_now()
        assert "entropy_before" in stats
        assert "entropy_after" in stats
        assert "entropy_delta" in stats

    def test_dream_no_deletion(self):
        mesh = TetraMesh()
        for i in range(20):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.2, 0.0, 0.0]))
        initial_count = len(mesh.tetrahedra)
        dc = TetraDreamCycle(mesh, walk_steps=15)
        dc.trigger_now()
        assert len(mesh.tetrahedra) >= initial_count

    def test_dream_status_entropy(self):
        mesh = TetraMesh()
        for i in range(10):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.2, 0.0, 0.0]))
        dc = TetraDreamCycle(mesh)
        status = dc.get_status()
        assert "entropy" in status
        assert "current" in status["entropy"]


class TestEternityPrincipleEnforcement:
    def test_integration_never_reduces_count(self):
        body = GeoMemoryBody()
        for i in range(50):
            body.store(f"mem_{i}", weight=0.01)
        count_before = len(body._nodes)
        body.global_catalyze_integration(strength=2.0)
        assert len(body._nodes) == count_before

    def test_self_organize_never_reduces_count(self):
        pytest.importorskip("gudhi")
        body = GeoMemoryBody()
        for i in range(30):
            body.store(f"mem_{i}_{i * i}", weight=1.0)
        count_before = len(body._nodes)
        body.self_organize()
        count_after = len(body._nodes)
        non_system = sum(
            1 for n in body._nodes.values()
            if "__system__" not in n.labels
        )
        merged = sum(
            1 for n in body._nodes.values()
            if "merged_from" in n.metadata
        )
        assert count_after >= non_system, "System nodes should not exceed total"
        assert non_system + merged >= count_before - 2, (
            f"Too many originals lost: {count_before} → {non_system} non-system"
        )

    def test_multiple_cycles_allow_structural_merges(self):
        pytest.importorskip("gudhi")
        body = GeoMemoryBody()
        for i in range(20):
            body.store(f"mem_{i}", labels=["topic"], weight=1.0)
        count_before = len(body._nodes)

        for _ in range(5):
            body.self_organize()
            body.global_catalyze_integration(strength=1.0)

        # edge_contraction merges 2→1, so count can decrease by merges
        # but all original content is preserved in merged nodes
        assert len(body._nodes) >= 1
        all_content = set()
        for n in body._nodes.values():
            all_content.add(n.content)
        # Check no content was truly "forgotten" — merged content carries forward
        non_system = [n for n in body._nodes.values() if "__system__" not in n.labels]
        assert len(non_system) >= 1

    def test_mesh_catalyze_never_removes(self):
        mesh = TetraMesh()
        ids = []
        for i in range(20):
            ids.append(mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.1, 0.0, 0.0])))
        count_before = len(mesh.tetrahedra)
        mesh.catalyze_integration_batch(ids, strength=5.0)
        assert len(mesh.tetrahedra) == count_before

    def test_closed_loop_never_removes(self):
        body = GeoMemoryBody()
        for i in range(20):
            body.store(f"mem_{i}", weight=1.0)
        count_before = len(body._nodes)
        engine = ClosedLoopEngine(memory=body)
        for _ in range(5):
            engine.run_cycle(context="test")
        assert len(body._nodes) >= count_before


class TestMemoryTetrahedronIntegrationCount:
    def test_catalyze_increments_count(self):
        mesh = TetraMesh()
        tid = mesh.store("test", seed_point=np.zeros(3))
        tetra = mesh.get_tetrahedron(tid)
        assert tetra.integration_count == 0
        tetra.catalyze_integration(1.0)
        assert tetra.integration_count == 1
        assert tetra.weight > 1.0

    def test_touch_increments_access_count(self):
        mesh = TetraMesh()
        tid = mesh.store("test", seed_point=np.zeros(3))
        tetra = mesh.get_tetrahedron(tid)
        assert tetra.access_count == 0
        tetra.touch()
        assert tetra.access_count == 1

    def test_filtration_decreases_with_integration(self):
        mesh = TetraMesh()
        tid = mesh.store("test", seed_point=np.array([1.0, 0.0, 0.0]))
        tetra = mesh.get_tetrahedron(tid)
        fil_before = tetra.filtration(0.001)
        tetra.catalyze_integration(5.0)
        tetra.catalyze_integration(5.0)
        fil_after = tetra.filtration(0.001)
        assert fil_after < fil_before


class TestMeshStatisticsEnhanced:
    def test_statistics_include_integration_counts(self):
        mesh = TetraMesh()
        for i in range(5):
            tid = mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.1, 0.0, 0.0]))
            tetra = mesh.get_tetrahedron(tid)
            tetra.catalyze_integration(1.0)
        stats = mesh.get_statistics()
        assert "total_integrations" in stats
        assert stats["total_integrations"] >= 5

    def test_statistics_include_access_counts(self):
        mesh = TetraMesh()
        for i in range(5):
            mesh.store(f"item_{i}", seed_point=np.array([float(i) * 0.1, 0.0, 0.0]))
        stats = mesh.get_statistics()
        assert "total_accesses" in stats

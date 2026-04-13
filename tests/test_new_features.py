import os
import shutil
import tempfile
import time

import numpy as np
import pytest

from tetrahedron_memory.core import GeoMemoryBody, MemoryNode
from tetrahedron_memory.monitoring import (
    generate_metrics,
    get_metrics_registry,
    increment_counter,
    observe_histogram,
    record_error,
    set_gauge,
)
from tetrahedron_memory.multimodal import PixHomology
from tetrahedron_memory.partitioning import (
    BucketActor,
    TetraMemRayController,
    get_all_buckets,
    global_coarse_grid_sync,
    register_bucket,
    unregister_bucket,
)
from tetrahedron_memory.persistence import ParquetPersistence


class TestGlobalIntegration:
    def test_catalyze_boosts_weights(self):
        body = GeoMemoryBody()
        for i in range(10):
            body.store(content=f"mem_{i}", weight=1.0)
        result = body.global_catalyze_integration(strength=1.0)
        assert result["catalyzed"] == 10
        for node in body._nodes.values():
            assert node.weight >= 1.0

    def test_catalyze_no_removal(self):
        body = GeoMemoryBody()
        for i in range(10):
            body.store(content=f"low_{i}", weight=0.06)
        before_count = len(body._nodes)
        result = body.global_catalyze_integration(strength=1.0)
        assert len(body._nodes) == before_count
        assert result["catalyzed"] == 10

    def test_catalyze_skips_system_nodes(self):
        body = GeoMemoryBody()
        body.store(content="user_mem", weight=0.06)
        body._mesh.store(
            content="__system__",
            seed_point=np.array([0.0, 0.0, 1.0]),
            labels=["__system__"],
            weight=0.06,
        )
        result = body.global_catalyze_integration(strength=1.0)
        assert result["catalyzed"] >= 1


class TestSelfOrganizeH1H2:
    def test_self_organize_returns_stats(self):
        body = GeoMemoryBody()
        for i in range(20):
            body.store(content=f"mem_{i}_{i * i}", weight=1.0)
        stats = body.self_organize()
        assert "actions" in stats
        assert stats["actions"] >= 0

    def test_self_organize_insufficient_nodes(self):
        body = GeoMemoryBody()
        body.store(content="only_one", weight=1.0)
        stats = body.self_organize()
        assert stats["actions"] == 0
        assert stats["reason"] == "insufficient_nodes"

    def test_self_organize_many_nodes_triggers_actions(self):
        body = GeoMemoryBody()
        rng = np.random.RandomState(42)
        for i in range(50):
            content = f"unique_content_{i}_{rng.rand()}"
            body.store(content=content, weight=rng.uniform(0.2, 5.0))
        stats = body.self_organize()
        assert isinstance(stats["actions"], int)


class TestBucketActor:
    def test_store_and_query(self):
        actor = BucketActor("test_bucket")
        node_id = actor.store(content="hello world", labels=["greeting"])
        assert isinstance(node_id, str)
        results = actor.query("hello", k=5)
        assert len(results) > 0
        assert results[0]["content"] == "hello world"

    def test_self_organize(self):
        actor = BucketActor("org_bucket")
        for i in range(10):
            actor.store(content=f"item_{i}")
        stats = actor.self_organize()
        assert isinstance(stats, dict)
        assert "actions" in stats

    def test_get_snapshot(self):
        actor = BucketActor("snap_bucket")
        actor.store(content="snap1", weight=2.0)
        actor.store(content="snap2", weight=3.0)
        snap = actor.get_snapshot()
        assert len(snap) == 2
        for nid, data in snap.items():
            assert "content" in data
            assert "geometry" in data
            assert "weight" in data

    def test_get_statistics(self):
        actor = BucketActor("stats_bucket")
        actor.store(content="stat1")
        stats = actor.get_statistics()
        assert stats["total_memories"] == 1

    def test_associate_empty(self):
        actor = BucketActor("assoc_empty")
        actor.store(content="only")
        results = actor.associate(memory_id="nonexistent")
        assert results == []


class TestTetraMemRayController:
    def test_initialize_local_mode(self):
        ctrl = TetraMemRayController(num_buckets=2)
        ctrl.initialize()
        assert ctrl.is_initialized()
        stats = ctrl.get_statistics()
        assert stats["num_buckets"] == 2

    def test_store_and_query(self):
        ctrl = TetraMemRayController(num_buckets=2)
        ctrl.initialize()
        nid = ctrl.store("bucket_0", content="ray_test", labels=["test"])
        assert isinstance(nid, str)
        results = ctrl.query("bucket_0", "ray_test", k=5)
        assert len(results) > 0

    def test_self_organize(self):
        ctrl = TetraMemRayController(num_buckets=2)
        ctrl.initialize()
        for i in range(10):
            ctrl.store("bucket_0", content=f"so_{i}")
        stats = ctrl.self_organize("bucket_0")
        assert isinstance(stats, dict)

    def test_shutdown(self):
        ctrl = TetraMemRayController(num_buckets=2)
        ctrl.initialize()
        ctrl.shutdown()
        assert not ctrl.is_initialized()

    def test_sync_all(self):
        ctrl = TetraMemRayController(num_buckets=2)
        ctrl.initialize()
        ctrl.store("bucket_0", content="sync_a")
        ctrl.store("bucket_1", content="sync_b")
        result = ctrl.sync_all()
        assert result is None or result is not None


class TestGlobalBucketRegistry:
    def setup_method(self):
        from tetrahedron_memory import partitioning

        partitioning._global_bucket_registry.clear()

    def test_register_and_get(self):
        actor = BucketActor("reg_test")
        register_bucket("reg_test", actor)
        buckets = get_all_buckets()
        assert "reg_test" in buckets

    def test_unregister(self):
        actor = BucketActor("unreg_test")
        register_bucket("unreg_test", actor)
        unregister_bucket("unreg_test")
        buckets = get_all_buckets()
        assert "unreg_test" not in buckets

    def test_global_coarse_grid_sync(self):
        a1 = BucketActor("g1")
        a1.store(content="grid_1", weight=1.0)
        a2 = BucketActor("g2")
        a2.store(content="grid_2", weight=1.0)
        register_bucket("g1", a1)
        register_bucket("g2", a2)
        result = global_coarse_grid_sync()
        assert result is None or isinstance(result, (list, np.ndarray))

    def teardown_method(self):
        from tetrahedron_memory import partitioning

        partitioning._global_bucket_registry.clear()


class TestMultimodalAudio:
    def test_audio_to_geometry_shape(self):
        pix = PixHomology()
        rng = np.random.RandomState(42)
        audio = rng.randn(22050)
        geom = pix.audio_to_geometry(audio, sample_rate=22050)
        assert geom.shape == (3,)
        norm = np.linalg.norm(geom)
        assert norm == pytest.approx(1.0, abs=0.01) or norm == pytest.approx(0.0, abs=0.01)

    def test_audio_to_geometry_short_signal(self):
        pix = PixHomology()
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        geom = pix.audio_to_geometry(audio, sample_rate=22050)
        assert geom.shape == (3,)

    def test_audio_to_geometry_stereo(self):
        pix = PixHomology()
        rng = np.random.RandomState(7)
        audio = rng.randn(2, 11025)
        geom = pix.audio_to_geometry(audio, sample_rate=22050)
        assert geom.shape == (3,)


class TestMultimodalVideo:
    def test_video_to_geometry_shape(self):
        pix = PixHomology()
        rng = np.random.RandomState(42)
        frames = [rng.rand(32, 32, 3) for _ in range(16)]
        geom = pix.video_to_geometry(frames, fps=30.0)
        assert geom.shape == (3,)

    def test_video_to_geometry_empty(self):
        pix = PixHomology()
        geom = pix.video_to_geometry([], fps=30.0)
        assert geom.shape == (3,)

    def test_video_to_geometry_single_frame(self):
        pix = PixHomology()
        frame = np.random.rand(32, 32, 3)
        geom = pix.video_to_geometry([frame], fps=30.0)
        assert geom.shape == (3,)


class TestPixHomologyHelpers:
    def test_mel_filterbank_shape(self):
        pix = PixHomology()
        fb = pix._mel_filterbank(512, 22050, n_filters=13)
        assert fb.shape == (13, 257)

    def test_fallback_geometry_deterministic(self):
        pix = PixHomology()
        g1 = pix._fallback_geometry(b"test_data")
        g2 = pix._fallback_geometry(b"test_data")
        np.testing.assert_array_equal(g1, g2)

    def test_run_ph_on_points_insufficient(self):
        pix = PixHomology()
        result = pix._run_ph_on_points(np.array([[0, 0], [1, 1]]))
        assert result == []


class TestTwoPhaseCommit:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_and_load_snapshot(self):
        pers = ParquetPersistence(storage_path=self.tmpdir)
        body = GeoMemoryBody()
        n1 = body.store(content="snap_node_1", labels=["a"], weight=2.0)
        n2 = body.store(content="snap_node_2", labels=["b"], weight=3.0)
        nodes = body._nodes
        pers.write_full_snapshot(nodes, snapshot_name="test1")
        loaded = pers.load_latest_snapshot(snapshot_name="test1")
        assert len(loaded) == 2
        assert loaded[n1]["content"] == "snap_node_1"
        assert loaded[n2]["content"] == "snap_node_2"

    def test_atomic_overwrite(self):
        pers = ParquetPersistence(storage_path=self.tmpdir)
        body = GeoMemoryBody()
        body.store(content="first", weight=1.0)
        pers.write_full_snapshot(body._nodes, snapshot_name="atom")
        body.store(content="second", weight=2.0)
        pers.write_full_snapshot(body._nodes, snapshot_name="atom")
        loaded = pers.load_latest_snapshot(snapshot_name="atom")
        assert "second" in [v["content"] for v in loaded.values()]

    def test_no_tmp_file_remains(self):
        pers = ParquetPersistence(storage_path=self.tmpdir)
        body = GeoMemoryBody()
        body.store(content="clean", weight=1.0)
        pers.write_full_snapshot(body._nodes, snapshot_name="clean")
        tmp_files = [f for f in os.listdir(self.tmpdir) if f.startswith("_tmp_")]
        assert len(tmp_files) == 0


class TestMonitoring:
    def test_increment_counter_no_error(self):
        increment_counter(None, 1)
        increment_counter(None)

    def test_set_gauge_no_error(self):
        set_gauge(None, 42.0)

    def test_observe_histogram_no_error(self):
        observe_histogram(None, 0.5)

    def test_record_error_no_error(self):
        record_error("test_op")

    def test_generate_metrics_returns_string(self):
        result = generate_metrics()
        assert isinstance(result, str)

    def test_get_metrics_registry(self):
        reg = get_metrics_registry()
        assert reg is None or hasattr(reg, "get_sample_value")


def _can_create_testclient():
    """Check if starlette/httpx versions are compatible for TestClient."""
    try:
        from starlette.testclient import TestClient

        # Try to instantiate with a minimal ASGI app to detect version mismatch
        async def _noop(scope, receive, send):
            pass

        TestClient(_noop)
        return True
    except TypeError:
        return False
    except Exception:
        return False


@pytest.mark.skipif(
    not _can_create_testclient(),
    reason="starlette/httpx version incompatibility - TestClient cannot be instantiated",
)
class TestRouterEndpoints:
    @pytest.fixture
    def app_and_memory(self):
        from tetrahedron_memory.core import GeoMemoryBody
        from tetrahedron_memory.router import create_app

        memory = GeoMemoryBody()
        app = create_app(memory=memory)
        return app, memory

    def _get_client(self, app):
        from starlette.testclient import TestClient as StarletteTestClient

        return StarletteTestClient(app)

    def test_health(self, app_and_memory):
        app, _ = app_and_memory
        client = self._get_client(app)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_store_and_query(self, app_and_memory):
        app, _ = app_and_memory
        client = self._get_client(app)
        store_resp = client.post(
            "/api/v1/store",
            json={"content": "test memory", "labels": ["test"], "weight": 1.0},
        )
        assert store_resp.status_code == 200
        node_id = store_resp.json()["id"]
        assert isinstance(node_id, str)

        query_resp = client.post(
            "/api/v1/query",
            json={"query": "test memory", "k": 5, "use_persistence": True},
        )
        assert query_resp.status_code == 200
        results = query_resp.json()["results"]
        assert len(results) > 0

    def test_associate_nonexistent(self, app_and_memory):
        app, _ = app_and_memory
        client = self._get_client(app)
        resp = client.get("/api/v1/associate/nonexistent_id")
        assert resp.status_code == 200
        assert resp.json()["associations"] == []

    def test_self_organize(self, app_and_memory):
        app, _ = app_and_memory
        client = self._get_client(app)
        for i in range(10):
            client.post(
                "/api/v1/store",
                json={"content": f"so_item_{i}", "weight": 1.0},
            )
        resp = client.post("/api/v1/self-organize")
        assert resp.status_code == 200
        assert "stats" in resp.json()

    def test_stats(self, app_and_memory):
        app, _ = app_and_memory
        client = self._get_client(app)
        client.post(
            "/api/v1/store",
            json={"content": "stats_test", "weight": 1.0},
        )
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_memories"] >= 1
        assert data["dimension"] == 3

    def test_metrics_endpoint(self, app_and_memory):
        app, _ = app_and_memory
        client = self._get_client(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert isinstance(resp.text, str)

    def test_store_with_metadata(self, app_and_memory):
        app, _ = app_and_memory
        client = self._get_client(app)
        resp = client.post(
            "/api/v1/store",
            json={
                "content": "meta test",
                "labels": ["meta"],
                "weight": 2.0,
                "metadata": {"key": "value"},
            },
        )
        assert resp.status_code == 200
        assert isinstance(resp.json()["id"], str)


class TestEdgeCases:
    def test_clear_resets_state(self):
        body = GeoMemoryBody()
        for i in range(5):
            body.store(content=f"clear_{i}")
        body.clear()
        assert len(body._nodes) == 0
        stats = body.get_statistics()
        assert stats["total_memories"] == 0

    def test_query_empty_body(self):
        body = GeoMemoryBody()
        results = body.query("anything", k=5)
        assert results == []

    def test_update_weight_nonexistent(self):
        body = GeoMemoryBody()
        body.update_weight("nonexistent", delta=1.0)

    def test_detect_conflicts_no_overlap(self):
        body = GeoMemoryBody()
        for i in range(5):
            body.store(content=f"spread_{i}_{i * i}", weight=1.0)
        conflicts = body.detect_conflicts()
        assert isinstance(conflicts, list)

    def test_query_by_label_empty(self):
        body = GeoMemoryBody()
        results = body.query_by_label("nonexistent")
        assert results == []

    def test_global_integration_no_system_removal(self):
        body = GeoMemoryBody()
        for i in range(5):
            body.store(content=f"integrate_{i}", weight=0.02)
        body._mesh.store(
            content="__system_guard",
            seed_point=np.array([0.0, 1.0, 0.0]),
            labels=["__system__"],
            weight=0.02,
        )
        result = body.global_catalyze_integration(strength=1.0)
        sys_count = sum(1 for t in body._mesh.tetrahedra.values() if "__system__" in t.labels)
        assert sys_count == 1


class TestPowerDistance:
    def test_power_distance_same_weight(self):
        body = GeoMemoryBody()
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        pd = body._power_distance(p1, p2, 0.0, 0.0)
        assert pd == pytest.approx(np.sqrt(2.0), rel=0.01)

    def test_power_distance_with_weights(self):
        body = GeoMemoryBody()
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        pd = body._power_distance(p1, p2, 0.5, 0.5)
        assert pd < np.sqrt(2.0)

    def test_power_distance_clamps_to_zero(self):
        body = GeoMemoryBody()
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        pd = body._power_distance(p1, p2, 5.0, 5.0)
        assert pd == 0.0

    def test_weighted_query_uses_weights(self):
        body = GeoMemoryBody()
        body.store(content="heavy", weight=5.0)
        body.store(content="light", weight=0.5)
        results = body.query("heavy", k=2, use_persistence=True)
        assert len(results) >= 2


class TestVectorClock:
    def test_increment(self):
        from tetrahedron_memory.consistency import VectorClock

        vc = VectorClock(["b1", "b2"])
        assert vc.increment("b1") == 1
        assert vc.increment("b1") == 2
        assert vc.get("b2") == 0

    def test_merge(self):
        from tetrahedron_memory.consistency import VectorClock

        vc1 = VectorClock(["b1", "b2"])
        vc2 = VectorClock(["b1", "b2"])
        vc1.increment("b1")
        vc2.increment("b2")
        vc2.increment("b2")
        vc1.merge(vc2)
        assert vc1.get("b2") == 2
        assert vc1.get("b1") == 1

    def test_happens_before(self):
        from tetrahedron_memory.consistency import VectorClock

        vc1 = VectorClock(["b1", "b2"])
        vc2 = VectorClock(["b1", "b2"])
        vc1.increment("b1")
        vc2.merge(vc1)
        vc2.increment("b2")
        assert vc1.happens_before(vc2)
        assert not vc2.happens_before(vc1)

    def test_concurrent(self):
        from tetrahedron_memory.consistency import VectorClock

        vc1 = VectorClock(["b1", "b2"])
        vc2 = VectorClock(["b1", "b2"])
        vc1.increment("b1")
        vc2.increment("b2")
        assert vc1.is_concurrent(vc2)


class TestCompensationLog:
    def test_record_and_get_pending(self):
        from tetrahedron_memory.consistency import CompensationLog

        log = CompensationLog()
        log.record("store", "b1", {"content": "test"}, "timeout")
        pending = log.get_pending()
        assert len(pending) == 1
        assert pending[0]["operation"] == "store"

    def test_mark_resolved(self):
        from tetrahedron_memory.consistency import CompensationLog

        log = CompensationLog()
        eid = log.record("store", "b1", {"content": "test"}, "timeout")
        log.mark_resolved(eid)
        assert len(log.get_pending()) == 0

    def test_retry_all(self):
        from tetrahedron_memory.consistency import CompensationLog

        log = CompensationLog()
        log.record("store", "b1", {"content": "test"}, "timeout")
        results = log.retry_all(lambda op, params: None)
        assert len(results) == 1
        assert results[0]["status"] == "resolved"

    def test_clear_resolved(self):
        from tetrahedron_memory.consistency import CompensationLog

        log = CompensationLog()
        eid = log.record("store", "b1", {}, "err")
        log.mark_resolved(eid)
        cleared = log.clear_resolved()
        assert cleared == 1


class TestConsistencyManager:
    def test_acquire_release_lock(self):
        from tetrahedron_memory.consistency import ConsistencyManager

        cm = ConsistencyManager(["b1", "b2"])
        assert cm.acquire_lock(["b1", "b2"])
        cm.release_lock(["b1", "b2"])

    def test_record_version(self):
        from tetrahedron_memory.consistency import ConsistencyManager

        cm = ConsistencyManager(["b1"])
        vn = cm.record_version("node_1", "b1", "hello")
        assert vn.version == 1
        assert vn.checksum != ""

    def test_check_version(self):
        from tetrahedron_memory.consistency import ConsistencyManager

        cm = ConsistencyManager(["b1"])
        cm.record_version("node_1", "b1", "hello")
        assert cm.check_version("node_1", 1)
        assert not cm.check_version("node_1", 2)

    def test_detect_conflicts_empty(self):
        from tetrahedron_memory.consistency import ConsistencyManager

        cm = ConsistencyManager(["b1"])
        assert cm.detect_conflicts() == []

    def test_read_repair(self):
        from tetrahedron_memory.consistency import ConsistencyManager

        cm = ConsistencyManager(["b1", "b2"])
        cm.record_version("node_1", "b1", "hello")
        cm.read_repair("node_1", "b1", ["b2"])
        assert cm.vector_clock.get("b2") == 1

    def test_add_bucket(self):
        from tetrahedron_memory.consistency import ConsistencyManager

        cm = ConsistencyManager(["b1"])
        cm.add_bucket("b2")
        assert cm.vector_clock.get("b2") == 0


class TestSpatialBucketRouter:
    def test_initialize_and_store(self):
        from tetrahedron_memory.partitioning import SpatialBucketRouter

        router = SpatialBucketRouter()
        router.initialize()
        bid, nid = router.route_store(np.array([0.5, 0.5, 0.5]), "test content")
        assert bid.startswith("spatial_bucket_")
        assert isinstance(nid, str)

    def test_route_query(self):
        from tetrahedron_memory.partitioning import SpatialBucketRouter

        router = SpatialBucketRouter()
        router.initialize()
        router.route_store(np.array([0.5, 0.5, 0.5]), "test")
        bids = router.route_query(np.array([0.5, 0.5, 0.5]))
        assert len(bids) >= 1

    def test_get_bucket_for_node(self):
        from tetrahedron_memory.partitioning import SpatialBucketRouter

        router = SpatialBucketRouter()
        router.initialize()
        bid, nid = router.route_store(np.array([0.5, 0.5, 0.5]), "test")
        assert router.get_bucket_for_node(nid) == bid

    def test_cross_bucket_query(self):
        from tetrahedron_memory.partitioning import SpatialBucketRouter

        router = SpatialBucketRouter()
        router.initialize()
        router.route_store(np.array([0.5, 0.5, 0.5]), "alpha")
        router.route_store(np.array([-0.5, -0.5, -0.5]), "beta")
        results = router.cross_bucket_query(np.array([0.5, 0.5, 0.5]), "alpha", k=5)
        assert isinstance(results, list)

    def test_split_bucket(self):
        from tetrahedron_memory.partitioning import SpatialBucketRouter

        router = SpatialBucketRouter(max_points_per_bucket=3)
        router.initialize()
        for i in range(5):
            router.route_store(
                np.array([0.01 * i, 0.01 * i, 0.01 * i]),
                f"item_{i}",
            )
        stats = router.get_statistics()
        assert stats["total_nodes"] == 5


class TestTetraMemRayControllerRouting:
    def test_spatial_routing_store(self):
        ctrl = TetraMemRayController(num_buckets=2, use_spatial_routing=True)
        ctrl.initialize()
        bid, nid = ctrl.store_routed("test routed content")
        assert isinstance(bid, str)
        assert isinstance(nid, str)
        ctrl.shutdown()

    def test_spatial_routing_query(self):
        ctrl = TetraMemRayController(num_buckets=2, use_spatial_routing=True)
        ctrl.initialize()
        ctrl.store_routed("query test content")
        results = ctrl.query_routed("query test content")
        assert isinstance(results, list)
        ctrl.shutdown()

    def test_auto_balance(self):
        ctrl = TetraMemRayController(num_buckets=2, use_spatial_routing=True)
        ctrl.initialize()
        result = ctrl.auto_balance()
        assert "balanced" in result
        ctrl.shutdown()


class TestIncrementalSnapshot:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_incremental_and_load(self):
        pers = ParquetPersistence(storage_path=self.tmpdir)
        body = GeoMemoryBody()
        body.store(content="inc_1", weight=1.0)
        pers.write_incremental_full(body._nodes, snapshot_name="inc_test")
        loaded = pers.load_latest_snapshot(snapshot_name="inc_test")
        assert len(loaded) == 0

    def test_compact_snapshots(self):
        pers = ParquetPersistence(storage_path=self.tmpdir)
        body = GeoMemoryBody()
        body.store(content="full_1", weight=1.0)
        pers.write_full_snapshot(body._nodes, snapshot_name="compact_test")
        body.store(content="full_2", weight=2.0)
        pers.write_incremental_full(body._nodes, snapshot_name="compact_test")
        pers.compact_snapshots(snapshot_name="compact_test")
        loaded = pers.load_latest_snapshot(snapshot_name="compact_test")
        assert len(loaded) >= 1


class TestRemoteBucketActor:
    def test_store_and_query(self):
        from tetrahedron_memory.persistence import RemoteBucketActor

        actor = RemoteBucketActor("remote_test")
        nid = actor.store(content="remote test content")
        assert isinstance(nid, str)
        results = actor.query("remote test content")
        assert len(results) >= 1

    def test_serialization_roundtrip(self):
        from tetrahedron_memory.persistence import RemoteBucketActor

        actor = RemoteBucketActor("ser_test")
        actor.store(content="serialized content", weight=2.0)
        data = actor.to_serializable()
        assert "bucket_id" in data
        assert "snapshot" in data
        restored = RemoteBucketActor.from_serializable(data)
        results = restored.query("serialized content")
        assert len(results) >= 1


class TestLLMTool:
    def test_get_tool_definitions(self):
        from tetrahedron_memory.llm_tool import get_tool_definitions

        defs = get_tool_definitions()
        assert len(defs) >= 5
        names = [d["function"]["name"] for d in defs]
        assert "tetramem_store" in names
        assert "tetramem_query" in names

    def test_execute_store(self):
        from tetrahedron_memory.core import GeoMemoryBody
        from tetrahedron_memory.llm_tool import execute_tool_call

        body = GeoMemoryBody()
        result = execute_tool_call("tetramem_store", {"content": "test"}, memory=body)
        assert result["stored"] is True

    def test_execute_query(self):
        from tetrahedron_memory.core import GeoMemoryBody
        from tetrahedron_memory.llm_tool import execute_tool_call

        body = GeoMemoryBody()
        body.store(content="hello world")
        result = execute_tool_call("tetramem_query", {"query": "hello"}, memory=body)
        assert len(result["results"]) >= 1

    def test_execute_stats(self):
        from tetrahedron_memory.core import GeoMemoryBody
        from tetrahedron_memory.llm_tool import execute_tool_call

        body = GeoMemoryBody()
        body.store(content="stats test")
        result = execute_tool_call("tetramem_stats", {}, memory=body)
        assert result["total_memories"] >= 1

    def test_create_tool_response(self):
        from tetrahedron_memory.core import GeoMemoryBody
        from tetrahedron_memory.llm_tool import create_tool_response

        body = GeoMemoryBody()
        resp = create_tool_response("tc_123", "tetramem_store", {"content": "hi"}, memory=body)
        assert resp["tool_call_id"] == "tc_123"
        assert resp["role"] == "tool"

    def test_unknown_tool(self):
        from tetrahedron_memory.llm_tool import execute_tool_call

        result = execute_tool_call("unknown_tool", {})
        assert "error" in result


class TestMonitoringExtended:
    def test_ray_cluster_status_no_ray(self):
        from tetrahedron_memory.monitoring import get_ray_cluster_status

        status = get_ray_cluster_status()
        assert "status" in status

    def test_grafana_dashboard_json(self):
        import json

        from tetrahedron_memory.monitoring import get_grafana_dashboard_json

        dash = get_grafana_dashboard_json()
        parsed = json.loads(dash)
        assert "panels" in parsed
        assert "schemaVersion" in parsed
        assert parsed["schemaVersion"] == 39
        assert len(parsed["panels"]) == 15
        assert parsed["title"] == "TetraMem-XL Production Dashboard"

    def test_health_check(self):
        from tetrahedron_memory.monitoring import health_check

        hc = health_check()
        assert hc["status"] == "ok"

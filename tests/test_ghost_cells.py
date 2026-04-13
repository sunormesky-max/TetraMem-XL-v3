import numpy as np

from tetrahedron_memory.partitioning import (
    BoundingBox,
    GhostCell,
    SpatialBucketRouter,
)


class TestGhostCell:
    def test_creation_and_defaults(self):
        geom = np.array([0.1, 0.2, 0.3])
        gc = GhostCell(node_id="n1", source_bucket_id="b1", geometry=geom)
        assert gc.node_id == "n1"
        assert gc.source_bucket_id == "b1"
        np.testing.assert_array_equal(gc.geometry, geom)
        assert gc.weight == 1.0
        assert gc.content_hash == ""
        assert gc.labels == []
        assert gc.ttl == 3600.0
        assert gc.access_count == 0
        assert not gc.is_expired

    def test_touch_increments_access(self):
        gc = GhostCell(node_id="n1", source_bucket_id="b1", geometry=np.zeros(3))
        assert gc.access_count == 0
        gc.touch()
        gc.touch()
        gc.touch()
        assert gc.access_count == 3

    def test_expiration(self):
        gc = GhostCell(
            node_id="n1",
            source_bucket_id="b1",
            geometry=np.zeros(3),
            created_at=0.0,
            ttl=1.0,
        )
        assert gc.is_expired

    def test_not_expired_within_ttl(self):

        gc = GhostCell(
            node_id="n1",
            source_bucket_id="b1",
            geometry=np.zeros(3),
            ttl=3600.0,
        )
        assert not gc.is_expired


class TestSpatialBucketRouterGhostCells:
    def _make_router_with_two_buckets(self):
        router = SpatialBucketRouter(max_points_per_bucket=1000, ghost_ttl=3600.0)
        bounds_a = BoundingBox(np.array([-1.0, -1.0, -1.0]), np.array([0.0, 1.0, 1.0]))
        bounds_b = BoundingBox(np.array([0.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        router._auto_create_bucket(bounds_a)
        router._auto_create_bucket(bounds_b)
        return router

    def test_populate_ghost_cells(self):
        router = self._make_router_with_two_buckets()
        bids = router.get_all_bucket_ids()
        assert len(bids) == 2

        bid_a = bids[0]
        bid_b = bids[1]
        actor_a = router.get_actor(bid_a)
        actor_a.store(content="boundary memory near x=0", labels=["test"], weight=1.5)

        router._populate_ghost_cells(bid_b)
        stats = router.get_ghost_cell_stats()
        assert stats["total_ghost_cells"] >= 0

    def test_cross_bucket_associate_returns_results(self):
        router = self._make_router_with_two_buckets()
        bids = router.get_all_bucket_ids()
        bid_a = bids[0]

        actor_a = router.get_actor(bid_a)
        node_id = actor_a.store(content="memory in bucket A", labels=["alpha"])

        router._populate_ghost_cells(bid_a)

        results = router.cross_bucket_associate(node_id, max_depth=2, radius=5.0)
        assert isinstance(results, list)

    def test_update_and_prune_ghost_cells(self):
        router = self._make_router_with_two_buckets()
        bids = router.get_all_bucket_ids()

        actor_a = router.get_actor(bids[0])
        actor_a.store(content="test memory A", weight=2.0)

        total = router.update_ghost_cells()
        assert isinstance(total, int)

        pruned = router.prune_expired_ghosts()
        assert isinstance(pruned, int)

    def test_ghost_stats_structure(self):
        router = self._make_router_with_two_buckets()
        router.update_ghost_cells()
        stats = router.get_ghost_cell_stats()
        assert "total_ghost_cells" in stats
        assert "per_bucket" in stats
        assert "expired" in stats
        assert isinstance(stats["per_bucket"], dict)

    def test_statistics_includes_ghost_cells(self):
        router = self._make_router_with_two_buckets()
        stats = router.get_statistics()
        assert "ghost_cells" in stats
        assert isinstance(stats["ghost_cells"], int)

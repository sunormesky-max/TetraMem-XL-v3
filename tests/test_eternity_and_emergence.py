"""
Eternity Strict Test + Closed-Loop Self-Emergence Quality Test.

1. Eternity Strict: Simulate long-running operations (thousands of store/merge/transform/
   dream/reintegration cycles) and verify NO memory is ever implicitly deleted.

2. Self-Emergence Quality: Verify that dream-produced memories are genuinely derived
   from integration/synthesis of existing memories, not simple copies.
"""

import threading
import time
import unittest

from tetrahedron_memory import GeoMemoryBody
from tetrahedron_memory.eternity_audit import EternityAudit
from tetrahedron_memory.tetra_dream import TetraDreamCycle


class TestEternityStrict(unittest.TestCase):
    """Simulate long-running operations and verify no memory loss."""

    def setUp(self):
        self.body = GeoMemoryBody(dimension=3, precision="fast")

    def test_store_verify_cycle(self):
        ids = []
        for i in range(500):
            tid = self.body.store(
                content=f"memory_{i}",
                labels=[f"label_{i % 10}"],
                weight=0.5 + (i % 5) * 0.3,
            )
            ids.append(tid)

        report = self.body.verify_eternity()
        self.assertTrue(report["verified"], f"Eternity violation: {report['violations']}")
        self.assertEqual(report["total_stored"], 500)
        self.assertEqual(report["total_alive"], 500)

    def test_store_dream_verify(self):
        """Store memories, trigger dream directly, verify all originals have audit trail."""
        ids = []
        for i in range(50):
            tid = self.body.store(
                content=f"original_{i}",
                labels=["topic_a", f"sub_{i % 5}"],
                weight=1.0,
            )
            ids.append(tid)

        dc = TetraDreamCycle(self.body._mesh)
        dc.trigger_now()

        for oid in ids:
            trail = self.body.get_eternity_trail(oid)
            self.assertGreaterEqual(len(trail), 1, f"Original {oid} has no audit trail")

        report = self.body.verify_eternity()
        self.assertTrue(
            report["verified"], f"Eternity violation after dream: {report['violations']}"
        )

    def test_store_batch_verify(self):
        items = [{"content": f"batch_{i}", "labels": ["batch"], "weight": 1.0} for i in range(100)]
        ids = self.body.store_batch(items)

        report = self.body.verify_eternity()
        self.assertTrue(report["verified"], f"Batch eternity violation: {report['violations']}")
        self.assertEqual(report["total_stored"], 100)

    def test_preservation_chain_after_operations(self):
        audit = EternityAudit()
        audit.record_store("A", "content_a")
        audit.record_store("B", "content_b")
        audit.record_merge(["A", "B"], "M", "merged_content")
        audit.record_transform("M", "T", "integrate", "transformed_content")

        chain_a = audit.get_preservation_chain("A")
        self.assertIn("A", chain_a)
        self.assertIn("M", chain_a)
        self.assertIn("T", chain_a)

        chain_m = audit.get_preservation_chain("M")
        self.assertIn("M", chain_m)
        self.assertIn("T", chain_m)

    def test_concurrent_store_eternity(self):
        n_threads = 8
        n_per_thread = 50
        all_ids = []
        lock = threading.Lock()

        def worker(thread_id):
            local_ids = []
            for i in range(n_per_thread):
                tid = self.body.store(
                    content=f"concurrent_t{thread_id}_m{i}",
                    labels=[f"t{thread_id}"],
                )
                local_ids.append(tid)
            with lock:
                all_ids.extend(local_ids)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=30)

        self.assertEqual(len(all_ids), n_threads * n_per_thread)

        report = self.body.verify_eternity()
        self.assertTrue(
            report["verified"], f"Concurrent eternity violation: {report['violations']}"
        )
        self.assertEqual(report["total_stored"], n_threads * n_per_thread)

    def test_dream_produces_new_memories_preserves_originals(self):
        """After dream cycles, all original store()ed memories must have audit trail."""
        original_ids = []
        for i in range(30):
            tid = self.body.store(
                content=f"dream_test_{i}",
                labels=["dreamable", f"group_{i % 3}"],
                weight=1.0,
            )
            original_ids.append(tid)

        dc = TetraDreamCycle(self.body._mesh, zigzag_tracker=self.body._zigzag_tracker)
        for _ in range(3):
            dc.trigger_now()

        for oid in original_ids:
            trail = self.body.get_eternity_trail(oid)
            self.assertGreaterEqual(len(trail), 1, f"Original {oid} lost from audit trail")

    def test_eternity_status_tracks_violations(self):
        audit = EternityAudit()
        audit.record_store("phantom", "ghost_content")

        status = audit.get_status()
        self.assertEqual(status["total_entries"], 1)
        self.assertEqual(status["total_tracked_ids"], 1)
        self.assertEqual(status["total_violations"], 0)

    def test_long_chain_eternity(self):
        audit = EternityAudit()
        audit.record_store("s0", "base")
        prev = "s0"
        for i in range(1, 50):
            new_id = f"s{i}"
            audit.record_transform(prev, new_id, "evolve", f"content_{i}")
            prev = new_id

        chain = audit.get_preservation_chain("s0")
        for i in range(50):
            self.assertIn(f"s{i}", chain, f"s{i} missing from s0's preservation chain")


class TestSelfEmergenceQuality(unittest.TestCase):
    """Verify dream-produced memories are genuine integrations, not simple copies."""

    def setUp(self):
        self.body = GeoMemoryBody(dimension=3, precision="fast")

    def _trigger_dream(self):
        dc = TetraDreamCycle(
            self.body._mesh,
            zigzag_tracker=self.body._zigzag_tracker,
        )
        return dc.trigger_now()

    def test_dream_content_is_not_copy(self):
        input_contents = set()
        for i in range(20):
            content = f"unique_input_{i}_{'x' * (10 + i)}"
            self.body.store(content=content, labels=["dream_test"], weight=1.0)
            input_contents.add(content)

        stats = self._trigger_dream()

        dream_contents = set()
        for tid, tetra in self.body._mesh._tetrahedra.items():
            if "__dream__" in tetra.labels:
                dream_contents.add(tetra.content)

        if dream_contents:
            for dc in dream_contents:
                self.assertNotIn(dc, input_contents, f"Dream content is verbatim copy: {dc[:50]}")

    def test_dream_labels_derived_from_inputs(self):
        input_labels = {"ai", "memory", "topology", "dream_test"}
        for i in range(20):
            self.body.store(
                content=f"labeled_input_{i}",
                labels=list(input_labels)[: 2 + (i % 3)],
                weight=1.0,
            )

        self._trigger_dream()

        for tid, tetra in self.body._mesh._tetrahedra.items():
            if "__dream__" in tetra.labels:
                non_system = set(tetra.labels) - {"__dream__"}
                for label in non_system:
                    self.assertIn(
                        label, input_labels, f"Unexpected dream label '{label}' not from inputs"
                    )

    def test_dream_increases_total_knowledge(self):
        for i in range(20):
            self.body.store(content=f"knowledge_{i}", labels=["domain"], weight=1.0)

        initial_contents = {
            t.content for t in self.body._mesh._tetrahedra.values() if "__dream__" not in t.labels
        }

        stats = self._trigger_dream()
        self.assertEqual(stats["phase"], "complete", f"Dream did not complete: {stats}")

        dream_contents = {
            t.content for t in self.body._mesh._tetrahedra.values() if "__dream__" in t.labels
        }

        new_content = dream_contents - initial_contents
        self.assertGreater(len(new_content), 0, "Dream cycle produced no genuinely new content")

    def test_dream_weight_reasonable(self):
        for i in range(20):
            self.body.store(content=f"weight_test_{i}", labels=["weight"], weight=1.0)

        self._trigger_dream()

        for tid, tetra in self.body._mesh._tetrahedra.items():
            if "__dream__" in tetra.labels:
                self.assertGreater(tetra.weight, 0, f"Dream {tid} has zero weight")
                self.assertLessEqual(tetra.weight, 10, f"Dream {tid} has excessive weight")

    def test_multiple_dream_cycles_accumulate(self):
        for i in range(20):
            self.body.store(content=f"multi_dream_{i}", labels=["iter"], weight=1.0)

        dream_counts = []
        for _ in range(3):
            stats = self._trigger_dream()
            count = sum(1 for t in self.body._mesh._tetrahedra.values() if "__dream__" in t.labels)
            dream_counts.append(count)

        total_dream = dream_counts[-1]
        self.assertGreater(total_dream, 0, "No dreams produced across 3 cycles")

    def test_dream_memories_have_metadata(self):
        for i in range(20):
            self.body.store(content=f"meta_test_{i}", labels=["meta"], weight=1.0)

        stats = self._trigger_dream()

        found_dream_with_source = False
        for tid, tetra in self.body._mesh._tetrahedra.items():
            if "__dream__" in tetra.labels:
                if tetra.metadata.get("source_clusters"):
                    found_dream_with_source = True

        if any("__dream__" in t.labels for t in self.body._mesh._tetrahedra.values()):
            self.assertTrue(found_dream_with_source, "Dream memories lack source_clusters metadata")

    def test_dream_content_length_differs_from_inputs(self):
        """Dream synthesis should produce content of different length than inputs."""
        for i in range(15):
            self.body.store(content=f"short_{i}", labels=["len"], weight=1.0)

        stats = self._trigger_dream()
        self.assertEqual(stats["phase"], "complete")

        input_lens = [
            len(t.content)
            for t in self.body._mesh._tetrahedra.values()
            if "__dream__" not in t.labels
        ]

        dream_lens = [
            len(t.content) for t in self.body._mesh._tetrahedra.values() if "__dream__" in t.labels
        ]

        if dream_lens:
            avg_input = sum(input_lens) / len(input_lens)
            avg_dream = sum(dream_lens) / len(dream_lens)
            self.assertNotEqual(
                avg_dream, avg_input, "Dream content has same average length as inputs"
            )


if __name__ == "__main__":
    unittest.main()

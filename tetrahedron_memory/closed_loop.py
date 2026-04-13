"""
ClosedLoop — Complete TetraMem-XL cognitive cycle.

Implements the full closed-loop per the Grok production specification:
    记忆 → 思考 → 执行 → 反思 → 整合 → 梦境 → 记忆 (循环)

The loop can be triggered from ANY stage:
    - External input → store memory → trigger cycle
    - Timer tick → self-organize → dream → integrate
    - Query hit → recall → reflect → integrate
    - Entropy threshold exceeded → integrate → dream → store insight

Each cycle:
    1. RECALL: Query relevant memories for the current context
    2. THINK: Analyze recalled memories, detect patterns (PH-weighted)
    3. EXECUTE: Generate output/action based on analysis
    4. REFLECT: Evaluate the result, detect contradictions or insights
    5. INTEGRATE: Merge insights into existing memory topology
    6. DREAM: If entropy is high, trigger dream cycle for consolidation

Key properties:
    - NO deletion — all intermediate results are stored as memories
    - Persistent entropy drives integration timing
    - Self-emergence: system can trigger cycles autonomously
    - Thread-safe background daemon mode
"""

import logging
import threading
import time
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .persistent_entropy import (
    EntropyTracker,
    compute_entropy_delta,
    compute_persistent_entropy,
    should_trigger_integration,
)

logger = logging.getLogger("tetramem.closedloop")


class LoopPhase(Enum):
    IDLE = auto()
    RECALL = auto()
    THINK = auto()
    EXECUTE = auto()
    REFLECT = auto()
    INTEGRATE = auto()
    DREAM = auto()
    COMPLETE = auto()


ThinkFn = Callable[[Dict[str, Any]], Dict[str, Any]]
ExecuteFn = Callable[[Dict[str, Any]], Dict[str, Any]]
ReflectFn = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def _default_think(recall_result: Dict[str, Any]) -> Dict[str, Any]:
    memories = recall_result.get("memories", [])
    if not memories:
        return {"analysis": "no_memories", "confidence": 0.0, "patterns": []}
    weights = [m.get("weight", 1.0) for m in memories]
    avg_weight = float(np.mean(weights))
    labels = set()
    for m in memories:
        labels.update(m.get("labels", []))
    labels.discard("__system__")
    labels.discard("__dream__")
    return {
        "analysis": f"analyzed_{len(memories)}_memories",
        "confidence": min(1.0, avg_weight / 5.0),
        "patterns": list(labels)[:5],
        "avg_weight": avg_weight,
        "memory_count": len(memories),
    }


def _default_execute(think_result: Dict[str, Any]) -> Dict[str, Any]:
    analysis = think_result.get("analysis", "idle")
    confidence = think_result.get("confidence", 0.0)
    return {
        "action": f"derived_from_{analysis}",
        "confidence": confidence,
        "output": f"Insight: {analysis} (confidence={confidence:.2f})",
    }


def _default_reflect(
    think_result: Dict[str, Any], execute_result: Dict[str, Any]
) -> Dict[str, Any]:
    confidence = execute_result.get("confidence", 0.0)
    contradictions = 0
    if confidence < 0.3:
        contradictions = 1
    return {
        "quality": confidence,
        "contradictions": contradictions,
        "should_integrate": True,
        "should_dream": confidence < 0.5,
        "insight": execute_result.get("output", ""),
    }


class ClosedLoopEngine:
    """
    Complete cognitive closed-loop engine for TetraMem-XL.

    Parameters
    ----------
    memory : GeoMemoryBody
        The core memory engine.
    think_fn : callable, optional
        Custom think function: (recall_result) -> think_result.
        Replace with LLM call for production.
    execute_fn : callable, optional
        Custom execute function: (think_result) -> execute_result.
    reflect_fn : callable, optional
        Custom reflect function: (think_result, execute_result) -> reflect_result.
    entropy_integration_threshold : float
        Trigger integration when current/baseline entropy exceeds this ratio.
    dream_on_high_entropy : bool
        Trigger dream cycle when entropy is high after integration.
    auto_cycle_interval : float
        Seconds between autonomous cycles. 0 = disabled.
    """

    def __init__(
        self,
        memory: Any,
        think_fn: Optional[ThinkFn] = None,
        execute_fn: Optional[ExecuteFn] = None,
        reflect_fn: Optional[ReflectFn] = None,
        entropy_integration_threshold: float = 1.3,
        dream_on_high_entropy: bool = True,
        auto_cycle_interval: float = 0.0,
    ):
        self.memory = memory
        self.think_fn = think_fn or _default_think
        self.execute_fn = execute_fn or _default_execute
        self.reflect_fn = reflect_fn or _default_reflect
        self.entropy_integration_threshold = entropy_integration_threshold
        self.dream_on_high_entropy = dream_on_high_entropy
        self.auto_cycle_interval = auto_cycle_interval

        self._entropy_tracker = EntropyTracker()
        self._cycle_count = 0
        self._total_stores = 0
        self._total_integrations = 0
        self._total_dreams = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._last_cycle_result: Dict[str, Any] = {}

    def run_cycle(
        self,
        context: str = "",
        k: int = 5,
        force_dream: bool = False,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "phase": LoopPhase.IDLE.name,
            "context": context,
            "cycle": self._cycle_count,
            "entropy_before": 0.0,
            "entropy_after": 0.0,
            "entropy_delta": 0.0,
            "stores": 0,
            "integrations": 0,
            "dream_triggered": False,
        }

        with self._lock:
            result["entropy_before"] = self._measure_entropy()

            # Phase 1: RECALL
            result["phase"] = LoopPhase.RECALL.name
            recall = self._recall(context, k)
            result["recall_count"] = len(recall.get("memories", []))

            # Phase 2: THINK
            result["phase"] = LoopPhase.THINK.name
            think = self.think_fn(recall)
            result["think"] = think

            # Phase 3: EXECUTE
            result["phase"] = LoopPhase.EXECUTE.name
            execute = self.execute_fn(think)
            result["execute"] = execute

            # Phase 4: REFLECT
            result["phase"] = LoopPhase.REFLECT.name
            reflect = self.reflect_fn(think, execute)
            result["reflect"] = reflect

            # Phase 5: INTEGRATE
            result["phase"] = LoopPhase.INTEGRATE.name
            integration_count = self._integrate(reflect, think, execute)
            result["integrations"] = integration_count
            self._total_integrations += integration_count

            if reflect.get("should_integrate") and reflect.get("insight"):
                insight_id = self.memory.store(
                    content=reflect["insight"],
                    labels=["__closed_loop__", "__insight__"] + [
                        p for p in think.get("patterns", [])[:3]
                    ],
                    metadata={
                        "type": "closed_loop_insight",
                        "cycle": self._cycle_count,
                        "confidence": reflect.get("quality", 0.0),
                        "phase": "integrate",
                    },
                    weight=0.3 + reflect.get("quality", 0.0) * 0.7,
                )
                result["insight_id"] = insight_id
                result["stores"] += 1
                self._total_stores += 1

            # Phase 6: DREAM
            result["phase"] = LoopPhase.DREAM.name
            should_dream = (
                force_dream
                or (self.dream_on_high_entropy and reflect.get("should_dream", False))
                or self._entropy_tracker.should_integrate(self.entropy_integration_threshold)
            )
            if should_dream:
                dream_result = self._trigger_dream()
                result["dream_triggered"] = True
                result["dream_result"] = dream_result
                self._total_dreams += 1

            result["entropy_after"] = self._measure_entropy()
            if result["entropy_before"] > 0:
                result["entropy_delta"] = compute_entropy_delta(
                    result["entropy_before"], result["entropy_after"]
                )

            result["phase"] = LoopPhase.COMPLETE.name
            self._cycle_count += 1
            self._last_cycle_result = result

        return result

    def start_daemon(self, interval: Optional[float] = None) -> None:
        if interval is not None:
            self.auto_cycle_interval = interval
        if self.auto_cycle_interval <= 0:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._daemon_loop, name="tetramem-closedloop", daemon=True
        )
        self._thread.start()
        logger.info("ClosedLoopEngine daemon started (interval=%.0fs)", self.auto_cycle_interval)

    def stop_daemon(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "cycle_count": self._cycle_count,
            "total_stores": self._total_stores,
            "total_integrations": self._total_integrations,
            "total_dreams": self._total_dreams,
            "entropy": self._entropy_tracker.get_summary(),
            "last_cycle": {
                k: v for k, v in self._last_cycle_result.items()
                if k not in ("think", "execute", "reflect", "dream_result")
            },
        }

    def _recall(self, context: str, k: int) -> Dict[str, Any]:
        if not context:
            with self.memory._lock:
                nodes = list(self.memory._nodes.values())
            if len(nodes) <= k:
                sampled = nodes
            else:
                indices = np.random.choice(len(nodes), size=min(k, len(nodes)), replace=False)
                sampled = [nodes[i] for i in indices]
            memories = [
                {
                    "content": n.content,
                    "weight": n.weight,
                    "labels": n.labels,
                    "id": n.id,
                }
                for n in sampled
            ]
            return {"memories": memories, "method": "random"}

        results = self.memory.query(context, k=k)
        memories = [
            {
                "content": r.node.content,
                "weight": r.node.weight,
                "labels": r.node.labels,
                "id": r.node.id,
                "distance": r.distance,
                "persistence_score": r.persistence_score,
            }
            for r in results
        ]
        return {"memories": memories, "method": "query", "context": context}

    def _integrate(
        self,
        reflect: Dict[str, Any],
        think: Dict[str, Any],
        execute: Dict[str, Any],
    ) -> int:
        count = 0
        if not reflect.get("should_integrate", True):
            return 0

        if self.memory._use_mesh:
            mesh = self.memory._mesh
            st = mesh.compute_ph()
            if st is not None:
                entropy = compute_persistent_entropy(st)
                self._entropy_tracker.record(entropy)
                if should_trigger_integration(
                    entropy, self._entropy_tracker.baseline, self.entropy_integration_threshold
                ):
                    low_weight_ids = [
                        tid for tid, t in mesh.tetrahedra.items()
                        if t.weight < 0.5 and "__system__" not in t.labels
                    ]
                    if low_weight_ids:
                        result = mesh.catalyze_integration_batch(low_weight_ids[:20])
                        count += result.get("catalyzed", 0)
        else:
            integrations = self.memory.global_catalyze_integration(strength=1.0)
            count += integrations.get("catalyzed", 0)

        return count

    def _trigger_dream(self) -> Dict[str, Any]:
        if self.memory._use_mesh:
            from .tetra_dream import TetraDreamCycle
            mesh = self.memory._mesh
            dc = TetraDreamCycle(mesh, cycle_interval=999999)
            return dc.trigger_now()
        return {"phase": "skipped_no_mesh"}

    def _measure_entropy(self) -> float:
        if self.memory._use_mesh:
            st = self.memory._mesh.compute_ph()
            if st is not None:
                entropy = compute_persistent_entropy(st)
                self._entropy_tracker.record(entropy)
                return entropy
        return 0.0

    def _daemon_loop(self) -> None:
        while not self._stop.wait(timeout=self.auto_cycle_interval):
            try:
                self.run_cycle(context="")
            except Exception as e:
                logger.error("ClosedLoop daemon error: %s", e, exc_info=True)

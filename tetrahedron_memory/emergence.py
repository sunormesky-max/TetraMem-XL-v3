"""
Emergence Pressure & Adaptive Threshold for TetraMem-XL.

Per the design specification (Grok production plan):
  - Emergence pressure is a composite topological signal that drives
    self-emergence triggering. It integrates:
      1. Persistent entropy delta (noise accumulation rate)
      2. H2 void growth (structural gaps needing fill)
      3. H1 loop change (association pattern evolution)
      4. Local density anomaly (memory clustering)
      5. Time since last integration (staleness)
  - Adaptive threshold evolves based on integration effectiveness:
      - Good effect -> lower threshold -> encourage emergence
      - Poor effect -> raise threshold -> avoid wasteful cycles
      - Threshold history is recorded as meta-dream memory
  - Together they implement "AI decides when to think" — the system
    self-monitors its topological health and autonomously triggers
    dream cycles when needed.

Core principles:
  - NO deletion — thresholds only trigger integration, never forgetting
  - Self-adaptive — threshold itself enters the closed loop
  - Multi-signal — driven by topology, not timers
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("tetramem.emergence")


class AdaptiveThreshold:
    """
    Self-evolving threshold for emergence triggering.

    The threshold adjusts after each dream/integration cycle based on
    the effectiveness of the cycle (entropy reduction achieved).
    """

    def __init__(
        self,
        initial_value: float = 0.5,
        min_value: float = 0.1,
        max_value: float = 2.0,
        learning_rate: float = 0.1,
        good_effect_threshold: float = 0.18,
        poor_effect_threshold: float = 0.05,
    ):
        self._value = initial_value
        self._initial = initial_value
        self._min = min_value
        self._max = max_value
        self._lr = learning_rate
        self._good = good_effect_threshold
        self._poor = poor_effect_threshold
        self._history: List[Dict[str, Any]] = []
        self._max_history = 100
        self._consecutive_poor = 0
        self._consecutive_good = 0
        self._total_adjustments = 0
        self._lock = threading.RLock()

    @property
    def value(self) -> float:
        with self._lock:
            return self._value

    def update(self, effect_delta: float, pressure_before: float) -> Dict[str, Any]:
        with self._lock:
            old_value = self._value
            adjustment = 0.0
            direction = "none"

            if effect_delta >= self._good:
                self._consecutive_good += 1
                self._consecutive_poor = 0
                decrease = self._lr * min(effect_delta, 0.5)
                self._value = max(self._min, self._value - decrease)
                adjustment = self._value - old_value
                direction = "down"
            elif effect_delta <= self._poor and effect_delta >= 0:
                self._consecutive_poor += 1
                self._consecutive_good = 0
                increase = self._lr * 0.5
                if self._consecutive_poor >= 3:
                    increase *= 1.5
                self._value = min(self._max, self._value + increase)
                adjustment = self._value - old_value
                direction = "up"
            else:
                self._consecutive_good = 0
                self._consecutive_poor = 0
                direction = "hold"

            record = {
                "timestamp": time.time(),
                "old_threshold": old_value,
                "new_threshold": self._value,
                "adjustment": adjustment,
                "direction": direction,
                "effect_delta": effect_delta,
                "pressure_before": pressure_before,
                "consecutive_good": self._consecutive_good,
                "consecutive_poor": self._consecutive_poor,
            }
            self._history.append(record)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]
            self._total_adjustments += 1
            return record

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "value": self._value,
                "initial": self._initial,
                "min": self._min,
                "max": self._max,
                "total_adjustments": self._total_adjustments,
                "consecutive_good": self._consecutive_good,
                "consecutive_poor": self._consecutive_poor,
                "recent_directions": [r["direction"] for r in self._history[-5:]],
            }


class EmergencePressure:
    """
    Composite topological signal for self-emergence triggering.

    Integrates multiple topological signals into a single pressure value
    that indicates how urgently the system needs to dream/integrate.
    """

    def __init__(
        self,
        w_entropy: float = 0.35,
        w_h2: float = 0.25,
        w_h1: float = 0.15,
        w_density: float = 0.15,
        w_staleness: float = 0.10,
        staleness_halflife: float = 300.0,
        density_outlier_sigma: float = 2.0,
    ):
        self._weights = {
            "entropy": w_entropy,
            "h2": w_h2,
            "h1": w_h1,
            "density": w_density,
            "staleness": w_staleness,
        }
        self._staleness_halflife = staleness_halflife
        self._density_sigma = density_outlier_sigma
        self._last_integration_time: float = time.time()
        self._last_components: Dict[str, float] = {}
        self._last_pressure: float = 0.0
        self._history: List[Dict[str, Any]] = []
        self._max_history = 100
        self._lock = threading.RLock()

    def mark_integration(self) -> None:
        with self._lock:
            self._last_integration_time = time.time()

    def compute(self, mesh: Any, simplex_tree: Any = None) -> Dict[str, Any]:
        from .persistent_entropy import compute_persistent_entropy, compute_entropy_by_dimension

        components: Dict[str, float] = {}

        entropy_val = 0.0
        entropy_delta = 0.0
        if simplex_tree is not None:
            entropy_val = compute_persistent_entropy(simplex_tree)
            ent_by_dim = compute_entropy_by_dimension(simplex_tree)
            h0_e = ent_by_dim.get(0, 0.0)
            h1_e = ent_by_dim.get(1, 0.0)
            h2_e = ent_by_dim.get(2, 0.0)
            entropy_delta = h0_e + h1_e + h2_e - entropy_val
        components["entropy"] = min(1.0, entropy_val / 3.0) if entropy_val > 0 else 0.0

        h2_count = 0
        h2_growth = 0.0
        if simplex_tree is not None:
            try:
                h2_iv = simplex_tree.persistence_intervals_in_dimension(2)
                if h2_iv is not None and len(h2_iv) > 0:
                    h2_count = len(h2_iv)
                    persistences = h2_iv[:, 1] - h2_iv[:, 0]
                    long_h2 = float(np.sum(persistences > 1.0))
                    h2_growth = long_h2 / max(len(persistences), 1)
            except Exception:
                pass
        components["h2"] = min(1.0, h2_growth * 2.0) if h2_count > 0 else 0.0

        h1_count = 0
        h1_change = 0.0
        if simplex_tree is not None:
            try:
                h1_iv = simplex_tree.persistence_intervals_in_dimension(1)
                if h1_iv is not None and len(h1_iv) > 0:
                    h1_count = len(h1_iv)
                    avg_persist = float(np.mean(h1_iv[:, 1] - h1_iv[:, 0]))
                    h1_change = min(1.0, avg_persist * 0.5)
            except Exception:
                pass
        components["h1"] = h1_change

        density_anomaly = 0.0
        tetrahedra = mesh.tetrahedra
        if len(tetrahedra) >= 10:
            centroids = np.array([t.centroid for t in tetrahedra.values()])
            if len(centroids) > 10:
                try:
                    from scipy.spatial import cKDTree

                    tree = cKDTree(centroids)
                    k = min(6, len(centroids) - 1)
                    dists, _ = tree.query(centroids, k=k)
                    avg_dists = np.mean(dists[:, 1:], axis=1)
                    mean_d = float(np.mean(avg_dists))
                    std_d = float(np.std(avg_dists))
                    if std_d > 1e-12:
                        low_d = float(np.sum(avg_dists < mean_d - std_d * self._density_sigma))
                        density_anomaly = min(1.0, low_d / max(len(centroids) * 0.3, 1))
                except Exception:
                    counts = np.zeros(len(centroids))
                    for i in range(len(centroids)):
                        dists_i = np.linalg.norm(centroids - centroids[i], axis=1)
                        counts[i] = float(np.sum(dists_i < 0.3))
                    mean_c = float(np.mean(counts))
                    std_c = float(np.std(counts))
                    if std_c > 0:
                        high_c = float(np.sum(counts > mean_c + std_c * self._density_sigma))
                        density_anomaly = min(1.0, high_c / max(len(centroids) * 0.3, 1))
        components["density"] = density_anomaly

        staleness = 0.0
        elapsed = time.time() - self._last_integration_time
        if elapsed > 0:
            staleness = min(1.0, 1.0 - 0.5 ** (elapsed / self._staleness_halflife))
        components["staleness"] = staleness

        pressure = 0.0
        for key, weight in self._weights.items():
            pressure += components.get(key, 0.0) * weight

        self._last_components = components
        self._last_pressure = pressure

        record = {
            "pressure": pressure,
            "components": dict(components),
            "h2_count": h2_count,
            "h1_count": h1_count,
            "entropy": entropy_val,
            "staleness_seconds": elapsed,
            "total_tetra": len(tetrahedra),
        }
        with self._lock:
            self._history.append(record)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]
        return record

    @property
    def last_pressure(self) -> float:
        with self._lock:
            return self._last_pressure

    @property
    def last_components(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._last_components)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "last_pressure": self._last_pressure,
                "last_components": dict(self._last_components),
                "weights": dict(self._weights),
                "history_len": len(self._history),
            }

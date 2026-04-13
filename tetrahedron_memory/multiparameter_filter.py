"""
Multi-parameter Filter — Multi-dimensional query for TetraMem-XL.

Supports composable filtering along multiple dimensions:
  - Spatial: geometric proximity to query point
  - Temporal: recency of creation or last access
  - Density: local neighbor count around each tetrahedron
  - Weight: memory importance (integration-driven)
  - Label: semantic category matching
  - Topology: persistence score, connectivity depth

Each filter produces a normalized score in [0, 1].
Filters are combined with configurable weights into a composite score.

Design:
  - Filters are composable and independently configurable
  - Score normalization ensures no single dimension dominates
  - Supports both AND (hard filter) and OR (soft boost) modes
  - Results can be sorted by composite score or any individual dimension
  - Pipeline architecture: filter -> score -> rank -> return

Integration:
  - GeoMemoryBody.query_multiparam() delegates to MultiParameterQuery
  - TetraDreamCycle uses density + label diversity for walk seeding
  - Resolution pyramid uses spatial + density for cluster quality
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("tetramem.multiparam")


@dataclass
class FilterCriteria:
    name: str
    weight: float = 1.0
    hard_filter: bool = False
    min_score: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiParamResult:
    tetra_id: str
    composite_score: float
    individual_scores: Dict[str, float]
    content: str = ""
    centroid: Optional[np.ndarray] = None
    weight: float = 1.0
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiParameterQuery:
    """
    Composable multi-parameter query engine.

    Usage:
        mpq = MultiParameterQuery(mesh)
        mpq.add_filter("spatial", {"query_point": point, "max_distance": 2.0}, weight=0.4)
        mpq.add_filter("temporal", {"recency_seconds": 3600}, weight=0.2)
        mpq.add_filter("density", {"neighbor_radius": 0.5}, weight=0.2)
        mpq.add_filter("weight", {"min_weight": 0.5}, weight=0.2)
        results = mpq.execute(k=10)
    """

    def __init__(self, mesh: Any):
        self._mesh = mesh
        self._filters: List[FilterCriteria] = []
        self._density_cache: Optional[Dict[str, float]] = None
        self._density_cache_time: float = 0.0

    def add_filter(
        self,
        name: str,
        params: Dict[str, Any],
        weight: float = 1.0,
        hard_filter: bool = False,
        min_score: float = 0.0,
    ) -> "MultiParameterQuery":
        self._filters.append(FilterCriteria(
            name=name,
            weight=weight,
            hard_filter=hard_filter,
            min_score=min_score,
            params=params,
        ))
        return self

    def clear_filters(self) -> None:
        self._filters.clear()

    def execute(self, k: int = 10) -> List[MultiParamResult]:
        if not self._filters:
            return []

        tetrahedra = self._mesh.tetrahedra
        if not tetrahedra:
            return []

        candidates: List[MultiParamResult] = []

        density_scores = self._precompute_density()

        for tid, tetra in tetrahedra.items():
            scores: Dict[str, float] = {}
            passed = True

            for fc in self._filters:
                score = self._compute_filter_score(tid, tetra, fc, density_scores)
                scores[fc.name] = score

                if fc.hard_filter and score < fc.min_score:
                    passed = False
                    break

            if not passed:
                continue

            composite = self._compute_composite(scores)
            candidates.append(MultiParamResult(
                tetra_id=tid,
                composite_score=composite,
                individual_scores=scores,
                content=tetra.content,
                centroid=tetra.centroid.copy() if hasattr(tetra.centroid, "copy") else np.array(tetra.centroid),
                weight=tetra.weight,
                labels=list(tetra.labels),
                metadata=dict(tetra.metadata),
            ))

        candidates.sort(key=lambda r: r.composite_score, reverse=True)
        return candidates[:k]

    def execute_with_ids(
        self, candidate_ids: List[str], k: int = 10
    ) -> List[MultiParamResult]:
        if not self._filters or not candidate_ids:
            return []

        density_scores = self._precompute_density()
        candidates: List[MultiParamResult] = []

        for tid in candidate_ids:
            tetra = self._mesh.get_tetrahedron(tid)
            if tetra is None:
                continue

            scores: Dict[str, float] = {}
            passed = True

            for fc in self._filters:
                score = self._compute_filter_score(tid, tetra, fc, density_scores)
                scores[fc.name] = score

                if fc.hard_filter and score < fc.min_score:
                    passed = False
                    break

            if not passed:
                continue

            composite = self._compute_composite(scores)
            candidates.append(MultiParamResult(
                tetra_id=tid,
                composite_score=composite,
                individual_scores=scores,
                content=tetra.content,
                centroid=tetra.centroid.copy() if hasattr(tetra.centroid, "copy") else np.array(tetra.centroid),
                weight=tetra.weight,
                labels=list(tetra.labels),
                metadata=dict(tetra.metadata),
            ))

        candidates.sort(key=lambda r: r.composite_score, reverse=True)
        return candidates[:k]

    def _compute_filter_score(
        self,
        tid: str,
        tetra: Any,
        fc: FilterCriteria,
        density_scores: Dict[str, float],
    ) -> float:
        handler = getattr(self, f"_filter_{fc.name}", None)
        if handler is None:
            return 0.0
        return handler(tid, tetra, fc.params, density_scores)

    def _filter_spatial(
        self,
        tid: str,
        tetra: Any,
        params: Dict[str, Any],
        density_scores: Dict[str, float],
    ) -> float:
        query_point = params.get("query_point")
        if query_point is None:
            return 1.0

        max_dist = params.get("max_distance", 5.0)
        dist = float(np.linalg.norm(np.array(query_point) - tetra.centroid))

        if dist > max_dist:
            score = 0.0
        else:
            score = 1.0 - dist / max_dist

        return score

    def _filter_temporal(
        self,
        tid: str,
        tetra: Any,
        params: Dict[str, Any],
        density_scores: Dict[str, float],
    ) -> float:
        recency_seconds = params.get("recency_seconds", 3600.0)
        mode = params.get("mode", "creation")

        if mode == "access":
            ref_time = tetra.last_access_time
        else:
            ref_time = tetra.creation_time

        age = time.time() - ref_time
        if age < 0:
            return 1.0

        half_life = recency_seconds * 0.5
        score = 0.5 ** (age / max(half_life, 1e-6))
        return float(min(1.0, score))

    def _filter_density(
        self,
        tid: str,
        tetra: Any,
        params: Dict[str, Any],
        density_scores: Dict[str, float],
    ) -> float:
        if not density_scores:
            return 0.5

        d = density_scores.get(tid, 0.0)
        invert = params.get("invert", False)

        if invert:
            return 1.0 - d
        return d

    def _filter_weight(
        self,
        tid: str,
        tetra: Any,
        params: Dict[str, Any],
        density_scores: Dict[str, float],
    ) -> float:
        min_w = params.get("min_weight", 0.0)
        max_w = params.get("max_weight", 10.0)

        if tetra.weight < min_w or tetra.weight > max_w:
            return 0.0

        return min(1.0, tetra.weight / max(max_w, 1e-6))

    def _filter_label(
        self,
        tid: str,
        tetra: Any,
        params: Dict[str, Any],
        density_scores: Dict[str, float],
    ) -> float:
        required = params.get("required", [])
        preferred = params.get("preferred", [])
        penalized = params.get("penalized", [])

        if required:
            req_set = set(required)
            tetra_labels = set(tetra.labels)
            if not req_set.issubset(tetra_labels):
                return 0.0

        score = 0.5
        tetra_labels = set(tetra.labels)

        if preferred:
            overlap = len(tetra_labels & set(preferred))
            score += 0.5 * overlap / max(len(preferred), 1)

        if penalized:
            penalty = len(tetra_labels & set(penalized))
            score -= 0.3 * penalty / max(len(penalized), 1)

        return max(0.0, min(1.0, score))

    def _filter_topology(
        self,
        tid: str,
        tetra: Any,
        params: Dict[str, Any],
        density_scores: Dict[str, float],
    ) -> float:
        integration_boost = params.get("integration_boost", True)
        connectivity_weight = params.get("connectivity_weight", 0.3)

        score = 0.0

        if integration_boost:
            ic = tetra.integration_count
            score += min(0.5, ic * 0.05)

        if connectivity_weight > 0:
            neighbors = 0
            with self._mesh._lock:
                neighbors = len(self._mesh._face_neighbors(tid))
                neighbors += len(self._mesh._edge_neighbors(tid)) * 0.5
            score += min(0.5, neighbors * connectivity_weight * 0.1)

        return min(1.0, score)

    def _filter_access(
        self,
        tid: str,
        tetra: Any,
        params: Dict[str, Any],
        density_scores: Dict[str, float],
    ) -> float:
        min_access = params.get("min_access_count", 0)

        if tetra.access_count < min_access:
            return 0.0

        max_access = params.get("max_access_count", 1000)
        return min(1.0, tetra.access_count / max(max_access, 1))

    def _compute_composite(self, scores: Dict[str, float]) -> float:
        total_weight = 0.0
        weighted_sum = 0.0

        for fc in self._filters:
            s = scores.get(fc.name, 0.0)
            weighted_sum += s * fc.weight
            total_weight += fc.weight

        if total_weight <= 0:
            return 0.0

        return weighted_sum / total_weight

    def _precompute_density(self) -> Dict[str, float]:
        has_density = any(fc.name == "density" for fc in self._filters)
        if not has_density:
            return {}

        now = time.time()
        if self._density_cache is not None and (now - self._density_cache_time) < 5.0:
            return self._density_cache

        tetrahedra = self._mesh.tetrahedra
        if len(tetrahedra) < 2:
            self._density_cache = {}
            self._density_cache_time = now
            return {}

        density_radius = 1.0
        for fc in self._filters:
            if fc.name == "density":
                density_radius = fc.params.get("neighbor_radius", 1.0)
                break

        centroids = np.array([t.centroid for t in tetrahedra.values()])
        ids = list(tetrahedra.keys())
        n = len(ids)

        density_map: Dict[str, float] = {}
        if n >= 10:
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(centroids)
                counts = tree.query_ball_point(centroids, r=density_radius)
                raw = np.array([len(c) for c in counts], dtype=float)
            except ImportError:
                raw = np.zeros(n)
                for i in range(n):
                    dists = np.linalg.norm(centroids - centroids[i], axis=1)
                    raw[i] = float(np.sum(dists < density_radius))
        else:
            raw = np.zeros(n)
            for i in range(n):
                dists = np.linalg.norm(centroids - centroids[i], axis=1)
                raw[i] = float(np.sum(dists < density_radius))

        max_raw = float(np.max(raw)) if len(raw) > 0 and np.max(raw) > 0 else 1.0
        for i, tid in enumerate(ids):
            density_map[tid] = float(raw[i] / max_raw)

        self._density_cache = density_map
        self._density_cache_time = now
        return density_map

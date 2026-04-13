"""
FastAPI REST API router for TetraMem-XL.

All FastAPI / uvicorn / prometheus_client imports are guarded so the module
imports cleanly even when those packages are not installed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .core import GeoMemoryBody, QueryResult
from .monitoring import (
    ASSOCIATE_COUNTER,
    NODE_COUNT_GAUGE,
    QUERY_COUNTER,
    SELF_ORGANIZE_COUNTER,
    STORE_COUNTER,
    generate_metrics,
    increment_counter,
    record_error,
    set_gauge,
)

_fastapi_available = False
try:
    from fastapi import FastAPI, HTTPException, Response
    from pydantic import BaseModel, Field

    _fastapi_available = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    HTTPException = None  # type: ignore[assignment, misc]
    Response = None  # type: ignore[assignment, misc]

    class BaseModel:  # type: ignore[no-redef]
        """Placeholder when pydantic is not installed."""

        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Field:  # type: ignore[no-redef]
        """Placeholder when pydantic is not installed."""

        def __init__(self, default: Any = ..., **kwargs: Any) -> None:
            self.default = default


class StoreRequest(BaseModel):
    content: str = Field(..., description="Memory content text")
    labels: Optional[List[str]] = Field(default=None, description="Index labels")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Extra metadata")
    weight: float = Field(default=1.0, ge=0.1, le=10.0, description="Initial weight")


class StoreResponse(BaseModel):
    id: str


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    k: int = Field(default=5, ge=1, le=100, description="Number of results")
    use_persistence: bool = Field(default=True, description="Use PH scoring")


class QueryResultItem(BaseModel):
    id: str
    content: str
    distance: float
    persistence_score: float
    weight: float
    labels: List[str]


class QueryResponse(BaseModel):
    results: List[QueryResultItem]


class AssociateResponse(BaseModel):
    associations: List[Dict[str, Any]]


class SelfOrganizeResponse(BaseModel):
    stats: Dict[str, Any]


class StatsResponse(BaseModel):
    total_memories: int
    total_labels: int
    avg_weight: float
    dimension: int
    precision: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float


class DreamRequest(BaseModel):
    force: bool = Field(default=False, description="Force dream regardless of entropy")


class DreamResponse(BaseModel):
    result: Dict[str, Any]


class ClosedLoopRequest(BaseModel):
    context: str = Field(default="", description="Context for the cycle")
    k: int = Field(default=5, ge=1, le=50, description="Number of memories to recall")
    force_dream: bool = Field(default=False, description="Force dream phase")


class ClosedLoopResponse(BaseModel):
    result: Dict[str, Any]


class BatchStoreRequest(BaseModel):
    items: List[StoreRequest]


class BatchStoreResponse(BaseModel):
    ids: List[str]


class WeightUpdateRequest(BaseModel):
    memory_id: str = Field(..., description="Memory ID to update")
    delta: float = Field(..., description="Weight change")
    use_ema: bool = Field(default=True, description="Use exponential moving average")
    alpha: float = Field(default=0.1, ge=0.01, le=1.0, description="EMA smoothing factor")


class WeightUpdateResponse(BaseModel):
    memory_id: str
    status: str


class PersistResponse(BaseModel):
    status: str
    snapshot_name: str


class ConsistencyResponse(BaseModel):
    status: Dict[str, Any]


class MultiParamQueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    k: int = Field(default=10, ge=1, le=100)
    spatial_weight: float = Field(default=0.4)
    temporal_weight: float = Field(default=0.2)
    density_weight: float = Field(default=0.15)
    weight_weight: float = Field(default=0.15)
    topology_weight: float = Field(default=0.1)
    recency_seconds: float = Field(default=3600.0)
    max_distance: float = Field(default=5.0)
    labels_required: Optional[List[str]] = Field(default=None)
    labels_preferred: Optional[List[str]] = Field(default=None)


class PyramidQueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    k: int = Field(default=5, ge=1, le=100)
    level: int = Field(default=-1, description="Pyramid level (-1 = auto)")


class ZigzagResponse(BaseModel):
    result: Dict[str, Any]


class GetMemoryResponse(BaseModel):
    id: str
    content: str
    weight: float
    labels: List[str]
    metadata: Dict[str, Any]
    timestamp: float
    access_count: int = 0


class UpdateMemoryRequest(BaseModel):
    content: Optional[str] = Field(default=None, description="New content text")
    labels: Optional[List[str]] = Field(default=None, description="New labels (replaces all)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata to merge")
    weight: Optional[float] = Field(default=None, ge=0.1, le=10.0, description="New weight")


class UpdateMemoryResponse(BaseModel):
    memory_id: str
    updated: bool


class LabelQueryRequest(BaseModel):
    label: str = Field(..., description="Label to search for")
    k: int = Field(default=20, ge=1, le=100, description="Max results")


class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., description="List of query texts")
    k: int = Field(default=5, ge=1, le=100, description="Results per query")


class BatchQueryResponse(BaseModel):
    results: List[List[Dict[str, Any]]]


class PyramidBuildResponse(BaseModel):
    result: Dict[str, Any]


def create_app(
    memory: Optional[GeoMemoryBody] = None,
    dimension: int = 3,
    precision: str = "fast",
) -> Any:
    """Build and return a ``FastAPI`` application.

    If *memory* is ``None`` a fresh ``GeoMemoryBody`` is created with the
    given *dimension* and *precision*.

    Raises ``ImportError`` when ``fastapi`` is not installed.
    """
    if not _fastapi_available:
        raise ImportError(
            "FastAPI is required for the REST API. "
            "Install with: pip install tetrahedron-memory[api]"
        )

    if memory is None:
        memory = GeoMemoryBody(dimension=dimension, precision=precision)

    app: Any = FastAPI(
        title="TetraMem-XL",
        version="2.2.0",
        description="Pure geometric-driven AI memory system REST API",
    )

    _start_time: float = time.time()

    @app.middleware("http")
    async def add_cors_and_timing(request: Any, call_next: Any) -> Any:
        import asyncio

        response = (
            await call_next(request)
            if not _fastapi_available
            else await _cors_handler(request, call_next)
        )
        return response

    async def _cors_handler(request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

    @app.options("/{path:path}")
    async def options_handler(path: str) -> Any:
        from starlette.responses import JSONResponse

        resp = JSONResponse({})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "*"
        return resp

    @app.post("/api/v1/store", response_model=StoreResponse)
    def store(req: StoreRequest) -> StoreResponse:
        try:
            node_id = memory.store(
                content=req.content,
                labels=req.labels,
                metadata=req.metadata,
                weight=req.weight,
            )
            increment_counter(STORE_COUNTER)
            set_gauge(NODE_COUNT_GAUGE, len(memory._nodes))
            return StoreResponse(id=node_id)
        except Exception as exc:
            record_error("store")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        try:
            results: List[QueryResult] = memory.query(
                query_text=req.query,
                k=req.k,
                use_persistence=req.use_persistence,
            )
            increment_counter(QUERY_COUNTER)
            items: List[QueryResultItem] = []
            for r in results:
                items.append(
                    QueryResultItem(
                        id=r.node.id,
                        content=r.node.content,
                        distance=r.distance,
                        persistence_score=r.persistence_score,
                        weight=r.node.weight,
                        labels=r.node.labels,
                    )
                )
            return QueryResponse(results=items)
        except Exception as exc:
            record_error("query")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/v1/associate/{memory_id}", response_model=AssociateResponse)
    def associate(memory_id: str, max_depth: int = 2) -> AssociateResponse:
        try:
            raw = memory.associate(memory_id=memory_id, max_depth=max_depth)
            increment_counter(ASSOCIATE_COUNTER)
            items: List[Dict[str, Any]] = []
            for node, score, assoc_type in raw:
                items.append(
                    {
                        "id": node.id,
                        "content": node.content,
                        "score": score,
                        "type": assoc_type,
                        "weight": node.weight,
                        "labels": node.labels,
                    }
                )
            return AssociateResponse(associations=items)
        except Exception as exc:
            record_error("associate")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/self-organize", response_model=SelfOrganizeResponse)
    def self_organize() -> SelfOrganizeResponse:
        try:
            stats = memory.self_organize()
            increment_counter(SELF_ORGANIZE_COUNTER)
            set_gauge(NODE_COUNT_GAUGE, len(memory._nodes))
            return SelfOrganizeResponse(stats=stats)
        except Exception as exc:
            record_error("self_organize")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/v1/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        try:
            s = memory.get_statistics()
            return StatsResponse(
                total_memories=s["total_memories"],
                total_labels=s["total_labels"],
                avg_weight=float(s["avg_weight"]),
                dimension=s["dimension"],
                precision=s["precision"],
            )
        except Exception as exc:
            record_error("stats")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            uptime_seconds=time.time() - _start_time,
        )

    @app.get("/metrics")
    def metrics() -> Any:
        body = generate_metrics()
        return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.post("/api/v1/dream", response_model=DreamResponse)
    def dream(req: DreamRequest) -> DreamResponse:
        try:
            if not memory._use_mesh:
                return DreamResponse(result={"status": "no_mesh"})
            from .tetra_dream import TetraDreamCycle

            dc = TetraDreamCycle(memory._mesh, cycle_interval=999999)
            result = dc.trigger_now()
            return DreamResponse(result=result)
        except Exception as exc:
            record_error("dream")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/closed-loop", response_model=ClosedLoopResponse)
    def closed_loop(req: ClosedLoopRequest) -> ClosedLoopResponse:
        try:
            from .closed_loop import ClosedLoopEngine

            engine = ClosedLoopEngine(memory)
            result = engine.run_cycle(context=req.context, k=req.k, force_dream=req.force_dream)
            return ClosedLoopResponse(result=result)
        except Exception as exc:
            record_error("closed_loop")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/batch-store", response_model=BatchStoreResponse)
    def batch_store(req: BatchStoreRequest) -> BatchStoreResponse:
        try:
            items = [
                {
                    "content": r.content,
                    "labels": r.labels,
                    "metadata": r.metadata,
                    "weight": r.weight,
                }
                for r in req.items
            ]
            ids = memory.store_batch(items)
            increment_counter(STORE_COUNTER)
            set_gauge(NODE_COUNT_GAUGE, len(memory._nodes))
            return BatchStoreResponse(ids=ids)
        except Exception as exc:
            record_error("batch_store")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/weight-update", response_model=WeightUpdateResponse)
    def weight_update(req: WeightUpdateRequest) -> WeightUpdateResponse:
        try:
            memory.update_weight(
                memory_id=req.memory_id,
                delta=req.delta,
                use_ema=req.use_ema,
                alpha=req.alpha,
            )
            return WeightUpdateResponse(memory_id=req.memory_id, status="ok")
        except Exception as exc:
            record_error("weight_update")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/persist", response_model=PersistResponse)
    def persist() -> PersistResponse:
        try:
            memory.flush_persistence()
            return PersistResponse(status="ok", snapshot_name=memory._bucket_id)
        except Exception as exc:
            record_error("persist")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/v1/consistency", response_model=ConsistencyResponse)
    def consistency() -> ConsistencyResponse:
        return ConsistencyResponse(status=memory.get_consistency_status())

    @app.post("/api/v1/query-multiparam", response_model=QueryResponse)
    def query_multiparam(req: MultiParamQueryRequest) -> QueryResponse:
        try:
            results = memory.query_multiparam(
                query_text=req.query,
                k=req.k,
                spatial_weight=req.spatial_weight,
                temporal_weight=req.temporal_weight,
                density_weight=req.density_weight,
                weight_weight=req.weight_weight,
                topology_weight=req.topology_weight,
                recency_seconds=req.recency_seconds,
                max_distance=req.max_distance,
                labels_required=req.labels_required,
                labels_preferred=req.labels_preferred,
            )
            items = [
                QueryResultItem(
                    id=r.node.id,
                    content=r.node.content,
                    distance=r.distance,
                    persistence_score=r.persistence_score,
                    weight=r.node.weight,
                    labels=r.node.labels,
                )
                for r in results
            ]
            return QueryResponse(results=items)
        except Exception as exc:
            record_error("query_multiparam")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/query-pyramid", response_model=QueryResponse)
    def query_pyramid(req: PyramidQueryRequest) -> QueryResponse:
        try:
            results = memory.query_pyramid(
                query_text=req.query,
                k=req.k,
                level=req.level,
            )
            items = [
                QueryResultItem(
                    id=r.node.id,
                    content=r.node.content,
                    distance=r.distance,
                    persistence_score=r.persistence_score,
                    weight=r.node.weight,
                    labels=r.node.labels,
                )
                for r in results
            ]
            return QueryResponse(results=items)
        except Exception as exc:
            record_error("query_pyramid")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/build-pyramid", response_model=PyramidBuildResponse)
    def build_pyramid() -> PyramidBuildResponse:
        return PyramidBuildResponse(result=memory.build_pyramid())

    @app.post("/api/v1/zigzag-snapshot", response_model=ZigzagResponse)
    def zigzag_snapshot() -> ZigzagResponse:
        result = memory.record_zigzag_snapshot()
        if result is None:
            return ZigzagResponse(result={"status": "not_mesh_mode"})
        return ZigzagResponse(result=result)

    @app.get("/api/v1/zigzag-status", response_model=ZigzagResponse)
    def zigzag_status() -> ZigzagResponse:
        return ZigzagResponse(result=memory.get_zigzag_status())

    @app.get("/api/v1/predict-topology", response_model=ZigzagResponse)
    def predict_topology() -> ZigzagResponse:
        return ZigzagResponse(result=memory.predict_topology())

    @app.get("/api/v1/dynamic-barcode", response_model=ZigzagResponse)
    def dynamic_barcode(dimension: int = -1) -> ZigzagResponse:
        return ZigzagResponse(result=memory.get_dynamic_barcode(dimension))

    @app.get("/api/v1/health/topology", response_model=ZigzagResponse)
    def topology_health() -> ZigzagResponse:
        try:
            health: Dict[str, Any] = {
                "mesh_mode": memory._use_mesh,
                "total_memories": len(memory._nodes),
                "tetrahedra": len(memory._mesh.tetrahedra) if memory._use_mesh else 0,
            }

            if memory._use_mesh:
                st = memory._mesh.compute_ph()
                if st is not None:
                    from .persistent_entropy import compute_persistent_entropy

                    h0 = st.persistence_intervals_in_dimension(0)
                    h1 = st.persistence_intervals_in_dimension(1)
                    h2 = st.persistence_intervals_in_dimension(2)
                    health["persistent_entropy"] = compute_persistent_entropy(st)
                    health["h0_count"] = len(h0) if h0 is not None else 0
                    health["h1_count"] = len(h1) if h1 is not None else 0
                    health["h2_count"] = len(h2) if h2 is not None else 0

                    if h2 is not None and len(h2) > 0:
                        persistences = h2[:, 1] - h2[:, 0]
                        health["h2_long_lived"] = int(np.sum(persistences > 1.0))

                health["zigzag_stability"] = memory._zigzag_tracker.get_zigzag_stability()
                health["emergence_threshold"] = memory._adaptive_threshold.get_status()
                health["pyramid_status"] = memory._pyramid.get_status()

            if memory._consistency is not None:
                health["consistency"] = memory._consistency.get_health()

            return ZigzagResponse(result=health)
        except Exception as exc:
            record_error("topology_health")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/v1/memory/{memory_id}")
    def get_memory(memory_id: str) -> Any:
        result = memory.get(memory_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
        return result

    @app.put("/api/v1/memory/{memory_id}")
    def update_memory(memory_id: str, req: UpdateMemoryRequest) -> UpdateMemoryResponse:
        try:
            success = memory.update(
                memory_id=memory_id,
                content=req.content,
                labels=req.labels,
                metadata=req.metadata,
                weight=req.weight,
            )
            if not success:
                raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
            return UpdateMemoryResponse(memory_id=memory_id, updated=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/v1/query-by-label")
    def query_by_label(req: LabelQueryRequest) -> QueryResponse:
        try:
            nodes = memory.query_by_label(req.label, k=req.k)
            items = [
                QueryResultItem(
                    id=n.id,
                    content=n.content,
                    distance=0.0,
                    persistence_score=0.0,
                    weight=n.weight,
                    labels=n.labels,
                )
                for n in nodes
            ]
            return QueryResponse(results=items)
        except Exception as exc:
            record_error("query_by_label")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/query-batch", response_model=BatchQueryResponse)
    def query_batch(req: BatchQueryRequest) -> BatchQueryResponse:
        try:
            all_results = memory.query_batch(req.queries, k=req.k)
            batch = []
            for results in all_results:
                batch.append(
                    [
                        {
                            "id": r.node.id,
                            "content": r.node.content,
                            "distance": r.distance,
                            "persistence_score": r.persistence_score,
                            "weight": r.node.weight,
                            "labels": r.node.labels,
                        }
                        for r in results
                    ]
                )
            return BatchQueryResponse(results=batch)
        except Exception as exc:
            record_error("query_batch")
            raise HTTPException(status_code=500, detail=str(exc))

    return app

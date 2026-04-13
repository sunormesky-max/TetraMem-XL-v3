# TetraMem-XL Deployment Guide

## Prerequisites

- Python 3.8+
- Required: numpy, gudhi, scipy
- Optional: ray (distributed), fastapi+uvicorn (API), prometheus-client (monitoring), boto3 (S3)

## Step 1: Installation

```bash
# Basic
pip install tetrahedron-memory

# Production (all features)
pip install tetrahedron-memory[all]

# Or from source
git clone https://github.com/sunormesky-max/sunorm-space-memory.git
cd sunorm-space-memory
pip install -e ".[all]"
```

## Step 2: Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TETRAMEM_STORAGE` | `~/.tetramem_data` | Storage directory for persistence |
| `TETRAMEM_EXPORT` | `~/tetramem_export.md` | Export file path |
| `RAY_WORKERS` | `4` | Number of Ray workers |
| `API_PORT` | `8000` | REST API port |
| `API_WORKERS` | `4` | Uvicorn worker count |

### Storage Setup

```bash
mkdir -p /data/tetramem
export TETRAMEM_STORAGE=/data/tetramem
```

## Step 3: One-Click Deployment

```bash
bash deploy.sh
```

This script:
1. Runs production tests (413+ tests)
2. Starts Ray cluster (if available)
3. Launches API server with persistence
4. Performs health check
5. Starts Prometheus monitoring (if available)

### Manual Deployment

```bash
# Start API with persistence
python start_api_persisted.py

# Or directly with uvicorn
uvicorn start_api_persisted:app --host 0.0.0.0 --port 8000 --workers 4
```

## Step 4: Monitoring Configuration

### Prometheus

Create `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'tetramem'
    scrape_interval: 10s
    static_configs:
      - targets: ['localhost:8000']
```

Start Prometheus:

```bash
prometheus --config.file=prometheus.yml --storage.tsdb.path=/data/tetramem/prometheus
```

### Grafana

Import the built-in dashboard:

```python
from tetrahedron_memory.monitoring import get_grafana_dashboard_json
print(get_grafana_dashboard_json())
```

15 panels covering:
- Store/Query throughput
- Memory node count
- Error rate by operation
- Weight distribution
- Store/Query latency (p50/p95/p99)
- Persistent entropy
- Integration & Dream cycles

### Alert Rules

4 built-in alert rules:

| Alert | Condition | Severity |
|-------|-----------|----------|
| EntropySpike | `tetramem_persistent_entropy > 4.0` for 2m | Warning |
| HighErrorRate | `error_rate > 2%` for 1m | Critical |
| StoreLatencyHigh | `store p99 > 100ms` for 5m | Warning |
| QueryLatencyHigh | `query p99 > 50ms` for 5m | Warning |

```python
from tetrahedron_memory.monitoring import get_alert_rules
print(get_alert_rules())
```

## Step 5: Verification

### Health Check

```bash
# Basic health
curl http://localhost:8000/api/v1/health

# Topology health (entropy, H2 voids, zigzag stability)
curl http://localhost:8000/api/v1/health/topology

# Consistency status
curl http://localhost:8000/api/v1/consistency
```

### Store & Query

```bash
# Store
curl -X POST http://localhost:8000/api/v1/store \
  -H "Content-Type: application/json" \
  -d '{"content": "test memory", "labels": ["test"], "weight": 1.0}'

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "k": 5}'

# Multi-parameter query
curl -X POST http://localhost:8000/api/v1/query-multiparam \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "k": 10, "labels_required": ["test"]}'
```

### Dream Cycle

```bash
curl -X POST http://localhost:8000/api/v1/dream \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Fault Recovery

### Node Failure Recovery

1. **Detection**: VectorClock detects stale versions via `get_staleness()`
2. **Read Repair**: `read_repair_multi()` auto-syncs from majority
3. **Compensation**: Failed operations logged in CompensationLog
4. **Retry**: `retry_pending_compensations()` on node recovery

### Data Recovery

```bash
# From Parquet snapshot
python -c "
from tetrahedron_memory.core import GeoMemoryBody
body = GeoMemoryBody(dimension=3, persistence=...)
count = body.load_from_persistence()
print(f'Recovered {count} memories')
"

# Or use CLI
tetramem stats
tetramem status
```

### Version Conflict Resolution

Automatic conflict resolution priority:
1. Higher version wins
2. Newer timestamp wins (versions equal)
3. Conflicts logged in `get_conflict_history()`

## CLI Usage

```bash
# Store
tetramem store "my memory" -l topic1 topic2

# Query
tetramem query "search" -k 5

# Multi-parameter query
tetramem mquery "filtered" --labels topic1 -k 10

# Dream cycle
tetramem dream -n 3

# Build & query pyramid
tetramem build-pyramid
tetramem pyquery "fast search" -k 10

# Topology tracking
tetramem zigzag
tetramem predict

# Status
tetramem status
tetramem stats
```

## Distributed Deployment (Ray)

```bash
# Start Ray cluster
ray up cluster.yaml --yes

# Submit distributed job
ray job submit -- python -m tetrahedron_memory.partitioning --num-buckets 256

# Monitor
ray dashboard http://<head-ip>:8265
```

## Performance Tuning

| Parameter | Default | Tuning |
|-----------|---------|--------|
| `precision` | `"fast"` | Use `"safe"` for correctness, `"fast"` for speed |
| `auto_emerge_interval` | `0.0` (disabled) | Set to `300.0` for auto-emergence every 5 min |
| `_persist_flush_interval` | `50` | Increase for fewer disk writes |
| `max_dream_tetra` | `50` | Increase for more dream memories |
| `walk_steps` | `12` | Increase for deeper dream exploration |

## Stopping

```bash
# Stop API
kill $(lsof -ti:8000)

# Stop Ray
ray stop

# Or use Ctrl+C on the deploy.sh process
```

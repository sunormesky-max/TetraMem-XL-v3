# TetraMem-XL Architecture

## 概念

TetraMem-XL 是一个**纯几何驱动、拓扑自组织**的 AI 记忆系统。核心思想：

1. **记忆附加在四面体上** — 每条记忆是一个 3-simplex（四面体），不是向量
2. **纯几何检索** — 沿共享面/边/顶点导航，零向量嵌入
3. **永恒法则** — 不删除、不遗忘，所有记忆通过整合永存
4. **时间法则** — filtration 驱动整合优先级，老记忆优先被整合
5. **梦境法则** — 自涌现闭环：随机游走→发现远距关联→融合→新记忆
6. **分布式可扩展** — 空间分区 + Ghost Cell 跨桶拓扑

## 架构总览

```
┌───────────────────────────────────────────────────────┐
│                  TetraDistributedController            │
│  (统一 API: store / query / dream / self-org / balance) │
├───────────────────────────────────────────────────────┤
│                   TetraMeshRouter                      │
│  (空间分区路由 + 跨桶拓扑导航 + 分布式 dream/self-org)    │
├──────────┬──────────┬──────────┬──────────────────────┤
│TetraBucket│TetraBucket│TetraBucket│  ...               │
│(TetraMesh)│(TetraMesh)│(TetraMesh)│                    │
│  + Ghost  │  + Ghost  │  + Ghost  │                    │
│   Cells   │   Cells   │   Cells   │                    │
└──────────┴──────────┴──────────┴──────────────────────┘
```

## 核心模块

### MemoryTetrahedron (tetra_mesh.py)
记忆的基本单元。每条记忆是一个四面体：
- `content` — 记忆文本
- `vertex_indices` — 4个顶点（纯几何）
- `centroid` — 质心
- `labels` — 语义标签
- `weight` — 权重（整合催化剂，永不衰减）
- `secondary_memories` — 子记忆列表（密度增长）
- `filtration()` — 时间法则：spatial_alpha × integration_bonus + time_lambda × age
- `integrate_secondary()` — 抽象重组：内容融合 + 标签合并 + 主题提取 + 来源溯源

### TetraMesh (tetra_mesh.py)
四面体网格。管理所有四面体的存储、连接、查询：
- **store()** — 新四面体附着到边界面，网格向外生长
- **query_topological()** — 纯拓扑查询：先按结构选种子，再 BFS 沿面/边/顶点导航
- **navigate_topology()** — BFS 拓扑导航，返回 (id, connection_type, hop_distance)
- **abstract_reorganize()** — 批量抽象重组：扫描密四面体 + 跨节点概念融合
- **edge_contraction()** — 边收缩合并两个相邻四面体

### TetraDreamCycle (tetra_dream.py)
梦境闭环实现：
- 随机游走→拓扑聚类→跨簇融合→插入新四面体→触发自组织
- **DreamStore** — 独立梦境注册表，支持来源追踪和质量统计
- **DreamRecord** — 每条梦的完整溯源：source_tetra_ids, fusion_quality, entropy_delta
- **DreamProtocol** — 三阶段协议：THINK（分析）→ EXECUTE（合成）→ REFLECT（评估）

### DreamProtocol (tetra_dream.py)
结构化梦境认知闭环：
```
THINK    → 分析来源，选择策略（surface/deepen/bridge）
EXECUTE  → 生成合成内容（自定义 LLM 回调或默认）
REFLECT  → 评估质量（7维拓扑评分），accept/reject
```
- `think_fn(inputs) → analysis` — 可替换为 LLM 分析
- `execute_fn(inputs, analysis) → content` — 可替换为 LLM 生成
- `reflect_fn(inputs, content) → quality` — 默认使用 fusion_quality_score v2

### TetraSelfOrganizer (tetra_self_org.py)
自组织循环：
- H2洞穴→插入排斥点（洞穴增长）
- 短H0间隔→边收缩合并
- 低热记忆→整合催化剂
- 持久熵收敛检测

### TetraMeshRouter (tetra_router.py)
分布式路由：
- **route_store()** — 按 centroid 分配到空间桶
- **route_query()** — 多桶查询 + 合并排序
- **navigate_cross_bucket()** — 跨桶拓扑导航（Ghost Cell 桥接）
- **distributed_dream()** — 逐桶 dream + 跨桶 cross-pollination
- **distributed_self_org()** — 并行自组织 + Ghost Cell 清理
- **verify_ghost_cells()** — 一致性校验（版本号 + 来源验证）

### Ghost Cell (partitioning.py)
跨桶拓扑桥接，带版本化一致性：
- `version` / `source_version` — 版本追踪
- `is_stale` — 版本不匹配检测
- `needs_verification` — 定时校验触发
- `verify()` — 与源桶同步版本和权重
- `invalidate_ghost_for()` — 变异触发失效传播

## 评分体系

### fusion_quality_score v2 — 7维拓扑感知评分

| 维度 | 权重 | 含义 |
|------|------|------|
| Source diversity | 0-0.15 | 来源数量多样性 |
| Label diversity | 0-0.10 | 标签独特性 |
| Weight balance | 0-0.10 | 权重均衡度 |
| Content richness | 0-0.15 | 合成内容丰富度 |
| **Topological connectivity** | **0-0.20** | 共享标签作为拓扑连接代理 |
| **Source depth** | **0-0.15** | integration_count 加权 |
| **Centroid dispersion** | **0-0.15** | 空间分散度（桥接价值） |

## 永恒 + 自涌现闭环

```
    ┌─── Store（新记忆附着到边界面）────┐
    │                                    │
    ▼                                    │
  TetraMesh                             │
    │                                    │
    ├─→ query_topological（纯拓扑检索）   │
    │                                    │
    ├─→ abstract_reorganize（抽象重组）──┤
    │     └→ 内容融合 + 标签合并 + 溯源  │
    │                                    │
    ├─→ Dream Cycle（梦境自涌现）───────┤
    │     ├→ THINK: 分析来源            │
    │     ├→ EXECUTE: 合成新记忆        │
    │     ├→ REFLECT: 评估质量          │
    │     └→ 插入新四面体               │
    │                                    │
    ├─→ Self-Organization（拓扑自修复）─┤
    │     ├→ H2洞穴 → 排斥点            │
    │     ├→ 短H0 → 边收缩              │
    │     └→ 持久熵收敛                 │
    │                                    │
    └──────→ 永恒循环（不删除，只整合）──┘
```

**核心理念**：记忆永不衰减。低活跃记忆不是被遗忘，而是被整合到更高阶的抽象概念中。梦境系统自发发现远距关联，自组织维持拓扑健康。

## 使用示例

```python
from tetrahedron_memory.tetra_distributed import TetraDistributedController
import numpy as np

ctrl = TetraDistributedController(num_buckets=4, use_ray=False)
ctrl.initialize()

# 存储记忆
bid, tid = ctrl.store(
    "Machine learning fundamentals",
    seed_point=np.array([0.5, 0.0, 0.0]),
    labels=["ai", "ml"],
    weight=1.5,
)

# 拓扑查询（无向量嵌入）
results = ctrl.query(np.array([0.5, 0.0, 0.0]), k=5)

# 梦境周期
dream_stats = ctrl.run_dream_cycle()

# 自组织
org_stats = ctrl.run_self_organization()

# 统计
stats = ctrl.get_statistics()
```

## 文件结构

```
tetrahedron_memory/
├── __init__.py              # 包导出
├── tetra_mesh.py            # MemoryTetrahedron + TetraMesh（核心）
├── tetra_dream.py           # TetraDreamCycle + DreamProtocol + DreamStore
├── tetra_self_org.py        # TetraSelfOrganizer
├── tetra_router.py          # TetraBucket + TetraMeshRouter（分布式路由）
├── tetra_distributed.py     # TetraDistributedController（统一API）
├── partitioning.py          # BoundingBox + GhostCell + Octree + SpatialBucketRouter
├── persistence.py           # Parquet + S3 持久化
├── global_coarse_mesh.py    # 全局粗网格拓扑校正
├── closed_loop.py           # 闭环熵控制
├── emergence.py             # 自涌现机制
├── circuit_breaker.py       # 熔断器
├── resolution_pyramid.py    # 分辨率金字塔
├── persistent_entropy.py    # 持久熵
├── multiparameter_filter.py # 多参数过滤
├── zigzag_persistence.py    # 锯齿持久同调
├── eternity_audit.py        # 永恒审计
└── ...
```

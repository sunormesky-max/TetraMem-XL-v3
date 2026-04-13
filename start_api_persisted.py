"""
TetraMem-XL API 启动脚本（带持久化支持）
"""

import os
from pathlib import Path

from tetrahedron_memory.router import create_app
from tetrahedron_memory.core import GeoMemoryBody, MemoryNode
from tetrahedron_memory.persistence import MemoryPersistence

# 配置
STORAGE_DIR = os.environ.get("TETRAMEM_STORAGE", "./tetramem_data")
DATA_FILE = Path(STORAGE_DIR) / "nodes.json"


def create_persisted_app():
    """创建带持久化的 API 应用"""

    # 确保存储目录存在
    Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)

    # 创建记忆体
    memory = GeoMemoryBody(dimension=3, precision="fast")

    # 加载已有记忆
    persistence = MemoryPersistence(storage_dir=STORAGE_DIR)
    saved_nodes = persistence.load_nodes()

    if saved_nodes:
        print(f"[TetraMem] 从 {DATA_FILE} 加载 {len(saved_nodes)} 条记忆...")
        for node_id, node in saved_nodes.items():
            if memory._use_mesh:
                memory._mesh.store(
                    content=node.content,
                    seed_point=node.geometry,
                    labels=node.labels,
                    metadata=node.metadata,
                    weight=node.weight,
                )
                memory._mesh_node_map[node_id] = node.geometry
            else:
                memory._nodes_dict[node_id] = node
                for label in node.labels:
                    memory._label_index_legacy[label].add(node_id)
        memory._needs_rebuild = True
        print(f"[TetraMem] 加载完成，共 {len(memory._nodes)} 条记忆")
    else:
        print(f"[TetraMem] 没有找到已有记忆，启动空记忆体")

    # 创建 API 应用
    app = create_app(memory=memory, dimension=3, precision="fast")

    # 添加关闭时保存的钩子
    @app.on_event("shutdown")
    async def shutdown_event():
        print(f"[TetraMem] 保存 {len(memory._nodes)} 条记忆到 {DATA_FILE}...")
        persistence.save_nodes(memory._nodes)
        print(f"[TetraMem] 保存完成")

    # 添加存储后自动保存的中间件
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware

    class AutoSaveMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            # 如果是存储请求，自动保存
            if request.url.path == "/api/v1/store" and request.method == "POST":
                persistence.save_nodes(memory._nodes)
            return response

    app.add_middleware(AutoSaveMiddleware)

    return app


# 创建应用
app = create_persisted_app()

if __name__ == "__main__":
    import uvicorn

    print(f"[TetraMem] 启动 API 服务，存储目录: {STORAGE_DIR}")
    uvicorn.run(app, host="127.0.0.1", port=8000)

import logging
import os
import secrets
import time
import uuid
from datetime import datetime
from collections import deque
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from mem0 import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

MIN_KEY_LENGTH = 16

if not ADMIN_API_KEY:
    logging.warning(
        "ADMIN_API_KEY not set - API endpoints are UNSECURED! "
        "Set ADMIN_API_KEY environment variable for production use."
    )
else:
    if len(ADMIN_API_KEY) < MIN_KEY_LENGTH:
        logging.warning(
            "ADMIN_API_KEY is shorter than %d characters - consider using a longer key for production.",
            MIN_KEY_LENGTH,
        )
    logging.info("API key authentication enabled")

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "memgraph")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "mem0graph")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_COLLECTION_NAME,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    },
    "llm": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "temperature": 0.2, "model": "gpt-4.1-nano-2025-04-14"}},
    "embedder": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"}},
    "history_db_path": HISTORY_DB_PATH,
}


MEMORY_INSTANCE = Memory.from_config(DEFAULT_CONFIG)

# 请求日志缓存（内存中保留最近200条）
REQUEST_LOGS: deque = deque(maxlen=200)

app = FastAPI(
    title="Mem0 REST APIs",
    description=(
        "A REST API for managing and searching memories for your AI Agents and Apps.\n\n"
        "## Authentication\n"
        "When the ADMIN_API_KEY environment variable is set, all endpoints require "
        "the `X-API-Key` header for authentication."
    ),
    version="1.0.0",
)

# 添加 CORS 中间件，允许 dashboard 前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个请求的方法、路径、状态码、耗时等信息。"""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000)

    log_entry = {
        "id": str(uuid.uuid4()),
        "method": request.method,
        "endpoint": str(request.url.path),
        "status": response.status_code,
        "latency_ms": duration_ms,
        "time": datetime.utcnow().isoformat() + "Z",
    }
    REQUEST_LOGS.appendleft(log_entry)
    return response

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Validate the API key when ADMIN_API_KEY is configured. No-op otherwise."""
    if ADMIN_API_KEY:
        if api_key is None:
            raise HTTPException(
                status_code=401,
                detail="X-API-Key header is required.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        if not secrets.compare_digest(api_key, ADMIN_API_KEY):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
    return api_key


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any], _api_key: Optional[str] = Depends(verify_api_key)):
    """Set memory configuration."""
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}


@app.post("/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate, _api_key: Optional[str] = Depends(verify_api_key)):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    try:
        response = MEMORY_INSTANCE.add(messages=[m.model_dump() for m in memory_create.messages], **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")  # This will log the full traceback
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", summary="Get memories")
def get_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Retrieve stored memories. 不传过滤参数时返回所有记忆。"""
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        return MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Retrieve a specific memory by ID."""
    try:
        return MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="Search memories")
def search_memories(search_req: SearchRequest, _api_key: Optional[str] = Depends(verify_api_key)):
    """Search for memories based on a query."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return MEMORY_INSTANCE.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(memory_id: str, updated_memory: Dict[str, Any], _api_key: Optional[str] = Depends(verify_api_key)):
    """Update an existing memory with new content.
    
    Args:
        memory_id (str): ID of the memory to update
        updated_memory (str): New content to update the memory with
        
    Returns:
        dict: Success message indicating the memory was updated
    """
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Retrieve memory history."""
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}", summary="Delete a memory")
def delete_memory(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Delete a specific memory by ID."""
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories", summary="Delete all memories")
def delete_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Delete all memories for a given identifier. 不传参数时删除所有记忆。"""
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", summary="Reset all memories")
def reset_memory(_api_key: Optional[str] = Depends(verify_api_key)):
    """Completely reset stored memories."""
    try:
        MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


# ===================== 实体 & 统计 & 请求日志 接口 =====================


@app.get("/entities", summary="List all entities")
def list_entities(
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """列出所有已知的实体（user_id / agent_id / run_id）及其记忆数量。"""
    try:
        all_memories = MEMORY_INSTANCE.get_all()
        entities_map: Dict[str, Dict[str, Any]] = {}

        memories_list = all_memories if isinstance(all_memories, list) else all_memories.get("results", all_memories.get("memories", []))

        for mem in memories_list:
            # 提取实体标识
            metadata = mem.get("metadata", {}) or {}
            user_id = mem.get("user_id") or metadata.get("user_id")
            agent_id = mem.get("agent_id") or metadata.get("agent_id")
            run_id = mem.get("run_id") or metadata.get("run_id")

            if user_id:
                key = f"user:{user_id}"
                if key not in entities_map:
                    entities_map[key] = {"id": user_id, "type": "user", "name": user_id, "memory_count": 0}
                entities_map[key]["memory_count"] += 1

            if agent_id:
                key = f"agent:{agent_id}"
                if key not in entities_map:
                    entities_map[key] = {"id": agent_id, "type": "agent", "name": agent_id, "memory_count": 0}
                entities_map[key]["memory_count"] += 1

            if run_id:
                key = f"run:{run_id}"
                if key not in entities_map:
                    entities_map[key] = {"id": run_id, "type": "run", "name": run_id, "memory_count": 0}
                entities_map[key]["memory_count"] += 1

        return {"entities": list(entities_map.values())}
    except Exception as e:
        logging.exception("Error in list_entities:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", summary="Get dashboard statistics")
def get_stats(
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """获取仪表盘统计数据：总记忆数、实体数、请求数等。"""
    try:
        all_memories = MEMORY_INSTANCE.get_all()
        memories_list = all_memories if isinstance(all_memories, list) else all_memories.get("results", all_memories.get("memories", []))
        total_memories = len(memories_list)

        # 统计实体数
        entity_ids = set()
        for mem in memories_list:
            metadata = mem.get("metadata", {}) or {}
            for field in ["user_id", "agent_id", "run_id"]:
                val = mem.get(field) or metadata.get(field)
                if val:
                    entity_ids.add(f"{field}:{val}")

        # 统计请求日志
        total_requests = len(REQUEST_LOGS)
        add_events = sum(1 for r in REQUEST_LOGS if r["method"] == "POST" and r["endpoint"] == "/memories")
        search_events = sum(1 for r in REQUEST_LOGS if r["endpoint"] == "/search")

        return {
            "total_memories": total_memories,
            "total_entities": len(entity_ids),
            "total_requests": total_requests,
            "add_events": add_events,
            "search_events": search_events,
        }
    except Exception as e:
        logging.exception("Error in get_stats:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/requests", summary="Get recent request logs")
def get_request_logs(
    limit: int = 50,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """获取最近的请求日志。"""
    return {"requests": list(REQUEST_LOGS)[:limit]}


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")

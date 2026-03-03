#!/usr/bin/env python3
"""RAG MCP Server for MockInterview.

通过 MCP stdio 协议提供向量检索能力：
  - ChromaDB  作为持久向量库
  - OpenAI Embedding API (text-embedding-3-small) 作为 Embedding 函数
  - FastMCP   作为 MCP 框架层

依赖环境变量：
  OPENAI_API_KEY      —— OpenAI API Key（必填）
  OPENAI_EMBED_MODEL  —— Embedding 模型名（默认 text-embedding-3-small）

对外暴露三个工具：
  query_knowledge_hub  —— 向量检索相关文档块
  list_collections     —— 列出所有集合及文档块数量
  add_document         —— 向指定集合添加文档

启动方式（由 start_rag_server.sh 调用，或手动测试）：
  python src/mcp_server/server.py
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from mcp.server.fastmcp import FastMCP

# ─── 路径配置 ──────────────────────────────────────────────────────────────────
# 文件位置：RAG-MCP-SERVER/src/mcp_server/server.py
#   parent       → mcp_server/
#   parent.parent → src/
#   parent.parent.parent → RAG-MCP-SERVER/
#   parent.parent.parent.parent → MockInterview/
_SERVER_ROOT = Path(__file__).resolve().parent.parent.parent   # RAG-MCP-SERVER/
_PROJECT_ROOT = _SERVER_ROOT.parent                            # MockInterview/
_DATA_DIR = _PROJECT_ROOT / "data" / "documents"
_CHROMA_DIR = _SERVER_ROOT / "chroma_data"

# ─── ChromaDB 初始化 ────────────────────────────────────────────────────────
_client = chromadb.PersistentClient(path=str(_CHROMA_DIR))

_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

if not _OPENAI_API_KEY:
    print("[RAG-MCP] 错误：未设置 OPENAI_API_KEY 环境变量", file=sys.stderr)
    sys.exit(1)

_ef = OpenAIEmbeddingFunction(
    api_key=_OPENAI_API_KEY,
    model_name=_EMBED_MODEL,
)

# 预定义集合名称及说明
_COLLECTIONS: dict[str, str] = {
    "resume": "候选人简历",
    "job_description": "职位描述",
    "default": "通用文档",
}


def _get_or_create(name: str):
    """获取或新建 ChromaDB 集合（cosine 距离）。"""
    return _client.get_or_create_collection(
        name=name,
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"},
    )


# ─── 文档处理 ──────────────────────────────────────────────────────────────────

def _chunk(text: str, size: int = 400, overlap: int = 60) -> list[str]:
    """将长文本切割为带重叠的块。"""
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def _ingest(file_path: Path, collection_name: str) -> int:
    """读取文件，分块后 upsert 到指定集合，返回块数。"""
    text = file_path.read_text(encoding="utf-8")
    chunks = _chunk(text)
    col = _get_or_create(collection_name)
    col.upsert(
        ids=[f"{file_path.stem}_chunk{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"source": file_path.name, "chunk_idx": i} for i in range(len(chunks))],
    )
    return len(chunks)


def _bootstrap() -> None:
    """服务器启动时预加载已知文档（使用 upsert，重复运行安全）。"""
    load_plan = [
        ("sample_resume_rag.txt", "resume"),
        ("sample_jd_rag.txt",     "job_description"),
        ("sample_resume.txt",     "resume"),
        ("sample_jd.txt",         "job_description"),
    ]
    for filename, col_name in load_plan:
        fp = _DATA_DIR / filename
        if fp.exists():
            n = _ingest(fp, col_name)
            _ingest(fp, "default")  # 同时写入 default 集合
            print(f"[RAG-MCP] 加载 {filename} → [{col_name}]（{n} 块）", file=sys.stderr)


try:
    _bootstrap()
except Exception as _exc:
    print(f"[RAG-MCP] 警告：预加载失败：{_exc}", file=sys.stderr)


# ─── MCP 服务器 ────────────────────────────────────────────────────────────────

mcp = FastMCP("rag-mcp-server")


@mcp.tool()
def query_knowledge_hub(
    query: str,
    collection: str = "default",
    top_k: int = 3,
) -> str:
    """从知识库检索与查询最相关的文档片段。

    Args:
        query:      搜索查询词或句子。
        collection: 集合名称，可选 resume / job_description / default。
        top_k:      返回的最大结果数（默认 3）。
    """
    try:
        col = _get_or_create(collection)
        count = col.count()
        if count == 0:
            return f"集合 [{collection}] 为空，请先通过 add_document 添加文档。"

        results = col.query(
            query_texts=[query],
            n_results=min(top_k, count),
        )
        docs: list[str] = results.get("documents", [[]])[0]
        distances: list[float] = results.get("distances", [[]])[0]

        if not docs:
            return "未找到相关文档。"

        parts = [f"[{collection}] 检索结果（{len(docs)} 条）：\n"]
        for i, (doc, dist) in enumerate(zip(docs, distances), 1):
            sim = max(0.0, 1.0 - dist)  # cosine distance → similarity
            parts.append(f"── 结果 {i}（相似度 {sim:.3f}）──\n{doc}")
        return "\n\n".join(parts)

    except Exception as exc:
        return f"查询失败：{exc}"


@mcp.tool()
def list_collections() -> str:
    """列出知识库中所有可用的集合及其文档块数量。"""
    rows: list[str] = []
    for name, desc in _COLLECTIONS.items():
        try:
            col = _client.get_collection(name, embedding_function=_ef)
            rows.append(f"  {name}（{desc}）：{col.count()} 块")
        except Exception:
            rows.append(f"  {name}（{desc}）：未初始化")
    return "可用集合：\n" + "\n".join(rows)


@mcp.tool()
def add_document(
    text: str,
    collection: str = "default",
    doc_id: Optional[str] = None,
) -> str:
    """向知识库的指定集合中添加文档。

    Args:
        text:       要添加的文档文本。
        collection: 目标集合名称（默认 default）。
        doc_id:     文档唯一标识符（可选，留空则自动生成）。
    """
    try:
        chunks = _chunk(text)
        col = _get_or_create(collection)
        base = doc_id or uuid.uuid4().hex[:8]
        col.upsert(
            ids=[f"{base}_chunk{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"chunk_idx": i, "doc_id": base} for i in range(len(chunks))],
        )
        return f"成功：将文档分为 {len(chunks)} 块并添加到集合 [{collection}]（doc_id={base}）。"
    except Exception as exc:
        return f"添加文档失败：{exc}"


# ─── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()  # 默认 stdio 传输

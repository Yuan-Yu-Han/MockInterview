"""
将文件（PDF、Markdown、TXT）通过 MCP ingest_documents 工具导入 RAG 知识库。

本脚本只做一件事：把文件路径发给 MODULAR-RAG-MCP-SERVER。
解析、切割、向量化、存储全部由 RAG 服务端完成。

用法：
    python scripts/ingest_questions.py
    python scripts/ingest_questions.py --file data/questions/ai_dev_interview_questions.md
    python scripts/ingest_questions.py --file /path/to/resume.pdf --collection resumes
    python scripts/ingest_questions.py --file ... --force
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAG_SERVER_CMD = [
    "/home/yuan0165/.conda/envs/vllm/bin/python",
    "/home/yuan0165/MODULAR-RAG-MCP-SERVER/src/mcp_server/server.py",
]


async def _call_mcp_ingest(file_path: str, collection: str, force: bool) -> str:
    """通过 MCP stdio JSON-RPC 调用 ingest_documents，返回结果文本。"""
    proc = await asyncio.create_subprocess_exec(
        *RAG_SERVER_CMD,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def send(msg: dict) -> None:
        proc.stdin.write((json.dumps(msg) + "\n").encode())
        await proc.stdin.drain()

    async def recv() -> dict:
        return json.loads(await proc.stdout.readline())

    # MCP handshake
    await send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "ingest_script", "version": "1.0"}}})
    await recv()
    await send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    # 只传文件路径，服务端负责读取和处理
    await send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "ingest_documents",
                           "arguments": {"file_path": file_path,
                                         "collection": collection,
                                         "force": force}}})
    result = await recv()

    proc.stdin.close()
    await proc.wait()

    if "error" in result:
        return f"MCP 错误: {result['error']}"
    blocks = result.get("result", {}).get("content", [])
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def ingest(file: Path, collection: str, force: bool) -> None:
    print(f"📄 文件: {file}")
    print(f"🔗 发送至 MODULAR-RAG-MCP-SERVER（集合: {collection}）…")
    result = asyncio.run(_call_mcp_ingest(str(file.resolve()), collection, force))
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 MCP 将文件导入 RAG 知识库")
    parser.add_argument(
        "--file",
        default=str(PROJECT_ROOT / "data" / "questions" / "ai_dev_interview_questions.md"),
        help="要导入的文件（.pdf / .md / .txt）",
    )
    parser.add_argument("--collection", default="knowledge_hub", help="目标集合名")
    parser.add_argument("--force", action="store_true", help="跳过去重，强制重新导入")
    args = parser.parse_args()

    f = Path(args.file)
    if not f.exists():
        print(f"❌ 文件不存在: {f}")
        sys.exit(1)

    ingest(f, collection=args.collection, force=args.force)


if __name__ == "__main__":
    main()

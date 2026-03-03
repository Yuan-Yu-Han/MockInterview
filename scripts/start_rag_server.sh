#!/bin/bash
# Start the MODULAR-RAG-MCP-SERVER as a stdio MCP server.
#
# This script is called automatically by agno's MCPTools when use_rag=True.
# It can also be called manually for testing:
#   echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | ./start_rag_server.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_DIR="$SCRIPT_DIR/RAG-MCP-SERVER"

export PYTHONPATH="$RAG_DIR:${PYTHONPATH:-}"

exec python "$RAG_DIR/src/mcp_server/server.py"

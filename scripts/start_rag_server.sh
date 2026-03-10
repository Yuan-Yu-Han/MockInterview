#!/bin/bash
# Start the MODULAR-RAG-MCP-SERVER as a stdio MCP server.
#
# This script is called automatically by agno's MCPTools when use_rag=True.
# It can also be called manually for testing:
#   echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | ./start_rag_server.sh

set -e

MODULAR_RAG="/home/yuan0165/MODULAR-RAG-MCP-SERVER"

exec /home/yuan0165/.conda/envs/vllm/bin/python "$MODULAR_RAG/src/mcp_server/server.py"

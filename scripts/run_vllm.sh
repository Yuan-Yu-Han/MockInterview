#!/bin/bash

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


MODEL="/projects/yuan0165/Qwen3-8B"
PORT=8000

# 创建日志目录
LOG_DIR="/home/yuan0165/MockInterview/logs/vllm"
mkdir -p $LOG_DIR

# 创建 outputs 目录（用于 PID 文件）
OUTPUT_DIR="/home/yuan0165/MockInterview/logs/outputs"
mkdir -p $OUTPUT_DIR

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/vllm_server_$TIMESTAMP.log"

echo "启动 vLLM API 服务，日志保存到: $LOG_FILE"

# 启动 vLLM API 服务并重定向日志
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name Qwen3-8B \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --dtype=float16 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    > $LOG_FILE 2>&1
    


# 保存进程ID到文件，方便后续管理
echo $! > $OUTPUT_DIR/vllm_server.pid
echo "vLLM服务已启动，PID: $!"
echo "日志文件: $LOG_FILE"
echo "PID文件: $OUTPUT_DIR/vllm_server.pid"


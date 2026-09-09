#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PREFIX="$SCRIPT_DIR/train_base"

# Pick a free port automatically
MASTER_PORT=$(python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
)

echo "Using master port: $MASTER_PORT"

# Auto-detect next available numbered log file
LOG_NUM=1
while [ -f "${LOG_PREFIX}_${LOG_NUM}.log" ]; do
    LOG_NUM=$((LOG_NUM + 1))
done
LOG_FILE="${LOG_PREFIX}_${LOG_NUM}.log"
echo "Logging to: $LOG_FILE"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
torchrun \
    --nproc_per_node=${NPROC_PER_NODE:-1} \
    --master_port=$MASTER_PORT \
    sft.py "$@" 2>&1 | tee "$LOG_FILE"
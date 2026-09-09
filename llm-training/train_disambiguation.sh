#!/bin/bash

# Find Conda python path
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    CONDA_PYTHON="$CONDA_PREFIX/bin/python"
elif [ -f "/home/hungpv/miniconda3/envs/qwen3-sft/bin/python" ]; then
    CONDA_PYTHON="/home/hungpv/miniconda3/envs/qwen3-sft/bin/python"
elif [ -f "/mnt/hungpv/miniconda3/envs/carbench_env/bin/python" ]; then
    CONDA_PYTHON="/mnt/hungpv/miniconda3/envs/carbench_env/bin/python"
else
    CONDA_PYTHON="python"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PREFIX="$SCRIPT_DIR/train_disambiguation"

# Make sure CUDA_VISIBLE_DEVICES is set to 0 if not provided
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# Auto-detect next available numbered log file
LOG_NUM=1
while [ -f "${LOG_PREFIX}_${LOG_NUM}.log" ]; do
    LOG_NUM=$((LOG_NUM + 1))
done
LOG_FILE="${LOG_PREFIX}_${LOG_NUM}.log"
echo "Logging to: $LOG_FILE"

# Run the training script, tee output to log file and stdout
$CONDA_PYTHON "$SCRIPT_DIR/train_disambiguation.py" "$@" 2>&1 | tee "$LOG_FILE"

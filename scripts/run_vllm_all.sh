#!/bin/bash
# =====================================================================
# Run vLLM Server for Unified SFT Merged Model (All Tasks)
# Uses modern vllm conda environment (vLLM 0.19.0 + Transformers 4.57.6)
# =====================================================================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONNOUSERSITE=1
export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON_BIN="/home/hungpv/miniconda3/envs/vllm/bin/python"
MODEL_PATH="/mnt/hungpv/outputs_all/sft_merged_model"

if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "? ERROR: Model not found at $MODEL_PATH!"
    exit 1
fi

echo "========================================================="
echo "Starting vLLM Server for CAR-Bench Agent Evaluation"
echo "  - Python Bin : $PYTHON_BIN"
echo "  - Model Path : $MODEL_PATH"
echo "  - Port       : 8000"
echo "  - GPU Device : $CUDA_VISIBLE_DEVICES"
echo "========================================================="

$PYTHON_BIN -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name sft_merged_model base disambiguation hallucination default dragonstorm123/qwen3.5-4b-sft-all \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.45 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code
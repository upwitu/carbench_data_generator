#!/bin/bash
# =====================================================================
# Run vLLM for Disambiguation SFT Model
# Automatically prioritizes local freshly-trained merged model over cached HF snapshots
# Multi-alias support for --served-model-name to prevent 404 Model Not Found errors
# =====================================================================

source /mnt/hungpv/miniconda3/etc/profile.d/conda.sh
conda activate carbench_env
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER_SAMPLER=0

# Default HuggingFace repo ID
MODEL_NAME="dragonstorm123/qwen3.5-4b-sft-disambiguation"

# Local freshly-trained model candidate paths
LOCAL_MODELS=(
    "/mnt/hungpv/outputs_disambiguation/sft_merged_model"
    "/mnt/hungpv/car_bench_notebook/outputs_disambiguation/sft_merged_model"
    "./outputs_disambiguation/sft_merged_model"
)

# Priority 1: Check if local trained merged model exists
FOUND_LOCAL=0
for path in "${LOCAL_MODELS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/config.json" ]; then
        echo "[MODEL LOAD] Using local freshly-trained merged model: $path"
        MODEL_NAME="$path"
        FOUND_LOCAL=1
        break
    fi
done

# Priority 2: If local model absent, serve from Hugging Face Hub (clear old outdated snapshot if present)
if [ $FOUND_LOCAL -eq 0 ]; then
    echo "[MODEL LOAD] Local merged model not found. Fetching latest version from Hugging Face: $MODEL_NAME"
    rm -rf /mnt/hungpv/.cache/huggingface/hub/models--dragonstorm123--qwen3.5-4b-sft-disambiguation/snapshots/* 2>/dev/null || true
fi

echo "[VLLM SERVE] Starting vLLM server with model: $MODEL_NAME"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --served-model-name disambiguation dragonstorm123/qwen3.5-4b-sft-disambiguation /mnt/hungpv/outputs_disambiguation/sft_merged_model default hallucination base \
  --port 8300 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.45 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --attention-backend FLASH_ATTN \
  --max-num-seqs 128

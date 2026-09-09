#!/usr/bin/env bash
# =====================================================================
# CAR-Bench Winner-Inspired SFT Data Pipeline (1-Command Runner)
# Automated environment setup, dataset sanitization, and execution
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-sanitize}"
CONCURRENCY="${CONCURRENCY:-30}"
VARIATIONS="${VARIATIONS:-10}"

echo "=========================================================="
echo " Starting CAR-Bench Data Generation & Verification Pipeline"
echo " Target Mode: $MODE | Concurrency: $CONCURRENCY | Variations: $VARIATIONS"
echo "=========================================================="

# 1. Ensure uv package manager exists
if ! command -v uv &> /dev/null; then
    echo "[1/4] Installing 'uv' package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# 2. Synchronize virtual environment
echo "[2/4] Synchronizing virtual environment dependencies..."
uv sync

# 3. Environment variables configuration
if [ ! -f ".env" ]; then
    echo "[3/4] Initializing .env configuration from example..."
    cat << 'EOF' > .env
# Hugging Face Configuration
HF_TOKEN=your_hf_token_here
HF_DATASET_REPO=upwitu/carbench_sft_winner_dataset

# LLM Backend Configuration
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_API_KEY=your_openai_api_key_here
CAR_BENCH_MODEL=deepseek-v4-flash

# Generation Limits
CONCURRENCY_LIMIT=30
VARIATIONS_PER_TASK=10
EOF
fi

# 4. Execute dataset sanitization
echo "[4/4] Sanitizing dataset files in data/new_data/..."
uv run python scripts/sanitize_dataset.py

# 5. Optional Mode Dispatch
case "$MODE" in
    all)
        echo "Executing full dataset synthesis (Multi-role JSON + CodeAct Python)..."
        uv run python -m sft_generator.main --dataset-type all --concurrency "$CONCURRENCY" --variations "$VARIATIONS"
        ;;
    multirole)
        echo "Executing Multi-role JSON synthesis..."
        uv run python -m sft_generator.main --dataset-type multi_role_json --concurrency "$CONCURRENCY" --variations "$VARIATIONS"
        ;;
    codeact)
        echo "Executing CodeAct Python synthesis..."
        uv run python -m sft_generator.main --dataset-type codeact_python --concurrency "$CONCURRENCY" --variations "$VARIATIONS"
        ;;
    upload-hf)
        echo "Publishing dataset to Hugging Face..."
        uv run python sft_generator/upload_to_hf.py
        ;;
    sanitize|*)
        echo "Pipeline initialized and datasets sanitized successfully."
        ;;
esac

echo "Pipeline execution finished."

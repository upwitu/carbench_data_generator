#!/usr/bin/env python3
"""Upload Qwen3-4B SFT LoRA Adapter and Merged Model to Hugging Face under upwitu organization."""

import os
import sys
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable not found. Please export HF_TOKEN before running.")
    sys.exit(1)

LORA_LOCAL_PATH = "/mnt/hungpv/outputs_all/sft_lora_adapter"
MERGED_LOCAL_PATH = "/mnt/hungpv/outputs_all/sft_merged_model"

LORA_REPO_ID = "upwitu/qwen3-4b-sft-all-lora"
MERGED_REPO_ID = "upwitu/qwen3-4b-sft-all"

api = HfApi(token=HF_TOKEN)
try:
    user = api.whoami()
    print(f"Authenticated as: {user['name']} with access to orgs: {[o['name'] for o in user.get('orgs', [])]}")
except Exception as e:
    print(f"Authentication error: {e}")
    sys.exit(1)

# 1. Create and Upload LoRA Adapter
if os.path.exists(LORA_LOCAL_PATH):
    print(f"Creating repo {LORA_REPO_ID}...")
    create_repo(repo_id=LORA_REPO_ID, repo_type="model", token=HF_TOKEN, exist_ok=True)
    print(f"Uploading LoRA adapter from {LORA_LOCAL_PATH}...")
    api.upload_folder(
        folder_path=LORA_LOCAL_PATH,
        repo_id=LORA_REPO_ID,
        repo_type="model",
        commit_message="Release Qwen3-4B SFT LoRA adapter for CAR-Bench all tasks"
    )
    print("LoRA adapter upload complete.")
else:
    print(f"LoRA path not found: {LORA_LOCAL_PATH} (if running locally, execute on remote server)")

# 2. Create and Upload Merged Model
if os.path.exists(MERGED_LOCAL_PATH):
    print(f"Creating repo {MERGED_REPO_ID}...")
    create_repo(repo_id=MERGED_REPO_ID, repo_type="model", token=HF_TOKEN, exist_ok=True)
    print(f"Uploading Merged model from {MERGED_LOCAL_PATH}...")
    api.upload_folder(
        folder_path=MERGED_LOCAL_PATH,
        repo_id=MERGED_REPO_ID,
        repo_type="model",
        commit_message="Release Qwen3-4B SFT merged BF16 weights for CAR-Bench all tasks"
    )
    print("Merged model upload complete.")
else:
    print(f"Merged model path not found: {MERGED_LOCAL_PATH} (if running locally, execute on remote server)")

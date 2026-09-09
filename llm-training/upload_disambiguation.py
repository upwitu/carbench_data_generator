import os
import torch
import ctypes
import glob

# Pre-load Nvidia CUDA libraries
nvidia_paths = [
    "/mnt/hungpv/miniconda3/envs/carbench_env/lib/python3.10/site-packages/nvidia/*/lib/*.so*",
    "/mnt/hungpv/miniconda3/envs/carbench_env/lib/python3.10/site-packages/nvidia/cu13/lib/*.so*"
]
for pattern in nvidia_paths:
    for so_file in glob.glob(pattern):
        try:
            ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

import builtins
builtins.input = lambda *args, **kwargs: "y"
os.environ["HF_TRUST_REMOTE_CODE"] = "1"

from unsloth import FastLanguageModel

ADAPTER_PATH = "/mnt/hungpv/outputs_disambiguation/sft_lora_adapter"
MERGED_REPO_ID = "dragonstorm123/qwen3.5-4b-sft-disambiguation"
LORA_REPO_ID = "dragonstorm123/qwen3.5-4b-sft-disambiguation-lora"

print(f"Loading trained LoRA adapter from {ADAPTER_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=16384,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    load_in_4bit=True,
    trust_remote_code=True
)

print(f"Uploading LoRA adapter to Hugging Face Hub ({LORA_REPO_ID})...")
try:
    model.push_to_hub(LORA_REPO_ID, tokenizer=tokenizer)
    print(f"Successfully uploaded LoRA adapter to {LORA_REPO_ID}!")
except Exception as e:
    print(f"Error uploading LoRA adapter: {e}")

print(f"Uploading 16-bit merged model to Hugging Face Hub ({MERGED_REPO_ID})...")
try:
    model.push_to_hub_merged(
        MERGED_REPO_ID,
        tokenizer,
        save_method="merged_16bit"
    )
    print(f"Successfully uploaded 16-bit merged model to {MERGED_REPO_ID}!")
except Exception as e:
    print(f"Error uploading merged model: {e}")

print("Upload script completed!")

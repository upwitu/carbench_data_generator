import os
import glob
import ctypes
import logging
import builtins

# Auto-confirm any interactive prompts (e.g. Unsloth trust_remote_code prompt)
builtins.input = lambda *args, **kwargs: "y"

# Pre-load Nvidia CUDA libraries
print("Pre-loading Nvidia CUDA libraries...")
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

import torch
os.environ["HF_TRUST_REMOTE_CODE"] = "1"
logging.getLogger("fla").setLevel(logging.ERROR)

from unsloth import FastLanguageModel

ADAPTER_PATH = "/mnt/hungpv/car_bench_notebook/llm-training/outputs_hallucination/sft_lora_adapter"
MERGED_REPO_ID = "dragonstorm123/qwen3.5-4b-sft-hallucination"
LORA_REPO_ID = "dragonstorm123/qwen3.5-4b-sft-hallucination-lora"
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Loading trained LoRA adapter from: {ADAPTER_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=16384,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    load_in_4bit=True,
    trust_remote_code=True
)

print(f"Pushing LoRA adapter to Hugging Face Hub ({LORA_REPO_ID})...")
try:
    model.push_to_hub(LORA_REPO_ID, tokenizer=tokenizer, token=HF_TOKEN)
    print(f"Successfully uploaded LoRA adapter to {LORA_REPO_ID}!")
except Exception as e:
    print(f"Warning pushing LoRA adapter: {e}")

print(f"Pushing merged 16-bit model directly to Hugging Face Hub ({MERGED_REPO_ID})...")
try:
    model.push_to_hub_merged(
        MERGED_REPO_ID,
        tokenizer,
        save_method="merged_16bit",
        token=HF_TOKEN
    )
    print(f"🎉 SUCCESS! Model successfully uploaded to https://huggingface.co/{MERGED_REPO_ID}")
except Exception as e:
    print(f"Error uploading merged model: {e}")

print("Upload script completed!")

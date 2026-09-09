import os
os.environ["HF_TRUST_REMOTE_CODE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import builtins
builtins.input = lambda *args, **kwargs: "y"

_orig_print = builtins.print
def _quiet_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if "Are you certain you want to do remote code execution" in msg:
        return
    _orig_print(*args, **kwargs)
builtins.print = _quiet_print

import sys
import json
import argparse
import logging
import ctypes
import glob
import re

logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("datasets.fingerprint").setLevel(logging.ERROR)

# Pre-load Nvidia CUDA libraries to prevent bitsandbytes loading errors
print("Pre-loading Nvidia CUDA libraries to prevent bitsandbytes loading errors...")
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
print("Nvidia CUDA libraries pre-loading completed.")

import torch
import torch.utils

# Suppress python version warning from flash-linear-attention (fla)
logging.getLogger("fla").setLevel(logging.ERROR)

# =====================================================================
# 1. MONKEYPATCH TORCHAO & PYTORCH COMPATIBILITY FOR OLDER SERVERS
# =====================================================================
print("Applying system compatibility patches...")
for i in range(1, 8):
    int_attr = f"int{i}"
    uint_attr = f"uint{i}"
    if not hasattr(torch, int_attr):
        setattr(torch, int_attr, torch.int8)
    if not hasattr(torch, uint_attr):
        setattr(torch, uint_attr, torch.uint8)

if not hasattr(torch.utils, "_pytree"):
    import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda cls: cls

# Prevent incompatible torchao quantizers from breaking transformers/unsloth
sys.modules['torchao'] = None
sys.modules['torchao.quantization'] = None
sys.modules['torchao.prototype'] = None
print("Compatibility patches successfully applied!")

# Unsloth must be imported before any other HuggingFace/transformers libraries
from unsloth import FastLanguageModel

# Bypass Unsloth telemetry check to prevent 120s HuggingFace timeout
import unsloth.models._utils
unsloth.models._utils.get_statistics = lambda *args, **kwargs: None
unsloth.models._utils._get_statistics = lambda *args, **kwargs: None

# Prevent transformers from loading native torchao internally and raising AttributeErrors
import transformers
transformers.utils.import_utils.is_torchao_available = lambda *args, **kwargs: False
transformers.utils.is_torchao_available = lambda *args, **kwargs: False
if hasattr(transformers.utils.import_utils, "_torchao_available"):
    transformers.utils.import_utils._torchao_available = False

from huggingface_hub import HfApi, hf_hub_download
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# Parse arguments
parser = argparse.ArgumentParser(description="Train Qwen 3.5 4B on CAR-bench Base tasks.")
parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test with 3 steps.")
args_cli = parser.parse_args()

# Unset DDP environment variables for single GPU training
for key in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
    if key in os.environ:
        del os.environ[key]

# Optimize PyTorch memory allocator to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = "cuda:0"
else:
    device = "cpu"
print(f"Using device: {device}")

# =====================================================================
# 2. LOAD LIBRARIES & MODEL
# =====================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
PERSISTENT_DIR = "/mnt/hungpv/outputs_all"
ADAPTER_PATH = os.path.join(PERSISTENT_DIR, "sft_lora_adapter")
MERGED_PATH = os.path.join(PERSISTENT_DIR, "sft_merged_model")

os.makedirs(PERSISTENT_DIR, exist_ok=True)

print("Loading Qwen 3.5 4B Model via Unsloth in BF16...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    load_in_4bit=True,
    trust_remote_code=True
)

text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
tokenizer.eos_token = "<|im_end|>"
text_tokenizer.eos_token = "<|im_end|>"

# =====================================================================
# 3. LOAD & FORMAT ALL IN-DOMAIN DATASETS (BASE + DISAMBIGUATION + HALLUCINATION)
# =====================================================================
candidate_paths = [
    "/mnt/hungpv/car_bench_notebook/sft_generator/outputs/carbench_sft_multirole_json.jsonl",
    "/mnt/hungpv/car_bench_notebook/data/car_base_sft.jsonl",
    "/mnt/hungpv/car_bench_notebook/data/car_disambiguation_sft.jsonl",
    "/mnt/hungpv/car_bench_notebook/data/car_hallucination_sft.jsonl",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sft_generator", "outputs", "carbench_sft_multirole_json.jsonl"),
]

downloaded_file_paths = []
for p in candidate_paths:
    if os.path.exists(p) and p not in downloaded_file_paths:
        downloaded_file_paths.append(p)
        print(f"Found local dataset file: '{p}'")

if not downloaded_file_paths:
    raise FileNotFoundError("No dataset files found on server!")

def _parse_messages(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return None
    elif isinstance(raw, list):
        return raw
    return None

def sanitize_tools_for_template(tools):
    """Validate and clean tool schemas to ensure apply_chat_template won't throw TypeError."""
    if not tools:
        return None
    result = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        t_clean = dict(t)
        fn = t_clean.get("function")
        if not isinstance(fn, dict):
            continue
        fn_clean = dict(fn)
        params = fn_clean.get("parameters", {})
        if isinstance(params, dict):
            params_clean = dict(params)
            props = params_clean.get("properties", {})
            if not isinstance(props, dict):
                params_clean["properties"] = {}
            else:
                for pk, pv in list(props.items()):
                    if not isinstance(pv, dict):
                        params_clean["properties"][pk] = {"type": "string"}
            fn_clean["parameters"] = params_clean
        else:
            fn_clean["parameters"] = {"type": "object", "properties": {}, "required": []}
        t_clean["function"] = fn_clean
        result.append(t_clean)
    return result if result else None

def sanitize_messages_for_template(messages):
    """Ensure all tool_calls in assistant messages have dict arguments for Jinja2 template."""
    if not messages:
        return []
    clean_msgs = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        m_clean = dict(msg)
        tcs = m_clean.get("tool_calls")
        if tcs is not None:
            if isinstance(tcs, list):
                clean_tcs = []
                for tc in tcs:
                    if isinstance(tc, dict):
                        tc_clean = dict(tc)
                        fn = tc_clean.get("function")
                        if isinstance(fn, dict):
                            fn_clean = dict(fn)
                            args = fn_clean.get("arguments")
                            if isinstance(args, str):
                                try:
                                    fn_clean["arguments"] = json.loads(args)
                                except Exception:
                                    fn_clean["arguments"] = {}
                            elif not isinstance(args, dict):
                                fn_clean["arguments"] = {}
                            tc_clean["function"] = fn_clean
                        clean_tcs.append(tc_clean)
                m_clean["tool_calls"] = clean_tcs if clean_tcs else None
                if m_clean["tool_calls"] is None:
                    m_clean.pop("tool_calls", None)
            else:
                m_clean.pop("tool_calls", None)
        clean_msgs.append(m_clean)
    return clean_msgs

print("Formatting Unified Multi-Task dataset (Base + Disambiguation + Hallucination) with chat template...")
formatted_samples = []
skipped_too_long = 0
_err_count = [0]
seen_conversations = set()

for file_path in downloaded_file_paths:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                raw_msgs = sample.get("conversations") or sample.get("messages") or sample.get("augmented_messages") or sample.get("raw_messages")
                messages = _parse_messages(raw_msgs)
                if not messages:
                    continue

                # Clean think tags
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                        msg["content"] = re.sub(r"<think>.*?</think>", "", msg["content"], flags=re.DOTALL).strip()

                # Deduplication check
                conv_hash = hash(json.dumps(messages, sort_keys=True) + json.dumps(sample.get("tools", []), sort_keys=True))
                if conv_hash in seen_conversations:
                    continue
                seen_conversations.add(conv_hash)

                tools_clean = sanitize_tools_for_template(sample.get("tools"))
                messages_clean = sanitize_messages_for_template(messages)

                # Format chat template with tools schema and disable thinking mode
                text = tokenizer.apply_chat_template(
                    messages_clean,
                    tools=tools_clean,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False
                )

                input_ids = text_tokenizer.encode(text, add_special_tokens=False)
                if len(input_ids) > 4096:
                    skipped_too_long += 1
                    continue

                formatted_samples.append({"text": text})
            except Exception as e:
                _err_count[0] += 1
                if _err_count[0] <= 3:
                    print(f"[SAMPLE ERROR #{_err_count[0]}] {type(e).__name__}: {str(e)[:300]}")

print(f"Total formatted Base task samples: {len(formatted_samples)}")
if skipped_too_long > 0:
    print(f"Skipped {skipped_too_long} samples exceeding max token length (16384).")

dataset_filtered = Dataset.from_list(formatted_samples)
dataset_split = dataset_filtered.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]
print(f"Split completed: {len(train_dataset)} train samples, {len(eval_dataset)} evaluation samples")

print("First sample formatted text preview:")
print(train_dataset[0]["text"][:600] + "\n...\n")

# Apply LoRA model wrappers
print("Configuring PEFT LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("LoRA configuration completed.")
model.print_trainable_parameters()

# =====================================================================
# 4. CONFIGURE SFT TRAINER & APPLY MASKING
# =====================================================================
sft_config = SFTConfig(
    dataset_text_field="text",
    output_dir=PERSISTENT_DIR,
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    num_train_epochs=2.0 if not args_cli.smoke_test else 1.0,
    max_steps=3 if args_cli.smoke_test else -1,
    weight_decay=0.02,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    fp16=False,
    bf16=True,
    optim="adamw_8bit",
    eval_strategy="epoch" if not args_cli.smoke_test else "no",
    save_strategy="epoch" if not args_cli.smoke_test else "no",
    save_total_limit=1,
    load_best_model_at_end=True if not args_cli.smoke_test else False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=1 if args_cli.smoke_test else 5,
    report_to="none",
    packing=False,
    max_seq_length=4096
)

sft_config.eos_token = None
tokenizer.eos_token = "<|im_end|>"
text_tokenizer.eos_token = "<|im_end|>"

print("Applying MultiTurnDataCollator for multi-turn assistant response masking...")
from transformers import DataCollatorForLanguageModeling

class MultiTurnDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, response_template, instruction_template, tokenizer, *args, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
        self.response_token_ids = response_template
        self.end_token_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        self.end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        
        for i in range(labels.shape[0]):
            seq = batch["input_ids"][i].tolist()
            labels[i, :] = -100
            
            start_search = 0
            while start_search < len(seq):
                resp_idx = -1
                for idx in range(start_search, len(seq) - len(self.response_token_ids) + 1):
                    if seq[idx:idx+len(self.response_token_ids)] == self.response_token_ids:
                        resp_idx = idx + len(self.response_token_ids)
                        break
                
                if resp_idx == -1:
                    break
                
                end_idx = len(seq)
                for idx in range(resp_idx, len(seq) - len(self.end_token_ids) + 1):
                    if seq[idx:idx+len(self.end_token_ids)] == self.end_token_ids:
                        end_idx = idx + len(self.end_token_ids)
                        break
                
                actual_start = resp_idx
                for idx in range(resp_idx, min(end_idx, resp_idx + 300)):
                    if idx + len(self.end_think_ids) <= len(seq) and seq[idx:idx+len(self.end_think_ids)] == self.end_think_ids:
                        actual_start = idx + len(self.end_think_ids)
                        if actual_start < len(seq) and seq[actual_start] == 198:
                            actual_start += 1
                        break

                labels[i, actual_start:end_idx] = batch["input_ids"][i][actual_start:end_idx]
                start_search = end_idx
        
        batch["labels"] = labels
        return batch

response_template = text_tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
instruction_template = text_tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)

collator = MultiTurnDataCollator(
    response_template=response_template,
    instruction_template=instruction_template,
    tokenizer=text_tokenizer
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=text_tokenizer,
    data_collator=collator
)

# =====================================================================
# 5. EXECUTE TRAINING
# =====================================================================
print("Starting Base Task SFT Fine-Tuning...")
trainer.train()

# =====================================================================
# 6. VISUALIZE AND SAVE PLOT
# =====================================================================
if not args_cli.smoke_test:
    print("Generating loss curve plot...")
    try:
        if plt is not None and hasattr(trainer, "state") and trainer.state.log_history:
            history = trainer.state.log_history
            train_steps, train_losses = [], []
            eval_steps, eval_losses = [], []
            
            for log in history:
                step = log.get("step")
                if "loss" in log:
                    train_steps.append(step)
                    train_losses.append(log["loss"])
                if "eval_loss" in log:
                    eval_steps.append(step)
                    eval_losses.append(log["eval_loss"])
                    
            plt.figure(figsize=(10, 5))
            if train_losses:
                plt.plot(train_steps, train_losses, label="Train Loss", color="red", marker="o")
            if eval_losses:
                plt.plot(eval_steps, eval_losses, label="Validation Loss", color="blue", marker="s")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Unified Multi-Task SFT Loss Curve: Train vs Validation")
            plt.legend()
            plt.grid(True)
            
            plot_path = os.path.join(PERSISTENT_DIR, "loss_plot.png")
            plt.savefig(plot_path)
            print(f"Loss plot saved to: {plot_path}")
    except Exception as e:
        print(f"Note: Could not generate loss plot ({e}), proceeding to model saving.")

# =====================================================================
# 7. SAVE ADAPTERS, MERGED MODEL AND PUSH TO HUGGING FACE HUB
# =====================================================================
MERGED_REPO_ID = "dragonstorm123/qwen3.5-4b-sft-all"
LORA_REPO_ID = "dragonstorm123/qwen3.5-4b-sft-all-lora"

print(f"Saving PEFT LoRA adapter checkpoint to {ADAPTER_PATH}...")
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)

try:
    print(f"Uploading LoRA adapter to Hugging Face Hub ({LORA_REPO_ID})...")
    model.push_to_hub(LORA_REPO_ID, tokenizer=tokenizer, token=os.environ.get("HF_TOKEN"))
    print(f"Successfully uploaded LoRA adapter to {LORA_REPO_ID}!")
except Exception as e:
    print(f"Warning: Failed pushing LoRA adapter to HF Hub: {e}")

try:
    print(f"Uploading merged 16-bit model to Hugging Face Hub ({MERGED_REPO_ID})...")
    model.push_to_hub_merged(
        MERGED_REPO_ID,
        tokenizer,
        save_method="merged_16bit",
        token=os.environ.get("HF_TOKEN")
    )
    print(f"Successfully uploaded merged 16-bit model to {MERGED_REPO_ID}!")
except Exception as e:
    print(f"Warning: Failed to push merged model to HF Hub: {e}")

try:
    print("Merging and saving model weights to 16-bit precision locally...")
    model.save_pretrained_merged(MERGED_PATH, tokenizer, save_method="merged_16bit")
    print(f"Merged model successfully saved to: {MERGED_PATH}")
except Exception as e:
    print(f"Warning: Failed saving merged model locally: {e}")

print("Unified Multi-Task SFT training pipeline successfully finished!")

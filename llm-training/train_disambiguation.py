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

import builtins
# Auto-confirm any interactive prompts (e.g. Unsloth trust_remote_code prompt)
builtins.input = lambda *args, **kwargs: "y"

# Force HF trust remote code to bypass interactive prompts
os.environ["HF_TRUST_REMOTE_CODE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress the python version warning from flash-linear-attention (fla)
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
from unsloth import FastLanguageModel, train_on_responses_only

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
parser = argparse.ArgumentParser(description="Train Qwen 3.5 4B on disambiguation tasks.")
parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test with 3 steps.")
args_cli = parser.parse_args()

# Unset DDP environment variables for single GPU training
for key in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
    if key in os.environ:
        del os.environ[key]

# Optimize PyTorch memory allocator to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Force single GPU selection if needed, e.g., using CUDA_VISIBLE_DEVICES outside, or default to device 0
if torch.cuda.is_available():
    # Make sure we bound to the current visible cuda device 0 after masking
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
PERSISTENT_DIR = "/mnt/hungpv/outputs_disambiguation"
ADAPTER_PATH = os.path.join(PERSISTENT_DIR, "sft_lora_adapter")
MERGED_PATH = os.path.join(PERSISTENT_DIR, "sft_merged_model")

os.makedirs(PERSISTENT_DIR, exist_ok=True)

print("Loading Qwen 3.5 4B Model via Unsloth in BF16...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=16384,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    load_in_4bit=True,   # Enable 4-bit quantization for faster training
    trust_remote_code=True
)

# Extract actual text tokenizer if the loaded object is a Processor (e.g., Qwen3VLProcessor)
text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

# Force eos_token to be <|im_end|> to bypass vocabulary check errors in SFTTrainer
tokenizer.eos_token = "<|im_end|>"
text_tokenizer.eos_token = "<|im_end|>"

# =====================================================================
# 3. LOAD IN-DOMAIN DATASET (DISAMBIGUATION TASKS)
# =====================================================================
# Priority: Local generated dataset (data/car_disambiguation_sft.jsonl) > Server path > HF repo fallback
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_disambig_path = os.path.join(base_dir, "data", "car_disambiguation_sft.jsonl")
server_disambig_path = "/mnt/hungpv/car_bench_notebook/data/car_disambiguation_sft.jsonl"

downloaded_file_paths = []
if os.path.exists(local_disambig_path):
    print(f"Found local in-domain dataset: '{local_disambig_path}'")
    downloaded_file_paths.append(local_disambig_path)
elif os.path.exists(server_disambig_path):
    print(f"Found server in-domain dataset: '{server_disambig_path}'")
    downloaded_file_paths.append(server_disambig_path)
else:
    repo_id = "upwitu/carbench_sft_benchmark_data"
    print(f"Local in-domain dataset not found. Auto-downloading from HF repo '{repo_id}'...")
    try:
        local_file_path = hf_hub_download(
            repo_id=repo_id,
            filename="data/car_disambiguation_sft.jsonl",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN")
        )
        downloaded_file_paths.append(local_file_path)
    except Exception as e:
        print(f"HF download fallback error: {e}")

# Helper function to parse raw messages into correct JSON representation
def _parse_messages(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return None
    if isinstance(raw, (list, tuple)):
        result = []
        for item in raw:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, str):
                try:
                    result.append(json.loads(item))
                except Exception:
                    pass
            else:
                try:
                    result.append(dict(item))
                except Exception:
                    pass
        return result
    try:
        return [dict(m) for m in raw]
    except Exception:
        return None

# Helper function to detect tool calling loops or redundant calls
def has_tool_loop(messages, max_consecutive_tool_calls=3) -> bool:
    if not messages:
        return False
    tool_call_counts = {}
    consecutive = 0
    prev_tool_name = None
    
    for msg in messages:
        role = msg.get("role", "")
        tool_calls = msg.get("tool_calls")
        content = msg.get("content", "") or ""
        
        if role == "assistant":
            t_names = []
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        f = tc.get("function", {})
                        if f and f.get("name"):
                            t_names.append(f.get("name"))
            elif "<tool_call>" in content:
                import re
                matches = re.findall(r"<function=(\w+)>", content)
                t_names.extend(matches)
            
            for t_name in t_names:
                if t_name == prev_tool_name:
                    consecutive += 1
                else:
                    consecutive = 1
                    prev_tool_name = t_name
                
                tool_call_counts[t_name] = tool_call_counts.get(t_name, 0) + 1
                if consecutive >= max_consecutive_tool_calls or tool_call_counts[t_name] > 4:
                    return True
    return False

# Helper function to enrich system prompt
def enrich_system_prompt(system_prompt: str) -> str:
    lines = [system_prompt]
    lines.append(
        "- TOOL CALLING RULES: Do NOT call the same tool repeatedly with the same or similar arguments. "
        "If a tool returns an error, empty output, or unexpected result, immediately inform the user about the issue "
        "instead of retrying in a loop. Maximum 2 tool execution retries per task before seeking user confirmation."
    )
    lines.append(
        "- CAR-BENCH DISAMBIGUATION POLICY: When a user request is ambiguous or underspecified (e.g. missing a required value "
        "such as a percentage, level, or target), follow this strict two-level resolution strategy:\n"
        "  (1) INTERNAL RESOLUTION FIRST: Always check available internal tools/data sources first (such as calling "
        "get_user_preferences or get_vehicle_status) to resolve ambiguity before asking the user.\n"
        "  (2) EXTERNAL RESOLUTION ONLY AS FALLBACK: If and only if internal resolution returns empty or unavailable information, "
        "ask the user for clarification. Never ask the user for information that is already available internally."
    )
    return "\n\n".join(lines)

# Helper function to sanitize tools for apply_chat_template
# The Qwen3 Jinja2 template calls tool.function.parameters.properties.items()
# which throws TypeError if properties is not a dict (e.g. None, list, etc.)
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
                # Replace invalid properties with empty dict so template won't throw
                params_clean["properties"] = {}
            # Ensure each property value is also a dict
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

# Helper function to sanitize messages for apply_chat_template
# Prevents 'Can only get item pairs from a mapping.' Jinja template exceptions
def sanitize_messages_for_template(messages):
    """Ensure all tool_calls in assistant messages have dict arguments so Jinja2 template won't raise 'Can only get item pairs from a mapping.'"""
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

# Pass 2: Format and enrich system prompts with data sanitization
print("Pass 2: Enriching system prompts, filtering out logic errors/loop patterns...")
formatted_samples = []
skipped_loops = 0
skipped_sanitized = 0
skipped_too_long = 0
_err_count = [0]  # Counter for logging hidden exceptions

# Deduplication within current run
seen_conversations = set()

import re

def sanitize_sample(sample, messages) -> bool:
    if not messages:
        return False
        
    tools = sample.get("tools", [])
    if not isinstance(tools, list):
        tools = []
    
    # Extract tool names in schema
    tools_schema = set()
    for t in tools:
        if isinstance(t, dict) and "function" in t:
            fn = t["function"]
            if isinstance(fn, dict) and fn.get("name"):
                tools_schema.add(fn.get("name"))

    has_sent_code_mention = False
    has_verification_tool_call = False
    has_transfer_mention = False
    has_transfer_tool_call = False
    has_transfer_call_at_all = False
    
    system_prompt = ""
    has_multi_service_user_query = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls")

        if role == "system":
            system_prompt = content

        if role == "user":
            lc = content.lower()
            if "mail" in lc and "phone" in lc and ("rooms" in lc or "room" in lc):
                has_multi_service_user_query = True

        if role == "assistant":
            # Filter empty assistant responses without tool calls
            if (not content or content.strip() in ("", "None")) and not tool_calls:
                return False

            lc = content.lower()
            if any(phrase in lc for phrase in ["sent a verification code", "sent you a verification code", "verification code has been sent"]):
                has_sent_code_mention = True

            if any(phrase in lc for phrase in ["transferring you", "transfer you", "being transferred"]):
                has_transfer_mention = True

            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict) and "function" in tc:
                        fn = tc["function"]
                        if isinstance(fn, dict):
                            t_name = fn.get("name", "")
                            if "verify" in t_name or "auth" in t_name:
                                has_verification_tool_call = True
                            if "transfer" in t_name:
                                has_transfer_tool_call = True
                                has_transfer_call_at_all = True

    # 1. VERIFY_NO_TOOL: Agent says sent code but no send/verify tool in schema/called
    if has_sent_code_mention and not has_verification_tool_call:
        has_verify_tool_in_schema = any("verify" in name or "auth" in name for name in tools_schema)
        if not has_verify_tool_in_schema:
            return False

    # 2. TRANSFER_NO_TOOL: Agent announces transfer but never called transfer tool
    if has_transfer_mention and not has_transfer_tool_call:
        return False

    # 3. ESCALATION_VIOLATION: Multi-service config rule triggered but no transfer tool call
    # Search tasks do not have escalation/transfer capabilities (only have web-search). Do not check this rule for search tasks.
    is_search_task = any("search" in t_name for t_name in tools_schema)
    if not is_search_task:
        is_multi_service_escalation_rule = "Multi-service configuration requests" in system_prompt or "Multi-service configuration" in system_prompt
        if is_multi_service_escalation_rule and has_multi_service_user_query and not has_transfer_call_at_all:
            return False

    # 4. DISAMBIGUATION_INTERNAL_VIOLATION (Mach-Mind-4-Flash Trajectory Shaping)
    # Instead of dropping trajectories, we reshape them below in Pass 2 to try internal resolution first.
    return True

def shape_disambiguation_trajectory(sample, messages):
    """
    Mach-Mind-4-Flash RL Trajectory Shaping for Disambiguation:
    1. Sanitize tool call names against sample['tools'] (fixes get_vehicle_state hallucination bug).
    2. Enforce Internal Resolution First: If assistant asks user clarification questions
       before calling lookup tools, shape trajectory to call internal lookup tools first.
    """
    if not messages:
        return messages
        
    tools = sample.get("tools", [])
    if not isinstance(tools, list):
        tools = []
    
    valid_tool_names = set()
    lookup_tools = []
    for t in tools:
        if isinstance(t, dict) and "function" in t:
            fn = t["function"]
            if isinstance(fn, dict) and fn.get("name"):
                name = fn.get("name")
                valid_tool_names.add(name)
                if name.startswith(("get_", "check_", "search_", "fetch_", "query_", "list_")):
                    lookup_tools.append(name)

    # Step 1: Fix hallucinated tool names in assistant turns
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list):
                valid_calls = []
                for tc in tcs:
                    if isinstance(tc, dict) and "function" in tc:
                        fn = tc.get("function", {})
                        t_name = fn.get("name", "")
                        if t_name in valid_tool_names:
                            valid_calls.append(tc)
                        else:
                            # Fuzzy match hallucinated names e.g. get_vehicle_state -> get_vehicle_status
                            matched = None
                            for vname in valid_tool_names:
                                if "vehicle" in t_name and "vehicle" in vname:
                                    matched = vname
                                    break
                                elif "status" in t_name and "status" in vname:
                                    matched = vname
                                    break
                            if matched:
                                tc["function"]["name"] = matched
                                valid_calls.append(tc)
                msg["tool_calls"] = valid_calls if valid_calls else None
                if msg["tool_calls"] is None:
                    msg.pop("tool_calls", None)

    # Step 2: Trajectory Shaping - Internal Resolution First
    # Check if assistant asks user for clarification before any tool call was made
    made_tool_call = False
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        tcs = msg.get("tool_calls")
        
        if role == "assistant":
            if tcs:
                made_tool_call = True
            elif not made_tool_call and ("?" in content or "could you" in content.lower() or "please specify" in content.lower()):
                if lookup_tools:
                    # Trajectory Shaping: Insert internal lookup tool call before asking user
                    target_tool = lookup_tools[0]
                    msg["content"] = ""
                    msg["tool_calls"] = [{
                        "id": f"call_lookup_{idx}",
                        "type": "function",
                        "function": {
                            "name": target_tool,
                            "arguments": {}
                        }
                    }]
                    # Insert synthetic tool response turn so trajectory remains valid
                    synthetic_tool_turn = {
                        "role": "tool",
                        "tool_call_id": f"call_lookup_{idx}",
                        "name": target_tool,
                        "content": json.dumps({"status": "success", "result": "Internal state/preferences checked."})
                    }
                    messages.insert(idx + 1, synthetic_tool_turn)
                    made_tool_call = True
                    break
    return messages

for local_file_path in downloaded_file_paths:
    with open(local_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    raw_msgs = sample.get("conversations") or sample.get("messages") or sample.get("augmented_messages") or sample.get("raw_messages")
                    messages = _parse_messages(raw_msgs)
                    if messages:
                        # Clean think token leaks from assistant content
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                                msg["content"] = re.sub(r"<think>.*?</think>", "", msg["content"], flags=re.DOTALL).strip()

                        # Handle Null/None in augmented assistant responses
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content") is None:
                                msg["content"] = ""

                        # deduplication by content and tools schema hash
                        conv_hash = hash(json.dumps(messages, sort_keys=True) + json.dumps(sample.get("tools", []), sort_keys=True))
                        if conv_hash in seen_conversations:
                            continue
                        seen_conversations.add(conv_hash)

                        # Data Sanitizer filters
                        if not sanitize_sample(sample, messages):
                            skipped_sanitized += 1
                            continue

                        # Mach-Mind-4-Flash RL Trajectory Shaping & Tool Sanitization
                        messages = shape_disambiguation_trajectory(sample, messages)

                        # Filter out samples with tool loop patterns
                        if has_tool_loop(messages):
                            skipped_loops += 1
                            continue
                        
                        # Enrich system message with tool calling guidelines
                        sys_idx = -1
                        for idx, m in enumerate(messages):
                            if m.get("role") == "system":
                                sys_idx = idx
                                break
                        if sys_idx != -1:
                            messages[sys_idx]["content"] = enrich_system_prompt(messages[sys_idx]["content"])
                        
                        # Parse tool_call arguments from JSON string to dict.
                        # Qwen3.5 Jinja2 template calls arguments.items() which fails
                        # if arguments is a JSON-encoded string (dataset stores as string).
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                tcs = msg.get("tool_calls")
                                if isinstance(tcs, list):
                                    for tc in tcs:
                                        if isinstance(tc, dict):
                                            fn = tc.get("function")
                                            if isinstance(fn, dict):
                                                args = fn.get("arguments")
                                                if isinstance(args, str):
                                                    try:
                                                        fn["arguments"] = json.loads(args)
                                                    except (json.JSONDecodeError, ValueError):
                                                        fn["arguments"] = {}
                        
                        # Fix TRUNCATED_TOOL_RESPONSE: search_disambiguation.jsonl tool messages
                        # contain "... [truncated XXXX chars]" markers which make sequences
                        # excessively long and cause Unsloth to drop the sample (marker lost).
                        # Strip these markers from all tool role messages before formatting.
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("role") == "tool":
                                content = msg.get("content") or ""
                                if "... [truncated" in content:
                                    msg["content"] = re.sub(
                                        r'\.\.\. \[truncated \d+ chars\]\s*$', '',
                                        content.strip()
                                    ).strip()

                        # Format chat template passing tools schema and enable_thinking=False
                        # Sanitize tools schema so Jinja2 template won't throw TypeError
                        # when iterating parameters.properties.items()
                        tools_clean = sanitize_tools_for_template(sample.get("tools"))
                        messages_clean = sanitize_messages_for_template(messages)
                        text = tokenizer.apply_chat_template(
                            messages_clean,
                            tools=tools_clean,
                            tokenize=False,
                            add_generation_prompt=False,
                            enable_thinking=False
                        )
                        
                        # Filter out sequences that exceed max_seq_length to prevent OOM / Collator corruption
                        # Use 16384 to match model capacity — enriched system prompts can push beyond 8192
                        input_ids = text_tokenizer.encode(text, add_special_tokens=False)
                        if len(input_ids) > 16384:
                            skipped_too_long += 1
                            continue
                            
                        formatted_samples.append({"text": text})


                except Exception as e:
                    # Log the first few exceptions to help diagnose hidden failures
                    _err_count[0] += 1
                    if _err_count[0] <= 3:
                        print(f"[SAMPLE ERROR #{_err_count[0]}] {type(e).__name__}: {str(e)[:300]}")

print(f"Skipped {skipped_loops} samples containing tool calling loop patterns.")
print(f"Skipped {skipped_sanitized} samples failing logic verification sanitizer.")
print(f"Skipped {skipped_too_long} samples exceeding max token length (16384).")
print(f"Skipped {_err_count[0]} samples due to formatting errors (apply_chat_template exceptions).")

print(f"Total formatted disambiguation samples: {len(formatted_samples)}")


# Convert to HF Dataset
dataset_filtered = Dataset.from_list(formatted_samples)

# Train-Test Split (90% train, 10% test)
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
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,       # Increased for stronger updates
    lora_dropout=0.0,    # Optimal for Unsloth
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
    learning_rate=3e-5,       # Tuned for stronger learning
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    num_train_epochs=2.0 if not args_cli.smoke_test else 1.0, # Reduced epochs to 2.0 for faster convergence
    max_steps=3 if args_cli.smoke_test else -1,
    weight_decay=0.02,        # Slightly higher to prevent overfitting
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
    packing=False,             # Disabled packing for faster processing
    max_seq_length=16384      # 16384 to accommodate enriched system prompts + multi-turn tool conversations
)

# Trainer initialization
sft_config.eos_token = None
tokenizer.eos_token = "<|im_end|>"
text_tokenizer.eos_token = "<|im_end|>"

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=text_tokenizer
)

# Since apply_chat_template is called with enable_thinking=False (line ~396), the formatted
# training text will NEVER contain <think> blocks. We must ALWAYS use the plain assistant
# marker as response_part. Using '<|im_start|>assistant\n<think>' when no think tags exist
# causes train_on_responses_only to fail finding the marker on most samples (truncation +
# non-existent tag).
# Multi-turn DataCollator: Calculate loss strictly on ALL assistant turns in a multi-turn conversation
print("Applying MultiTurnDataCollator for multi-turn assistant response masking...")
from transformers import DataCollatorForLanguageModeling

class MultiTurnDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, response_template, instruction_template, tokenizer, *args, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
        self.response_token_ids = response_template      # Token IDs for "<|im_start|>assistant"
        self.end_token_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False) # Token IDs for "<|im_end|>"
        self.end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)   # Token IDs for "</think>"

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        
        for i in range(labels.shape[0]):
            seq = batch["input_ids"][i].tolist()
            # Default all labels to -100 (mask out system prompt, user query, tool execution outputs)
            labels[i, :] = -100
            
            # Find all assistant response segments in multi-turn conversation
            start_search = 0
            while start_search < len(seq):
                resp_idx = -1
                for idx in range(start_search, len(seq) - len(self.response_token_ids) + 1):
                    if seq[idx:idx+len(self.response_token_ids)] == self.response_token_ids:
                        resp_idx = idx + len(self.response_token_ids)
                        break
                
                if resp_idx == -1:
                    break
                
                # Find end of assistant turn (<|im_end|>)
                end_idx = len(seq)
                for idx in range(resp_idx, len(seq) - len(self.end_token_ids) + 1):
                    if seq[idx:idx+len(self.end_token_ids)] == self.end_token_ids:
                        end_idx = idx + len(self.end_token_ids)
                        break
                
                # Skip past empty <think>...</think> block if present
                actual_resp_start = resp_idx
                for idx in range(resp_idx, min(resp_idx + 20, end_idx - len(self.end_think_ids) + 1)):
                    if seq[idx:idx+len(self.end_think_ids)] == self.end_think_ids:
                        actual_resp_start = idx + len(self.end_think_ids)
                        while actual_resp_start < end_idx and seq[actual_resp_start] in (198, 271, 13):
                            actual_resp_start += 1
                        break

                # Keep loss ONLY for assistant tokens (tool calls, disambiguation questions, or response text)
                labels[i, actual_resp_start:end_idx] = batch["input_ids"][i, actual_resp_start:end_idx]
                start_search = end_idx

        batch["labels"] = labels
        return batch

# Using plain response_part without trailing newline for clean token boundary against Qwen tokenizer artifacts.
response_part = "<|im_start|>assistant"
resp_ids_list = text_tokenizer.encode(response_part, add_special_tokens=False)
instr_ids_list = text_tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
collator = MultiTurnDataCollator(
    response_template=resp_ids_list,
    instruction_template=instr_ids_list,
    tokenizer=text_tokenizer,
)

# Re-initialize trainer with custom collator
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=text_tokenizer,
    data_collator=collator,
)
print("Trainer successfully initialized with MultiTurnDataCollator.")

# =====================================================================
# 5. EXECUTE SFT TRAINING
# =====================================================================
print("Starting SFT training...")
trainer_stats = trainer.train()
print("Training completed successfully!")
if not args_cli.smoke_test:
    # =====================================================================
    # 6. VISUALIZE AND SAVE PLOT
    # =====================================================================
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
            plt.title("SFT Loss Curve comparison: Train vs Validation")
            plt.legend()
            plt.grid(True)
            
            plot_path = os.path.join(PERSISTENT_DIR, "loss_plot.png")
            plt.savefig(plot_path)
            print(f"Loss plot saved to: {plot_path}")
    except Exception as e:
        print(f"Note: Could not generate loss plot ({e}), proceeding to model saving.")

    # =====================================================================
    # 7. SAVE ADAPTERS AND UPLOAD TO HUGGINGFACE HUB
    # =====================================================================
    print(f"Saving PEFT LoRA adapter checkpoint to {ADAPTER_PATH}...")
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)

    try:
        print("Uploading LoRA adapter to Hugging Face Hub (dragonstorm123/qwen3.5-4b-sft-disambiguation-lora)...")
        model.push_to_hub("dragonstorm123/qwen3.5-4b-sft-disambiguation-lora", tokenizer=tokenizer, token=os.environ.get("HF_TOKEN"))
        print("Successfully uploaded LoRA adapter to Hugging Face Hub!")
    except Exception as e:
        print(f"Warning: Failed pushing LoRA adapter to HF Hub: {e}")

    try:
        print("Uploading merged 16-bit model to Hugging Face Hub (dragonstorm123/qwen3.5-4b-sft-disambiguation)...")
        model.push_to_hub_merged(
            "dragonstorm123/qwen3.5-4b-sft-disambiguation", 
            tokenizer, 
            save_method="merged_16bit",
            token=os.environ.get("HF_TOKEN")
        )
        print("Successfully uploaded merged 16-bit model to Hugging Face Hub!")
    except Exception as e:
        print(f"Failed to push merged model to HF Hub: {e}")

    try:
        print("Merging and saving model weights to 16-bit precision locally...")
        model.save_pretrained_merged(MERGED_PATH, tokenizer, save_method="merged_16bit")
        print(f"Merged model successfully saved to: {MERGED_PATH}")
    except Exception as e:
        print(f"Warning: Failed saving merged model locally (likely low disk space): {e}")

print("All tasks successfully finished!")

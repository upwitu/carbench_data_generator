import os
import sys
import torch
import torch.utils

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

# Prevent transformers from loading native torchao internally and raising AttributeErrors
import transformers
transformers.utils.import_utils.is_torchao_available = lambda *args, **kwargs: False
transformers.utils.is_torchao_available = lambda *args, **kwargs: False
if hasattr(transformers.utils.import_utils, "_torchao_available"):
    transformers.utils.import_utils._torchao_available = False
print("Compatibility patches successfully applied!")

# Unset DDP environment variables for single GPU training
for key in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
    if key in os.environ:
        del os.environ[key]

# Force single GPU selection if needed, e.g., using CUDA_VISIBLE_DEVICES outside, or default to device 0
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =====================================================================
# 2. LOAD LIBRARIES
# =====================================================================
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server script
import matplotlib.pyplot as plt
from datasets import load_dataset
from unsloth import FastLanguageModel, train_on_responses_only
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen3.5-4B"
PERSISTENT_DIR = "./outputs"
ADAPTER_PATH = os.path.join(PERSISTENT_DIR, "sft_lora_adapter")
MERGED_PATH = os.path.join(PERSISTENT_DIR, "sft_merged_model")

os.makedirs(PERSISTENT_DIR, exist_ok=True)

# =====================================================================
# 3. LOAD DATASET (10,000 Samples)
# =====================================================================
print("Loading upwitu/trash_draft_am dataset from Hugging Face...")
dataset = load_dataset("upwitu/trash_draft_am", split="train", token=os.environ.get("HF_TOKEN"))
print(f"Original dataset size: {len(dataset)}")

# Select first 10,000 samples as requested
print("Selecting 10,000 samples for SFT...")
dataset_subset = dataset.select(range(min(10000, len(dataset))))

# Train-Test Split (90% train, 10% test)
dataset_split = dataset_subset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]
print(f"Split completed: {len(train_dataset)} train samples, {len(eval_dataset)} evaluation samples")

# =====================================================================
# 4. LOAD MODEL & CONFIG PEFT LORA
# =====================================================================
print("Loading Qwen 3.5 4B Model via Unsloth in BF16...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    load_in_4bit=False,  # Full precision (not quantized) since we are on A100
    trust_remote_code=True
)

# Extract actual text tokenizer if the loaded object is a Processor (e.g., Qwen3VLProcessor)
text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

# Force eos_token to be <|im_end|> to bypass vocabulary check errors in SFTTrainer
tokenizer.eos_token = "<|im_end|>"
text_tokenizer.eos_token = "<|im_end|>"

# Apply LoRA model wrappers
print("Configuring PEFT LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.0,  # Optimal for Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("LoRA configuration completed.")
model.print_trainable_parameters()

# =====================================================================
# 5. FORMAT CHAT TEMPLATE WITH TOOLS
# =====================================================================
def format_sft_data(row):
    tools = row.get("tools")
    # Use raw_messages for base task SFT
    messages = row.get("raw_messages")
    
    # Safely parse raw_messages if it is a string representation of a list
    if isinstance(messages, str):
        try:
            import json
            messages = json.loads(messages)
        except Exception:
            try:
                import ast
                messages = ast.literal_eval(messages)
            except Exception:
                pass
                
    # Safely parse tool_call arguments inside assistant messages to dict
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and "tool_calls" in msg:
                tc_list = msg["tool_calls"]
                if isinstance(tc_list, list):
                    for tc in tc_list:
                        if isinstance(tc, dict) and "function" in tc:
                            func = tc["function"]
                            if isinstance(func, dict) and "arguments" in func:
                                args = func["arguments"]
                                if isinstance(args, str):
                                    try:
                                        import json
                                        func["arguments"] = json.loads(args)
                                    except Exception:
                                        try:
                                            import ast
                                            func["arguments"] = ast.literal_eval(args)
                                        except Exception:
                                            pass
                
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools
    )
    return {"text": text}

print("Applying Qwen 3.5 Chat Template mapping to datasets...")
train_dataset = train_dataset.map(format_sft_data, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(format_sft_data, remove_columns=eval_dataset.column_names)

print("First sample formatted text preview:")
print(train_dataset[0]["text"][:600] + "\n...\n")

# =====================================================================
# 6. CONFIGURE SFT TRAINER & APPLY MASKING
# =====================================================================
sft_config = SFTConfig(
    dataset_text_field="text",
    output_dir=PERSISTENT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=2,       # Giảm xuống 2 để tránh OOM do không có FlashAttention-2
    gradient_accumulation_steps=16,      # Tăng lên 16 để giữ Global Batch Size = 32
    num_train_epochs=3.0,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=False,
    bf16=True,                           # Safe for A100
    optim="paged_adamw_8bit",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,                  # Save only the single best checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=5,
    report_to="none"
)

print("Trainer init: text_tokenizer.eos_token before force:", repr(text_tokenizer.eos_token))
tokenizer.eos_token = "<|im_end|>"
text_tokenizer.eos_token = "<|im_end|>"
print("Trainer init: text_tokenizer.eos_token after force:", repr(text_tokenizer.eos_token))

print("Trainer init: sft_config.eos_token before force:", repr(sft_config.eos_token))
sft_config.eos_token = None
print("Trainer init: sft_config.eos_token after force:", repr(sft_config.eos_token))

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=text_tokenizer
)

# Apply loss masking: Only calculate loss on assistant answers & thoughts
print("Applying loss masking filter to train on assistant responses only...")
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    # Mask until assistant starts and the think tag opens
    response_part="<|im_start|>assistant\n<think>\n",
)

# =====================================================================
# 7. EXECUTE SFT TRAINING
# =====================================================================
print("Starting SFT training...")
trainer_stats = trainer.train()
print("Training completed successfully!")

# =====================================================================
# 8. VISUALIZE AND SAVE PLOT
# =====================================================================
print("Generating loss curve plot...")
if hasattr(trainer, "state") and trainer.state.log_history:
    history = trainer.state.log_history
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    
    for log in history:
        step = log.get("step")
        if "loss" in log:
            train_steps.append(step)
            train_losses.append(log["loss"])
        if "eval_loss" in log:
            eval_steps.append(step)
            eval_losses.append(log["eval_loss"])
            
    try:
        if plt is not None:
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
# 9. SAVE ADAPTERS AND SAVE MERGED MODEL
# =====================================================================
print(f"Saving PEFT LoRA adapter checkpoint to {ADAPTER_PATH}...")
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)

print("Merging and saving model weights to 16-bit precision...")
model.save_pretrained_merged(MERGED_PATH, tokenizer, save_method="merged_16bit")
print(f"Merged model successfully saved to: {MERGED_PATH}")

# =====================================================================
# 10. UPLOAD TO HUGGINGFACE HUB
# =====================================================================
try:
    print("Uploading merged model to Hugging Face Hub...")
    model.push_to_hub_merged(
        "dragonstorm123/qwen3.5-4b-sft-base", 
        tokenizer, 
        save_method="merged_16bit",
        token=os.environ.get("HF_TOKEN")
    )
    print("Successfully uploaded to Hugging Face Hub (dragonstorm123/qwen3.5-4b-sft-base)!")
except Exception as e:
    print(f"Failed to push to HF Hub: {e}")

print("All tasks successfully finished!")

# Hướng dẫn Chạy Thử (Smoke Test) trên Jupyter Notebook

Tài liệu này hướng dẫn bạn cách thực hiện các bước chạy thử (Smoke Test) trực tiếp trên Jupyter Notebook của server để phát hiện sớm các lỗi cú pháp, cấu trúc dữ liệu không hợp lệ (như lỗi `TypeError` của tools, lỗi template) và kiểm tra tính đúng đắn của nhãn Loss (Response-only loss) trước khi tiến hành chạy phân tán DDP thực sự.

---

## Mục tiêu của Smoke Test
1. **Kiểm tra dữ liệu:** Đảm bảo dữ liệu JSONL trong `data/` được parse đúng và không gây lỗi khi chạy qua chat template của tokenizer.
2. **Kiểm tra Model & Tokenizer:** Đảm bảo `unsloth` tải và nạp mô hình/LoRA thành công mà không bị tràn bộ nhớ (VRAM).
3. **Kiểm tra Nhãn Loss (Masking):** Đảm bảo cơ chế `train_on_responses_only` hoạt động đúng (che các phần hội thoại của User/System và chỉ tính Loss trên phản hồi của Assistant).
4. **Kiểm tra Huấn luyện:** Chạy thử 2-3 bước (steps) huấn luyện để chắc chắn không phát sinh ngoại lệ (Exceptions).

---

## Các Cell chạy thử trên Jupyter Notebook (Chạy từng Cell một)

### Cell 1: Nạp cấu hình và giả lập môi trường Single-GPU
Đoạn code này sẽ đọc cấu hình từ file [ddp_config.yml](file:///mnt/hungpv/car_bench_sft/car_bench_notebook/llm-training/ddp_config.yml) và tắt các biến môi trường DDP của PyTorch để tránh xung đột khi chạy trên Jupyter:

```python
import os
import yaml
import pandas as pd
import json

# Tắt DDP để test trên môi trường Single-GPU của Jupyter
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

# Đọc cấu hình huấn luyện
CONFIG_PATH = "ddp_config.yml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("✅ Đã nạp cấu hình thành công!")
print(f"Mô hình đích: {config['model']['name']}")
print(f"File dữ liệu mẫu: {config['datasets']['sample']['path']}")
```

---

### Cell 2: Kiểm tra cấu trúc dữ liệu và xử lý Chat Template
Cell này sẽ đọc 5 dòng dữ liệu mẫu, áp dụng `apply_chat_template` và tokenizer để kiểm tra xem có bất kỳ dữ liệu nào bị lỗi Schema (như lỗi `properties` hoặc `tool_calls` không hợp lệ):

```python
from transformers import AutoTokenizer

# Tải tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    config["model"]["name"], 
    trust_remote_code=True
)

# Đọc thử file dữ liệu
dataset_path = config["datasets"]["sample"]["path"]
print(f"Đọc dữ liệu từ: {dataset_path}")
df = pd.read_json(dataset_path, lines=True).head(5)

# Chạy thử chuyển đổi
print("Chạy thử áp dụng Chat Template và Tokenize trên 5 dòng đầu:")
for i, row in enumerate(df.itertuples(index=False), 1):
    try:
        # Lấy tools nếu có
        tools = getattr(row, "tools", None) or None
        convs = getattr(row, "conversations")
        
        # Áp dụng template
        text = tokenizer.apply_chat_template(
            convs,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
            tools=tools,
        )
        # Tokenize thử
        enc = tokenizer(text, truncation=True, max_length=1024)
        print(f"  [Dòng {i}] Độ dài tokens: {len(enc['input_ids'])} -> OK")
        
    except Exception as e:
        print(f"  ❌ [Dòng {i}] BỊ LỖI: {e}")
```

---

### Cell 3: Nạp Mô hình Unsloth LoRA (Chế độ Tiết kiệm VRAM)
Để chạy thử nhanh trên Jupyter mà không lo bị tràn VRAM (OOM) làm sập kernel, ta ép cấu hình tải mô hình dạng **4-bit** (`load_in_4bit = True`):

```python
from unsloth import FastLanguageModel
import torch

print("Loading model in 4-bit for smoke testing...")
model, processor = FastLanguageModel.from_pretrained(
    model_name=config["model"]["name"],
    max_seq_length=2048, # Giới hạn độ dài ngắn để chạy test nhanh
    dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
    load_in_4bit=True,   # Ép 4-bit để test nhanh trên 1 GPU
)
tokenizer = processor.tokenizer

# Áp dụng cấu hình LoRA PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora"]["r"],
    target_modules=config["lora"]["target_modules"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=0.0,    # Unsloth tối ưu tốt nhất ở 0.0
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("✅ Nạp mô hình & LoRA thành công!")
```

---

### Cell 4: Tạo tập dữ liệu Dummy (5 dòng) để Test SFTTrainer
Tạo nhanh một Dataset nhỏ để đưa vào Trainer:

```python
from datasets import Dataset

input_ids_list = []
attention_mask_list = []

for row in df.itertuples(index=False):
    tools = getattr(row, "tools", None) or None
    convs = getattr(row, "conversations")
    text = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=False, enable_thinking=False, tools=tools
    )
    enc = tokenizer(text, truncation=True, max_length=2048)
    input_ids_list.append(enc["input_ids"])
    attention_mask_list.append(enc["attention_mask"])

dummy_dataset = Dataset.from_dict({
    "input_ids": input_ids_list,
    "attention_mask": attention_mask_list
})
print(f"Tạo dummy dataset thành công: {len(dummy_dataset)} dòng.")
```

---

### Cell 5: Thiết lập Masking Loss (Chỉ tính Loss trên phản hồi của Assistant)
Đây là bước quan trọng nhất để xem nhãn `-100` (bỏ qua loss) có được gán đúng cho các câu thoại của User/System hay không:

```python
from trl import SFTTrainer, SFTConfig
from unsloth import train_on_responses_only

# Cấu hình huấn luyện dummy
training_args = SFTConfig(
    output_dir="/tmp/dummy_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_length=2048,
    packing=False,
    dataset_text_field=None,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dummy_dataset,
    args=training_args,
)

# Áp dụng bộ lọc Masking Loss
response_cfg = config["training"].get("response_format", {})
instruction_part = response_cfg.get("instruction_part", "<|im_start|>user\n")
response_part = response_cfg.get("response_part", "<|im_start|>assistant\n<think>\n\n</think>\n\n")

trainer = train_on_responses_only(
    trainer,
    instruction_part=instruction_part,
    response_part=response_part,
)

# Giải mã thử mẫu đầu tiên để xem phần nào được tính Loss
# Những phần bị thay bằng khoảng trắng/không có nhãn là những phần bị bỏ qua (labels = -100)
example_answer = tokenizer.decode(
    [tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]
).replace(tokenizer.pad_token, " ")

print("=== PHẦN MÔ HÌNH SẼ TÍNH LOSS (Không bị mask) ===")
print(example_answer)
```

---

### Cell 6: Chạy thử huấn luyện 2 bước (Steps)
Khởi động quá trình train thực sự trong 2 steps để đảm bảo backward pass và cập nhật trọng số hoạt động trơn tru:

```python
# Cập nhật số step chạy thử bằng 2
trainer.args.max_steps = 2

print("🚀 Khởi chạy huấn luyện thử 2 steps...")
try:
    trainer.train()
    print("🎉 KHỞI CHẠY THỬ THÀNH CÔNG! Sẵn sàng chạy DDP với train.sh")
except Exception as e:
    print(f"❌ LỖI KHI HUẤN LUYỆN: {e}")
```

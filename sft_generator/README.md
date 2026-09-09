# HƯỚNG DẪN PIPELINE SINH DỮ LIỆU SFT CARBENCH (WINNER-BASED SYNTHESIS)

Tài liệu hướng dẫn triển khai hệ thống sinh 2 bộ dữ liệu SFT CarBench chất lượng cao từ kinh nghiệm các đội Winner (`10cars`, `FreudeDrive`, `Proxima Ultra`), quản lý qua `uv`, thực thi bất đồng bộ `asyncio concurrent requests`, streaming lưu trực tiếp vào JSONL và chạy huấn luyện SFT trên server.

---

## 1. Cấu Trúc 2 Bộ Dữ Liệu SFT Đột Phá

| Bộ Dữ Liệu | Nguồn Cảm Hứng | Định Dạng Dữ Liệu | Ràng Buộc & Điểm Khác Biệt |
| :--- | :--- | :--- | :--- |
| **Bộ 1: Multi-Role Verified JSON** | `10cars` & `FreudeDrive` | OpenAI Tool Calling JSON (`messages` + `tools`) | - Kèm chuỗi `<think>` CoT phân tích 4 bước.<br>- **L3 Pre-Flight Gate:** Bắt buộc *Read-Before-Write*, chặn lệnh ghi đè sửa sai tạm thời.<br>- **Confirmation Gate:** Bắt buộc xin xác nhận mở cốp, gửi email, bật đèn pha. |
| **Bộ 2: Programmatic CodeAct Python** | `Proxima Ultra` | CodeAct Python scripts (`execute_python`) | - Xâu chuỗi nhiều tool trong 1 lượt bằng code Python.<br>- **Unknown-Value Sentinels:** Bắt lỗi và thừa nhận giới hạn khi dữ liệu trả về `"unknown"`.<br>- **Response Obligations:** Tự động đính kèm cảnh báo điều hòa (mở cửa sổ >25%) và chênh lệch nhiệt độ (>3°C). |

---

## 2. Cài Đặt & Đồng Bộ Môi Trường (1 Chạm Với `uv`)

### Bước 1: Cài đặt thư viện qua `uv`
Tại thư mục gốc dự án:
```bash
uv sync
```

### Bước 2: Cấu hình biến môi trường (`.env`)
Tạo file `.env` (hoặc copy từ `sft_generator/.env.example`):
```bash
cp sft_generator/.env.example .env
```

Điền thông tin API và Hugging Face Token:
```ini
# 1. Hugging Face
HF_TOKEN=hf_your_token_here
HF_DATASET_REPO=upwitu/carbench_sft_winner_dataset

# 2. LLM API Endpoint (OpenAI-compatible)
# Gợi ý chỗ lấy API: OpenRouter (https://openrouter.ai) hoặc DeepSeek (https://api.deepseek.com)
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-your-key-here
CAR_BENCH_MODEL=google/gemma-2-27b-it

# 3. Tuning
CONCURRENCY_LIMIT=30
VARIATIONS_PER_TASK=10
```

---

## 3. Lệnh Chạy Sinh Dữ Liệu (Data Generation Commands)

### 1. Sinh toàn bộ 2 bộ dữ liệu (Khuyên dùng):
```bash
uv run python -m sft_generator.main --dataset-type all --concurrency 30 --variations 10 --upload-hf
```

### 2. Chỉ sinh Bộ 1 (Multi-Role JSON Tool-Calling):
```bash
uv run python -m sft_generator.main --dataset-type multi_role_json --concurrency 30 --variations 10
```

### 3. Chỉ sinh Bộ 2 (Programmatic CodeAct Python):
```bash
uv run python -m sft_generator.main --dataset-type codeact_python --concurrency 30 --variations 10
```

> **Cơ chế Streaming & Resume:**  
> Dữ liệu được ghi ngay vào `sft_generator/outputs/*.jsonl` sau mỗi task hoàn thành. Nếu tiến trình bị ngắt giữa chừng, khi chạy lại script sẽ tự động bỏ qua các task đã sinh xong và tiếp tục phần còn lại.

---

## 4. Hướng Dẫn Huấn Luyện SFT Trên Server (`hungpv@118.138.238.214`)

Trên server đã có sẵn GPU lớn và mô hình `Qwen3.5-4B` / `Qwen2.5-7B`.

### Bước 1: SSH vào server
```bash
ssh hungpv@118.138.238.214
```

### Bước 2: Clone hoặc đồng bộ mã nguồn vào thư mục làm việc
```bash
cd /home/hungpv/car-bench-ijcai-vsf
git pull origin main
```

### Bước 3: Chuẩn bị file cấu hình huấn luyện SFT
Chỉnh sửa file `llm-training/ddp_config.yml` để trỏ vào dataset mới sinh:
```yaml
datasets:
  winner_sft_data:
    path: "sft_generator/outputs/carbench_sft_multirole_json.jsonl"
    sample_ratio: 1.0

model:
  name: "Qwen/Qwen2.5-7B-Instruct"  # Hoặc Qwen/Qwen3.5-4B
  max_seq_length: 16384
  dtype: "bfloat16"
  load_in_4bit: false

lora:
  r: 16
  alpha: 32
  dropout: 0.0
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.5e-5
  num_train_epochs: 2.0
  optim: "adamw_8bit"
  output_dir: "models/qwen3.5_4b_carbench_winner_sft"
```

### Bước 4: Chạy huấn luyện SFT phân tán (Multi-GPU)
```bash
cd /home/hungpv/car-bench-ijcai-vsf/llm-training
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash train.sh
```

---

## 5. Chạy Đánh Giá CarBench Đo Lường Cải Thiện $Pass^3$

Sau khi model được huấn luyện xong:
```bash
# Đánh giá 3 trials trên Benchmark
bash scripts/run_bench_base.sh
bash scripts/run_bench_disambig.sh
bash scripts/run_bench_hallu.sh

# Sinh báo cáo so sánh độ tăng trưởng Pass^3
python generate_report.py
```

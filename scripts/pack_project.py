import os
import zipfile
from pathlib import Path

# Cac thu muc va file muon dong goi
INCLUDE_PATHS = [
    "llm-training",
    "src",
    "scenarios",
    "tests",
    "docs",
    "README.md",
    "generate_compose.py",
    "pyproject.toml",
    "uv.lock",
]

# Cac phan mo rong file muon loai bo
EXCLUDE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".log", ".pt", ".bin", ".safetensors", ".zip", ".tar.gz", ".tgz"
}

# Cac thu muc con muon loai bo
EXCLUDE_DIRS = {
    "__pycache__", ".ipynb_checkpoints", ".venv", "output", "temp", "cache", "checkpoint", "checkpoints"
}

def zip_project(output_filename="car_bench_delivery.zip"):
    root_dir = Path(__file__).resolve().parent.parent # Lay root tu scripts/pack_project.py
    output_path = root_dir / output_filename
    
    print(f"Bắt đầu đóng gói dự án từ: {root_dir}")
    print(f"File zip đầu ra sẽ lưu tại: {output_path}")
    
    zip_count = 0
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path_str in INCLUDE_PATHS:
            full_path = root_dir / path_str
            if not full_path.exists():
                print(f"Cảnh báo: Không tìm thấy {path_str}, bỏ qua.")
                continue
                
            if full_path.is_file():
                # Neu la file, ghi truc tiep
                zipf.write(full_path, arcname=path_str)
                zip_count += 1
                print(f"  + Zip file: {path_str}")
            elif full_path.is_dir():
                # Neu la thu muc, duyet de ghi tat ca file con
                for root, dirs, files in os.walk(full_path):
                    # Loai bo cac thu muc khong mong muon tai cho (in-place modification de os.walk skip)
                    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
                    
                    for file in files:
                        file_path = Path(root) / file
                        # Kiem tra extension
                        if file_path.suffix.lower() in EXCLUDE_EXTENSIONS:
                            continue
                        # Kiem tra file tam thoi
                        if file.startswith(".") or file.startswith("~$"):
                            continue
                            
                        # Lay duong dan relative tu root de ghi vao zip
                        rel_path = file_path.relative_to(root_dir)
                        zipf.write(file_path, arcname=rel_path)
                        zip_count += 1
                        
                print(f"  + Zip directory: {path_str}")
                
    print(f"Đóng gói thành công! Tổng cộng đã thêm {zip_count} file vào {output_filename}.")

if __name__ == "__main__":
    zip_project()

import os
import logging
from pathlib import Path
from typing import Optional, Any, List, Union
from huggingface_hub import HfApi, login
from sft_generator.config import settings

logger = logging.getLogger(__name__)


def upload_dataset_to_huggingface(
    file_path: Optional[Any] = None,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    hf_token: Optional[str] = None,
    files_to_upload: Optional[list] = None,
) -> bool:
    """Pushes local JSONL dataset file(s) to a Hugging Face Dataset repo."""
    target_repo = repo_id or settings.HF_DATASET_REPO
    auth_token = token or hf_token or settings.HF_TOKEN or os.getenv("HF_TOKEN")

    if not auth_token or auth_token == "your_huggingface_token_here":
        logger.warning("No valid HF_TOKEN found. Skipping upload to Hugging Face Hub.")
        return False

    targets = []
    if files_to_upload:
        targets.extend([Path(p) for p in files_to_upload if Path(p).exists()])
    elif file_path and Path(file_path).exists():
        targets.append(Path(file_path))

    if not targets:
        logger.warning("No existing files to upload to Hugging Face Hub.")
        return False

    try:
        api = HfApi(token=auth_token)
        api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True)

        for target in targets:
            logger.info(f"Uploading {target.name} to Hugging Face dataset '{target_repo}'...")
            api.upload_file(
                path_or_fileobj=str(target),
                path_in_repo=target.name,
                repo_id=target_repo,
                repo_type="dataset",
            )
            logger.info(f"Successfully uploaded {target.name} to https://huggingface.co/datasets/{target_repo}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to Hugging Face: {e}")
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        fpath = Path(sys.argv[1])
        upload_dataset_to_huggingface(fpath)
    else:
        print("Usage: python -m sft_generator.upload_to_hf <path_to_jsonl>")

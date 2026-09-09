"""CLI Entrypoint for CarBench SFT Data Generator.
Usage:
    python -m sft_generator.main --dataset-type all --concurrency 10 --variations 10 --model deepseek-ai/deepseek-v4-flash-0731
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from sft_generator.config import settings
from sft_generator.generator import DatasetGenerator
from sft_generator.upload_to_hf import upload_dataset_to_huggingface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sft_generator")


def parse_args():
    parser = argparse.ArgumentParser(description="CarBench SFT Winner Dataset Generator (NVIDIA NIM & OpenAI Compatible)")
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["multi_role_json", "codeact_python", "all"],
        default="all",
        help="Which winner dataset to synthesize (default: all)",
    )
    parser.add_argument(
        "--filter-task-type",
        type=str,
        default="all",
        help="Filter seed tasks by category: base, disambiguation, hallucination, all, or comma-separated e.g. 'disambiguation,hallucination'",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=settings.CONCURRENCY_LIMIT,
        help=f"Async concurrency limit (default: {settings.CONCURRENCY_LIMIT})",
    )
    parser.add_argument(
        "--variations",
        "--variations-per-task",
        dest="variations",
        type=int,
        default=settings.VARIATIONS_PER_TASK,
        help=f"Number of synthetic variations per seed task (default: {settings.VARIATIONS_PER_TASK})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.CAR_BENCH_MODEL,
        help=f"LLM model name on NVIDIA NIM (default: {settings.CAR_BENCH_MODEL})",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=settings.OPENAI_API_BASE,
        help=f"NVIDIA NIM or OpenAI-compatible API base URL (default: {settings.OPENAI_API_BASE})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="NVIDIA NIM API key (nvapi-...) or comma-separated list of keys",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=settings.REQUEST_DELAY_INTERVAL,
        help=f"Proactive dispatch throttle delay in seconds (default: {settings.REQUEST_DELAY_INTERVAL}s)",
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=settings.MAX_RPM,
        help=f"Strict maximum requests per minute per API key (default: {settings.MAX_RPM})",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable DeepSeek thinking mode (default: False to minimize token consumption & cost)",
    )
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Automatically push generated datasets to Hugging Face Hub when done or on exit",
    )
    return parser.parse_args()


async def async_main():
    args = parse_args()
    # Auto-route API base if DeepSeek model is used with default NVIDIA NIM endpoint
    if "deepseek" in args.model.lower() and ("nvidia.com" in args.api_base.lower() or args.api_base == "https://integrate.api.nvidia.com/v1"):
        logger.info("Detected DeepSeek model: automatically routing API Base to 'https://api.deepseek.com/v1' with 1000 RPM limit.")
        args.api_base = "https://api.deepseek.com/v1"
        if args.max_rpm == 38:
            args.max_rpm = 1000

    settings.CONCURRENCY_LIMIT = args.concurrency
    settings.VARIATIONS_PER_TASK = args.variations
    settings.CAR_BENCH_MODEL = args.model
    settings.OPENAI_API_BASE = args.api_base
    settings.REQUEST_DELAY_INTERVAL = args.delay
    settings.MAX_RPM = args.max_rpm
    settings.THINKING_MODE = args.thinking
    if args.api_key:
        settings.OPENAI_API_KEY = args.api_key
        settings.NVIDIA_API_KEY = args.api_key

    active_keys = settings.get_api_keys()
    masked_keys = [k[:6] + "..." + k[-4:] if len(k) > 10 else "key" for k in active_keys]

    logger.info("=" * 70)
    logger.info("🚗 CAR-BENCH WINNER SFT DATA GENERATOR (DEEPSEEK & NIM READY) 🚗")
    logger.info(f"Model: {settings.CAR_BENCH_MODEL}")
    logger.info(f"API Base: {settings.OPENAI_API_BASE}")
    logger.info(f"Active Key Pool: {len(active_keys)} keys ({', '.join(masked_keys)})")
    logger.info(f"Dataset Type: {args.dataset_type}")
    logger.info(f"Concurrency Limit: {settings.CONCURRENCY_LIMIT}")
    logger.info(f"Max Rate Limit: {settings.MAX_RPM} RPM / key (Strict Sliding-Window)")
    logger.info(f"Dispatch Delay (Anti-Burst): {settings.REQUEST_DELAY_INTERVAL}s")
    logger.info(f"Thinking Mode: {settings.THINKING_MODE} (Non-thinking saves 70%+ token cost)")
    logger.info(f"Variations per seed task: {settings.VARIATIONS_PER_TASK}")
    logger.info(f"Output directory: {settings.OUTPUT_DIR}")
    logger.info("=" * 70)

    generator = DatasetGenerator(concurrency_limit=settings.CONCURRENCY_LIMIT)
    seed_tasks = generator.load_seed_tasks()

    if args.filter_task_type and args.filter_task_type.lower() != "all":
        filters = [f.strip().lower() for f in args.filter_task_type.split(",") if f.strip()]
        seed_tasks = [
            t for t in seed_tasks
            if any(f in str(t.get("task_type", "")).lower() or f in str(t.get("task_id", "")).lower() for f in filters)
        ]
        logger.info(f"Filtered to {len(seed_tasks)} seed tasks matching {filters}")

    if not seed_tasks:
        logger.error(f"No seed tasks found in {settings.SEED_DATA_DIR}. Please provide raw tasks!")
        return

    generated_files = []

    try:
        if args.dataset_type in ["multi_role_json", "all"]:
            logger.info("\n--- Starting Dataset 1: Multi-Role Verified JSON SFT ---")
            out1 = await generator.generate_dataset_1_multirole_json(
                seed_tasks, variations_per_task=settings.VARIATIONS_PER_TASK
            )
            generated_files.append(out1)

        if args.dataset_type in ["codeact_python", "all"]:
            logger.info("\n--- Starting Dataset 2: Programmatic CodeAct Python SFT ---")
            out2 = await generator.generate_dataset_2_codeact_python(
                seed_tasks, variations_per_task=settings.VARIATIONS_PER_TASK
            )
            generated_files.append(out2)

        logger.info("\n" + "=" * 70)
        logger.info("🎉 DATASET GENERATION PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        for f in generated_files:
            logger.info(f"  - Generated file: {f}")
        logger.info("=" * 70)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.warning("\n" + "=" * 70)
        logger.warning("🛑 Interrupted by user (Ctrl+C). All generated samples are safely preserved on disk!")
        logger.warning("=" * 70)
        # Check files that exist in output directory
        for f in settings.OUTPUT_DIR.glob("*.jsonl"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    count = sum(1 for line in fp if line.strip())
                logger.info(f"  - Saved file: {f.name} ({count} samples)")
                if f not in generated_files:
                    generated_files.append(f)
            except Exception:
                pass

    finally:
        await generator.engine.close()

        if args.upload_hf and generated_files:
            logger.info("\n--- Uploading current dataset checkpoints to Hugging Face Hub ---")
            upload_dataset_to_huggingface(
                repo_id=settings.HF_DATASET_REPO,
                token=settings.HF_TOKEN,
                files_to_upload=generated_files,
            )


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n[INFO] Graceful shutdown complete. Data intact.")
        sys.exit(0)


if __name__ == "__main__":
    main()

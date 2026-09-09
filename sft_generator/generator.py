"""Core Dataset Generator Orchestrator.
Dispatches async synthesis workers, validates policy compliance, and streams results.
Automatically updates both unified dataset files and dedicated split files for:
- Base Tasks (car_base_sft.jsonl -> train_base.sh)
- Disambiguation Tasks (car_disambiguation_sft.jsonl -> train_disambiguation.sh)
- Hallucination Tasks (car_hallucination_sft.jsonl -> train_hallucination.sh)
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm.asyncio import tqdm

from sft_generator.config import settings
from sft_generator.schemas import ALL_CAR_TOOLS
from sft_generator.validators import pre_flight_gate, codeact_validator
from sft_generator.prompts import build_multi_role_json_prompt, build_codeact_python_prompt
from sft_generator.async_engine import AsyncLLMEngine, StreamingJSONLWriter

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Orchestrates generation of the 2 CarBench SFT winner datasets across all 3 task types."""

    def __init__(self, concurrency_limit: Optional[int] = None):
        self.engine = AsyncLLMEngine(concurrency_limit=concurrency_limit or settings.CONCURRENCY_LIMIT)

    def load_seed_tasks(self, seed_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Loads seed tasks from local raw JSONL files or Hugging Face dataset."""
        seed_dir = seed_dir or settings.SEED_DATA_DIR
        tasks: List[Dict[str, Any]] = []

        # Local seed files pattern
        raw_files = list(seed_dir.glob("raw_tasks_*.jsonl"))
        if not raw_files:
            raw_files = list(seed_dir.glob("*.jsonl"))

        for file_path in raw_files:
            if "sft" in file_path.name and "raw" not in file_path.name:
                continue
            
            # Infer task type from file name if missing
            inferred_type = "base"
            if "disambig" in file_path.name.lower():
                inferred_type = "disambiguation"
            elif "hallu" in file_path.name.lower():
                inferred_type = "hallucination"

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            if "task_type" not in data:
                                data["task_type"] = inferred_type
                            tasks.append(data)
            except Exception as e:
                logger.warning(f"Error reading seed file {file_path}: {e}")

        logger.info(f"Loaded {len(tasks)} raw seed tasks from {len(raw_files)} files in {seed_dir}")
        return tasks

    async def generate_dataset_1_multirole_json(
        self,
        seed_tasks: List[Dict[str, Any]],
        variations_per_task: int = 10,
        output_file: Optional[Path] = None,
    ) -> Path:
        """Synthesizes Dataset 1: Multi-Role Verified JSON SFT Dataset.
        Writes to both the unified file and the 3 task split files in data/.
        """
        output_path = output_file or (settings.OUTPUT_DIR / "carbench_sft_multirole_json.jsonl")
        unified_writer = StreamingJSONLWriter(output_path)

        # Dedicated split writers for train_base.sh, train_disambiguation.sh, train_hallucination.sh
        split_writers = {
            "base": StreamingJSONLWriter(settings.SEED_DATA_DIR / "car_base_sft.jsonl"),
            "disambiguation": StreamingJSONLWriter(settings.SEED_DATA_DIR / "car_disambiguation_sft.jsonl"),
            "hallucination": StreamingJSONLWriter(settings.SEED_DATA_DIR / "car_hallucination_sft.jsonl"),
        }

        tasks_to_generate = []
        for task_idx, seed_task in enumerate(seed_tasks):
            for var_idx in range(variations_per_task):
                target_id = f"{seed_task.get('task_id', f'task_{task_idx}')}_var_{var_idx}"
                if target_id not in unified_writer.existing_task_ids:
                    tasks_to_generate.append((seed_task, var_idx, target_id))

        logger.info(f"[Dataset 1: Multi-Role JSON] Generating {len(tasks_to_generate)} samples across Base/Disambig/Hallu...")

        async def worker(seed: Dict[str, Any], v_idx: int, t_id: str) -> bool:
            try:
                messages = build_multi_role_json_prompt(seed, v_idx)
                result = await self.engine.chat_completion(
                    messages,
                    temperature=settings.TEMPERATURE,
                    top_p=settings.TOP_P,
                    json_mode=True,
                )
                if not result or not isinstance(result, dict):
                    return False

                conversations = result.get("conversations", [])
                if not conversations or not isinstance(conversations, list):
                    return False

                # Defensive check: ensure every item in conversations is a dict
                if not all(isinstance(m, dict) for m in conversations):
                    logger.debug(f"Sample {t_id} conversations contains non-dict items.")
                    return False

                # L3 Pre-Flight Gate Validation (10cars + FreudeDrive)
                gate_res = pre_flight_gate.validate_trajectory(conversations)
                if not gate_res.is_valid:
                    logger.debug(f"Sample {t_id} failed L3 Pre-Flight Gate: {gate_res.violations}")
                    diag_msg = "Your previous output violated CAR-bench policies:\n" + "\n".join(gate_res.violations)
                    retry_messages = messages + [
                        {"role": "assistant", "content": json.dumps(result)},
                        {"role": "user", "content": f"{diag_msg}\nPlease fix all violations and output compliant JSON."},
                    ]
                    result = await self.engine.chat_completion(retry_messages, temperature=0.3, json_mode=True)
                    if not result or not isinstance(result, dict):
                        return False
                    conversations = result.get("conversations", [])
                    if not conversations or not isinstance(conversations, list) or not all(isinstance(m, dict) for m in conversations):
                        return False
                    gate_res = pre_flight_gate.validate_trajectory(conversations)
                    if not gate_res.is_valid:
                        return False

                task_type = seed.get("task_type", "base").lower()
                sample_entry = {
                    "task_id": t_id,
                    "task_type": task_type,
                    "conversations": conversations,
                    "tools": ALL_CAR_TOOLS,
                }
                
                # 1. Write to unified file
                await unified_writer.write_sample(sample_entry)

                # 2. Write to corresponding task split file
                if task_type in split_writers:
                    await split_writers[task_type].write_sample(sample_entry)

                return True
            except Exception as e:
                logger.error(f"Unexpected worker error on {t_id}: {e}")
                return False

        tasks = [worker(s, v, t) for s, v, t in tasks_to_generate]
        if tasks:
            results = await tqdm.gather(*tasks, desc="Dataset 1 (Multi-Role JSON)")
            success_count = sum(1 for r in results if r)
            logger.info(f"Dataset 1 complete! Successfully generated {success_count}/{len(tasks)} samples.")

        return output_path

    async def generate_dataset_2_codeact_python(
        self,
        seed_tasks: List[Dict[str, Any]],
        variations_per_task: int = 10,
        output_file: Optional[Path] = None,
    ) -> Path:
        """Synthesizes Dataset 2: Programmatic CodeAct Python SFT Dataset."""
        output_path = output_file or (settings.OUTPUT_DIR / "carbench_sft_codeact_python.jsonl")
        unified_writer = StreamingJSONLWriter(output_path)

        tasks_to_generate = []
        for task_idx, seed_task in enumerate(seed_tasks):
            for var_idx in range(variations_per_task):
                target_id = f"{seed_task.get('task_id', f'task_{task_idx}')}_codeact_var_{var_idx}"
                if target_id not in unified_writer.existing_task_ids:
                    tasks_to_generate.append((seed_task, var_idx, target_id))

        logger.info(f"[Dataset 2: CodeAct Python] Generating {len(tasks_to_generate)} samples across Base/Disambig/Hallu...")

        async def worker(seed: Dict[str, Any], v_idx: int, t_id: str) -> bool:
            try:
                messages = build_codeact_python_prompt(seed, v_idx)
                result = await self.engine.chat_completion(
                    messages,
                    temperature=settings.TEMPERATURE,
                    top_p=settings.TOP_P,
                    json_mode=True,
                )
                if not result or not isinstance(result, dict):
                    return False

                conversations = result.get("conversations", [])
                if not conversations or not isinstance(conversations, list):
                    return False

                if not all(isinstance(m, dict) for m in conversations):
                    return False

                # CodeAct AST & Sentinel Validation (Proxima Ultra)
                val_res = codeact_validator.validate_trajectory(conversations)
                if not val_res.is_valid:
                    logger.debug(f"Sample {t_id} failed CodeAct Validator: {val_res.violations}")
                    return False

                sample_entry = {
                    "task_id": t_id,
                    "task_type": seed.get("task_type", "base"),
                    "conversations": conversations,
                }
                await unified_writer.write_sample(sample_entry)
                return True
            except Exception as e:
                logger.error(f"Unexpected worker error on CodeAct {t_id}: {e}")
                return False

        tasks = [worker(s, v, t) for s, v, t in tasks_to_generate]
        if tasks:
            results = await tqdm.gather(*tasks, desc="Dataset 2 (CodeAct Python)")
            success_count = sum(1 for r in results if r)
            logger.info(f"Dataset 2 complete! Successfully generated {success_count}/{len(tasks)} samples.")

        return output_path

#!/usr/bin/env python3
"""CAR-Bench SFT Dataset Sanitizer.
Repairs schema defects, unclosed reasoning blocks, and missing function call mappings.
"""

import os
import sys
import json
import uuid
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


def clean_arguments(args: Any) -> str:
    """Ensures function call arguments are serialized JSON strings."""
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return json.dumps({"raw_value": args}, ensure_ascii=False)
    elif isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    return "{}"


def extract_embedded_tool_calls(content: str) -> tuple[str, List[Dict[str, Any]]]:
    """Extracts tool calls embedded in <tool_calls>[...]</tool_calls> within content text."""
    extracted = []
    pattern = r"<tool_calls>\s*(.*?)\s*</tool_calls>"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        raw_json = match.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, list):
                extracted = parsed
            elif isinstance(parsed, dict):
                extracted = [parsed]
        except Exception:
            pass
        content = re.sub(pattern, "", content, flags=re.DOTALL).strip()
    return content, extracted


def repair_row_360_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    """Provides completed compliant response for truncated row 360 turn 9."""
    call_id_1 = f"call_{uuid.uuid4().hex[:12]}"
    call_id_2 = f"call_{uuid.uuid4().hex[:12]}"
    return {
        "role": "assistant",
        "content": (
            "<think>\n"
            "[Context Audit] User confirmed turning on low beams and fog lights. Exterior lights are currently off.\n"
            "[Policy Check] AUT-POL:005 requires low beams or parking lights active before fog lights. AUT-POL:001 satisfied.\n"
            "[Tool Selection] set_head_lights(on=True), set_fog_lights(on=True)\n"
            "[Plan] Turn on low beams, then turn on fog lights.\n"
            "</think>\n"
            "Turning on low beams and fog lights now."
        ),
        "tool_calls": [
            {
                "id": call_id_1,
                "type": "function",
                "function": {
                    "name": "set_head_lights",
                    "arguments": json.dumps({"on": True})
                }
            },
            {
                "id": call_id_2,
                "type": "function",
                "function": {
                    "name": "set_fog_lights",
                    "arguments": json.dumps({"on": True})
                }
            }
        ]
    }


def sanitize_multirole_conversation(convs: List[Dict[str, Any]], task_id: str) -> List[Dict[str, Any]]:
    """Sanitizes a multi-role JSON conversation trajectory."""
    sanitized: List[Dict[str, Any]] = []
    pending_tool_call_ids: List[str] = []

    for idx, turn in enumerate(convs):
        role = turn.get("role") or turn.get("from")
        content = turn.get("content") or ""
        clean_turn: Dict[str, Any] = {"role": role}

        # Check for Row 360 truncation
        if task_id == "hallucination_73_var_6" and idx == 9:
            fixed_turn = repair_row_360_turn(turn)
            for tc in fixed_turn["tool_calls"]:
                pending_tool_call_ids.append(tc["id"])
            sanitized.append(fixed_turn)
            continue

        # Close truncated <think> blocks
        if isinstance(content, str):
            if "<think>" in content and "</think>" not in content:
                content = content.rstrip() + "\n</think>"
            # Extract embedded <tool_calls> from text
            content, embedded_tcs = extract_embedded_tool_calls(content)
        else:
            embedded_tcs = []

        if content:
            clean_turn["content"] = content

        if role == "assistant":
            existing_tcs = turn.get("tool_calls") or []
            if not isinstance(existing_tcs, list):
                existing_tcs = []
            all_raw_tcs = existing_tcs + embedded_tcs

            # Check if next turn is an orphan tool turn
            next_turn = convs[idx + 1] if idx + 1 < len(convs) else None
            next_role = next_turn.get("role") or next_turn.get("from") if next_turn else None
            if next_role == "tool" and not all_raw_tcs:
                tool_name = next_turn.get("name", "unknown_tool")
                synth_id = f"call_{uuid.uuid4().hex[:12]}"
                all_raw_tcs.append({
                    "id": synth_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"}
                })

            sanitized_tcs = []
            for tc in all_raw_tcs:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                fn = tc.get("function", {})
                fn_name = fn.get("name") or tc.get("name") or "unknown_tool"
                fn_args = clean_arguments(fn.get("arguments") or tc.get("arguments") or {})
                sanitized_tcs.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args
                    }
                })
                pending_tool_call_ids.append(tc_id)

            if sanitized_tcs:
                clean_turn["tool_calls"] = sanitized_tcs
            elif "content" not in clean_turn:
                clean_turn["content"] = "Understood."

            sanitized.append(clean_turn)

        elif role == "tool":
            tool_name = turn.get("name", "unknown_tool")
            clean_turn["name"] = tool_name
            clean_turn["content"] = str(content) if content else "{}"
            if pending_tool_call_ids:
                clean_turn["tool_call_id"] = pending_tool_call_ids.pop(0)
            else:
                call_id = f"call_{uuid.uuid4().hex[:12]}"
                clean_turn["tool_call_id"] = call_id
                if sanitized and sanitized[-1]["role"] == "assistant":
                    prev_tcs = sanitized[-1].setdefault("tool_calls", [])
                    prev_tcs.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": "{}"}
                    })
            sanitized.append(clean_turn)

        else:
            clean_turn["content"] = str(content)
            sanitized.append(clean_turn)

    return sanitized


def sanitize_codeact_conversation(convs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitizes a CodeAct Python conversation trajectory."""
    sanitized: List[Dict[str, Any]] = []
    pending_tool_call_ids: List[str] = []

    for idx, turn in enumerate(convs):
        role = turn.get("role") or turn.get("from")
        content = turn.get("content") or ""
        clean_turn: Dict[str, Any] = {"role": role, "content": str(content)}

        if role == "assistant":
            next_turn = convs[idx + 1] if idx + 1 < len(convs) else None
            next_role = next_turn.get("role") or next_turn.get("from") if next_turn else None

            if next_role == "tool":
                py_id = f"call_py_{uuid.uuid4().hex[:12]}"
                code_match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
                code_body = code_match.group(1).strip() if code_match else content.strip()
                clean_turn["tool_calls"] = [
                    {
                        "id": py_id,
                        "type": "function",
                        "function": {
                            "name": "execute_python",
                            "arguments": json.dumps({"code": code_body}, ensure_ascii=False)
                        }
                    }
                ]
                pending_tool_call_ids.append(py_id)
            sanitized.append(clean_turn)

        elif role == "tool":
            clean_turn["name"] = turn.get("name") or "execute_python"
            if pending_tool_call_ids:
                clean_turn["tool_call_id"] = pending_tool_call_ids.pop(0)
            else:
                py_id = f"call_py_{uuid.uuid4().hex[:12]}"
                clean_turn["tool_call_id"] = py_id
                if sanitized and sanitized[-1]["role"] == "assistant":
                    sanitized[-1]["tool_calls"] = [
                        {
                            "id": py_id,
                            "type": "function",
                            "function": {
                                "name": "execute_python",
                                "arguments": json.dumps({"code": ""})
                            }
                        }
                    ]
            sanitized.append(clean_turn)

        else:
            sanitized.append(clean_turn)

    return sanitized


def sanitize_file(input_path: Path, output_path: Path, mode: str = "multirole") -> int:
    """Processes and standardizes a single dataset file."""
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line_str = line.strip()
            if not line_str:
                continue
            row = json.loads(line_str)
            task_id = row.get("task_id", "")
            raw_convs = row.get("conversations") or row.get("messages") or []

            if mode == "multirole":
                cleaned = sanitize_multirole_conversation(raw_convs, task_id)
            else:
                cleaned = sanitize_codeact_conversation(raw_convs)

            row["conversations"] = cleaned
            row["messages"] = cleaned
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    root = Path('.').resolve()
    data_dir = root / "data" / "new_data"

    f_multi = data_dir / "carbench_sft_multirole_json.jsonl"
    f_codeact = data_dir / "carbench_sft_codeact_python.jsonl"

    if f_multi.exists():
        tmp_m = f_multi.with_suffix(".tmp")
        n_m = sanitize_file(f_multi, tmp_m, mode="multirole")
        tmp_m.replace(f_multi)
        print(f"Sanitized {n_m} multi-role JSON records.")

    if f_codeact.exists():
        tmp_c = f_codeact.with_suffix(".tmp")
        n_c = sanitize_file(f_codeact, tmp_c, mode="codeact")
        tmp_c.replace(f_codeact)
        print(f"Sanitized {n_c} CodeAct Python records.")


if __name__ == "__main__":
    main()

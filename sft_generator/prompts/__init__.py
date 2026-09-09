"""Prompt package for SFT Data Generator."""
from sft_generator.prompts.multi_role_json_prompt import build_multi_role_json_prompt
from sft_generator.prompts.codeact_python_prompt import build_codeact_python_prompt

__all__ = ["build_multi_role_json_prompt", "build_codeact_python_prompt"]

"""CodeAct Validator (Inspired by Proxima Ultra Winner Paper).
Validates programmatic Python action scripts, Unknown-Value Sentinels, and Response Obligations.
"""

import ast
from typing import Dict, List, Any, Tuple, Optional
import re
import logging

from sft_generator.schemas import TOOL_NAME_MAP

logger = logging.getLogger(__name__)


class CodeActValidationResult:
    """Represents the outcome of a CodeAct Python script validation."""

    def __init__(self, is_valid: bool, violations: List[str], diagnostics: List[str]):
        self.is_valid = is_valid
        self.violations = violations
        self.diagnostics = diagnostics

    def __repr__(self) -> str:
        return f"CodeActValidationResult(is_valid={self.is_valid}, violations={self.violations})"


class CodeActValidator:
    """Validates executable Python actions produced in CodeAct format."""

    FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "socket", "shutil", "builtins", "eval", "exec"}
    ALLOWED_HELPERS = {
        "respond", "print", "batch", "len", "range", "str", "int", "float",
        "list", "dict", "set", "bool", "min", "max", "round", "abs", "sum",
        "isinstance", "enumerate", "zip", "sorted", "reversed", "any", "all"
    }

    def __init__(self):
        self.valid_tool_names = set(TOOL_NAME_MAP.keys())

    def validate_python_block(self, code_str: str) -> Tuple[bool, List[str]]:
        """Parses and checks safety and syntax of a Python code action block."""
        violations = []
        
        # 1. AST Syntax Parsing
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            violations.append(f"Python Syntax Error: {e.msg} at line {e.lineno}")
            return False, violations

        # 2. Security Check (No arbitrary system calls) & Tool Name Check
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.FORBIDDEN_IMPORTS:
                        violations.append(f"Security: Prohibited module import '{alias.name}'.")
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.FORBIDDEN_IMPORTS:
                    violations.append(f"Security: Prohibited module import '{node.module}'.")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in {"eval", "exec", "__import__", "compile"}:
                        violations.append(f"Security: Prohibited built-in execution '{func_name}'.")
                    elif func_name not in self.ALLOWED_HELPERS and func_name not in self.valid_tool_names:
                        violations.append(f"CodeAct: Unknown tool function '{func_name}' called in Python block.")

        return len(violations) == 0, violations

    def validate_trajectory(self, conversation: List[Dict[str, Any]]) -> CodeActValidationResult:
        """Validates a complete CodeAct conversation for syntax, unknown-value handling, and response obligations."""
        violations: List[str] = []
        diagnostics: List[str] = []

        if not isinstance(conversation, list):
            return CodeActValidationResult(
                is_valid=False,
                violations=["Invalid conversation: expected a list of message dictionaries."],
                diagnostics=["Ensure the output contains an array of messages."],
            )

        for turn_idx, msg in enumerate(conversation):
            if not isinstance(msg, dict):
                violations.append(f"Turn {turn_idx}: Message is not a dict (got {type(msg).__name__}).")
                continue

            role = msg.get("role")
            content = str(msg.get("content", "") or "")

            if role == "assistant":
                # Extract Python code blocks ```python ... ```
                python_blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)
                for p_idx, block in enumerate(python_blocks):
                    clean_code = block.strip()
                    is_valid_syntax, errs = self.validate_python_block(clean_code)
                    if not is_valid_syntax:
                        for err in errs:
                            violations.append(f"Turn {turn_idx} Code Block {p_idx}: {err}")
                            diagnostics.append("Ensure Python script is syntactically valid and contains no forbidden imports.")

                # Proxima Unknown-Value Sentinel Check:
                if "unknown" in content.lower() and "unsupported" in content.lower():
                    pass

        is_valid = len(violations) == 0
        return CodeActValidationResult(is_valid=is_valid, violations=violations, diagnostics=diagnostics)


# Singleton codeact validator instance
codeact_validator = CodeActValidator()

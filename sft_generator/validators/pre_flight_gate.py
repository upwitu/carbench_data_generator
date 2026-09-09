"""Deterministic L3 Pre-Flight Gate Validator (Inspired by 10cars Winner Paper).
Strictly validates policy adherence before any tool call execution or dataset persistence.
"""

import json
import re
from typing import Dict, List, Set, Any, Optional
from sft_generator.schemas import (
    ACTUATOR_TO_READ_MAPPING,
    REQUIRES_CONFIRMATION_TOOLS,
    WAYPOINT_EDIT_TOOLS,
    TOOL_NAME_MAP,
)


class PreFlightGateResult:
    """Represents the outcome of an L3 Pre-Flight Gate check."""

    def __init__(self, is_valid: bool, violations: List[str], diagnostics: List[str]):
        self.is_valid = is_valid
        self.violations = violations
        self.diagnostics = diagnostics

    def __repr__(self) -> str:
        return f"PreFlightGateResult(is_valid={self.is_valid}, violations={self.violations})"


class PreFlightGateValidator:
    """Deterministic Pre-Flight Gate enforcing CAR-bench invariants:
    - Read-Before-Write (AUT-POL:001)
    - Confirmation Gate (AUT-POL:002)
    - Single Waypoint Mutation per Turn (AUT-POL:006)
    - Parameter Limits & Idempotence (AUT-POL:007, AUT-POL:008, AUT-POL:016)
    """

    def __init__(self):
        self.valid_tool_names: Set[str] = set(TOOL_NAME_MAP.keys())

    def validate_trajectory(self, conversation: List[Dict[str, Any]]) -> PreFlightGateResult:
        """Validates an entire multi-turn conversation trajectory.
        
        Args:
            conversation: List of message dictionaries with 'role', 'content', and optional 'tool_calls'.
            
        Returns:
            PreFlightGateResult containing validity boolean and diagnostic violation messages.
        """
        violations: List[str] = []
        diagnostics: List[str] = []
        
        if not isinstance(conversation, list):
            return PreFlightGateResult(
                is_valid=False,
                violations=["Invalid conversation: expected a list of message dictionaries."],
                diagnostics=["Ensure the JSON output contains a valid array in 'conversations'."],
            )

        executed_reads: Set[str] = set()
        executed_setters: Set[str] = set()
        pending_confirmation_tool: Optional[str] = None
        user_confirmed: bool = False
        
        for turn_idx, msg in enumerate(conversation):
            if not isinstance(msg, dict):
                violations.append(f"Turn {turn_idx}: Message is not a dict (got {type(msg).__name__}).")
                diagnostics.append("Each message in 'conversations' must be an object with 'role' and 'content'.")
                continue

            role = msg.get("role")
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls", [])
            
            # 1. Track User Confirmations
            if role == "user":
                lower_content = str(content).lower().strip()
                if any(w in lower_content for w in ["yes", "confirm", "proceed", "sure", "go ahead", "được", "đồng ý", "xác nhận"]):
                    user_confirmed = True
                else:
                    user_confirmed = False
                    
            # 2. Check Assistant Tool Calls
            elif role == "assistant" and tool_calls and isinstance(tool_calls, list):
                waypoint_edits_this_turn = 0
                
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        violations.append(f"Turn {turn_idx}: Tool call is not a dict.")
                        continue

                    func_data = tool_call.get("function", {})
                    if not isinstance(func_data, dict):
                        violations.append(f"Turn {turn_idx}: Tool call 'function' is not a dict.")
                        continue

                    tool_name = func_data.get("name", "")
                    raw_args = func_data.get("arguments", {})
                    
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except Exception:
                            violations.append(f"Turn {turn_idx}: Tool '{tool_name}' has malformed JSON arguments.")
                            diagnostics.append("Ensure tool call arguments are strictly valid JSON.")
                            continue
                    elif isinstance(raw_args, dict):
                        args = raw_args
                    else:
                        args = {}
                        
                    # Check 1: Tool existence in 57 tools (AUT-POL:015)
                    if tool_name not in self.valid_tool_names:
                        violations.append(f"Turn {turn_idx}: Unknown tool '{tool_name}' not in 57 car tools (Hallucination).")
                        diagnostics.append(f"Tool '{tool_name}' does not exist on this vehicle.")
                        continue
                        
                    # Check 2: Single waypoint mutation per turn (AUT-POL:006)
                    if tool_name in WAYPOINT_EDIT_TOOLS:
                        waypoint_edits_this_turn += 1
                        if waypoint_edits_this_turn > 1:
                            violations.append(f"Turn {turn_idx}: Batched multiple waypoint edits in one turn (AUT-POL:006).")
                            diagnostics.append("Only 1 waypoint modification is permitted per turn.")
                            
                    # Check 3: Read-Before-Write (AUT-POL:001)
                    if tool_name in ACTUATOR_TO_READ_MAPPING:
                        mandatory_read = ACTUATOR_TO_READ_MAPPING[tool_name]
                        if mandatory_read not in executed_reads:
                            violations.append(f"Turn {turn_idx}: Actuator '{tool_name}' called before reading state with '{mandatory_read}' (AUT-POL:001).")
                            diagnostics.append(f"You must call '{mandatory_read}' prior to modifying '{tool_name}'.")
                            
                    # Check 4: Confirmation Gate for Dangerous Tools (AUT-POL:002)
                    if tool_name in REQUIRES_CONFIRMATION_TOOLS:
                        if not user_confirmed and pending_confirmation_tool != tool_name:
                            violations.append(f"Turn {turn_idx}: Tool '{tool_name}' executed without explicit user confirmation (AUT-POL:002).")
                            diagnostics.append(f"Action '{tool_name}' requires prior explicit user confirmation.")
                            
                    # Check 4b: 2-Step ID Provenance (AUT-POL:018)
                    if tool_name == "get_contact_information":
                        c_ids = args.get("contact_ids", [])
                        if isinstance(c_ids, list):
                            for cid in c_ids:
                                if not str(cid).startswith("con_"):
                                    violations.append(f"Turn {turn_idx}: Invalid contact_id '{cid}' passed to get_contact_information (must start with 'con_'). Use get_contact_id_by_contact_name first (AUT-POL:018).")
                                    diagnostics.append("You must resolve contact names to contact_id (con_xxxx) before fetching information.")
                    elif tool_name == "get_routes":
                        dest = args.get("destination_id") or args.get("destination") or args.get("end_location")
                        if dest and not str(dest).startswith("loc_"):
                            violations.append(f"Turn {turn_idx}: Invalid destination location '{dest}' in get_routes (must start with 'loc_'). Use get_location_id_by_location_name first (AUT-POL:018).")
                            diagnostics.append("You must resolve location names to location_id (loc_xxxx) before fetching routes.")

                    # Check 5: Temperature boundary check (AUT-POL:007)
                    if tool_name == "set_temperature":
                        temp = args.get("temperature")
                        if temp is not None and isinstance(temp, (int, float)) and (temp < 16.0 or temp > 28.0):
                            violations.append(f"Turn {turn_idx}: AC temperature {temp}C out of bounds [16.0, 28.0] (AUT-POL:007).")
                            diagnostics.append("AC temperature must be between 16.0 and 28.0 Celsius.")
                            
                    # Check 6: Fan speed level check (AUT-POL:008)
                    if tool_name == "set_fan_speed":
                        speed = args.get("fan_speed") or args.get("speed")
                        if speed is not None and isinstance(speed, int) and (speed < 0 or speed > 7):
                            violations.append(f"Turn {turn_idx}: Fan speed {speed} out of bounds [0, 7] (AUT-POL:008).")
                            diagnostics.append("Fan speed level must be an integer between 0 and 7.")
                            
                    # Check 7: Idempotence check (AUT-POL:016)
                    try:
                        arg_sig = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                    except Exception:
                        arg_sig = f"{tool_name}:{str(args)}"

                    if arg_sig in executed_setters:
                        violations.append(f"Turn {turn_idx}: Redundant duplicate setter call '{arg_sig}' (AUT-POL:016).")
                        diagnostics.append("Do not issue identical consecutive setter calls.")
                    executed_setters.add(arg_sig)

                    # Check 8: Sunroof & Sunshade Coupling (AUT-POL:005)
                    if tool_name == "open_close_sunroof":
                        roof_pct = args.get("percentage", 0)
                        if roof_pct is not None and isinstance(roof_pct, (int, float)) and roof_pct > 0:
                            turn_tools = [tc.get("function", {}).get("name") for tc in tool_calls if isinstance(tc, dict)]
                            if "open_close_sunshade" not in turn_tools and "open_close_sunshade" not in executed_setters and "get_sunroof_and_sunshade_position" not in executed_reads:
                                violations.append(f"Turn {turn_idx}: Sunroof opened without sunshade opened or parallel open_close_sunshade call (AUT-POL:005).")
                                diagnostics.append("Sunroof can only be opened if sunshade is open or opened in parallel.")

                    # Check 9: Weather check before sunroof or fog lights (AUT-POL:009)
                    if tool_name == "open_close_sunroof":
                        roof_pct = args.get("percentage", 0)
                        if roof_pct is not None and isinstance(roof_pct, (int, float)) and roof_pct > 0:
                            turn_tools = [tc.get("function", {}).get("name") for tc in tool_calls if isinstance(tc, dict)]
                            if "get_weather" not in executed_reads and "get_weather" not in turn_tools:
                                violations.append(f"Turn {turn_idx}: Sunroof opened without checking weather via get_weather (AUT-POL:009).")
                                diagnostics.append("Weather condition must be checked before opening the sunroof.")
                    elif tool_name == "set_fog_lights":
                        if args.get("on") is True:
                            turn_tools = [tc.get("function", {}).get("name") for tc in tool_calls if isinstance(tc, dict)]
                            if "get_weather" not in executed_reads and "get_weather" not in turn_tools:
                                violations.append(f"Turn {turn_idx}: Fog lights turned on without checking weather via get_weather (AUT-POL:009).")
                                diagnostics.append("Weather condition must be checked before activating fog lights.")

                    # Check 10: Front window defrost climate coupling (AUT-POL:010)
                    if tool_name == "set_window_defrost":
                        if args.get("on") is True and args.get("defrost_window") in ["FRONT", "BOTH", "ALL"]:
                            turn_tools = [tc.get("function", {}).get("name") for tc in tool_calls if isinstance(tc, dict)]
                            climate_checked = "get_climate_settings" in executed_reads or (
                                "set_fan_speed" in turn_tools and "set_air_conditioning" in turn_tools
                            )
                            if not climate_checked:
                                violations.append(f"Turn {turn_idx}: Front defrost activated without prior climate check (AUT-POL:010).")
                                diagnostics.append("Climate settings must be checked before activating front window defrost.")

                    # Check 11: AC Activation Coupling (AUT-POL:011)
                    if tool_name == "set_air_conditioning" and args.get("on") is True:
                        turn_tools = [tc.get("function", {}).get("name") for tc in tool_calls if isinstance(tc, dict)]
                        state_checked = (
                            "get_climate_settings" in executed_reads
                            or "get_window_positions" in executed_reads
                            or "open_close_windows" in turn_tools
                        )
                        if not state_checked:
                            violations.append(f"Turn {turn_idx}: Air conditioning turned on without checking window positions or climate settings (AUT-POL:011).")
                            diagnostics.append("Window positions and climate settings must be verified before turning on AC.")

                    # Check 12: Fog lights & High beams coupling (AUT-POL:013 & AUT-POL:014)
                    if tool_name == "set_fog_lights" and args.get("on") is True:
                        turn_tools = [tc.get("function", {}).get("name") for tc in tool_calls if isinstance(tc, dict)]
                        lights_checked = "get_exterior_lights_status" in executed_reads or "set_head_lights" in turn_tools
                        if not lights_checked:
                            violations.append(f"Turn {turn_idx}: Fog lights activated without checking exterior lights status (AUT-POL:013).")
                            diagnostics.append("Exterior lights status must be checked before activating fog lights.")
                    elif tool_name == "set_head_lights_high_beams" and args.get("on") is True:
                        if any("set_fog_lights" in s for s in executed_setters):
                            violations.append(f"Turn {turn_idx}: High beams activated while fog lights are active (AUT-POL:014).")
                            diagnostics.append("High beams cannot be activated if fog lights are on.")

            # Check 13: Speakable natural voice text without visual markdown formatting (LLM-POL:002)
            elif role == "assistant" and content and not tool_calls:
                text_str = str(content)
                if re.search(r"^#{1,6}\s+", text_str, re.MULTILINE):
                    violations.append(f"Turn {turn_idx}: Markdown headers found in voice response (LLM-POL:002).")
                    diagnostics.append("Car assistant text must be natural speakable voice without markdown headers.")
                elif re.search(r"^\s*[-*]\s+", text_str, re.MULTILINE):
                    violations.append(f"Turn {turn_idx}: Markdown bullet list found in voice response (LLM-POL:002).")
                    diagnostics.append("Responses are forwarded to TTS; do not use formatted bullet lists.")

            # 3. Track Tool Results
            elif role == "tool":
                tool_name = msg.get("name", "")
                if tool_name.startswith("get_"):
                    executed_reads.add(tool_name)
                    
        is_valid = len(violations) == 0
        return PreFlightGateResult(is_valid=is_valid, violations=violations, diagnostics=diagnostics)


# Singleton gate validator instance
pre_flight_gate = PreFlightGateValidator()

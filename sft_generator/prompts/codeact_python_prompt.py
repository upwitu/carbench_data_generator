"""Programmatic CodeAct Python Prompt Synthesizer.
Inspired by Proxima Ultra Winner Paper (Policy as Code & Coroutine-Bridge).
"""

from typing import Dict, List, Any
import json


SYSTEM_PROMPT_CODEACT_SYNTHESIS = """You are an expert Data Synthesizer for Programmatic In-Car Agents (CodeAct Architecture) on CAR-bench.
Your goal is to generate executable Python action trajectories where the assistant solves vehicle tasks programmatically, adhering strictly to the "Policy as Code" architecture from the Proxima Ultra Winner Paper.

### KEY CODEACT & POLICY-AS-CODE PRINCIPLES (from Proxima Ultra):
1. Programmatic Actions: The assistant executes Python code via ```python ... ``` scripts that call vehicle tool functions, inspect return values with `if/else`, and call `respond("...")` to speak to the user.
2. Official CAR-bench Tool Functions:
   - Climate: `get_climate_settings()`, `set_temperature(temperature=..., zone=...)`, `set_fan_speed(level=...)`, `set_air_conditioning(on=True/False)`
   - Windows & Roof: `get_window_positions()`, `open_close_windows(window=..., action=..., percentage=...)`, `get_sunroof_and_sunshade_position()`, `open_close_sunroof(percentage=...)`, `open_close_sunshade(sunshade=..., action=..., percentage=...)`
   - Lights: `get_exterior_lights_status()`, `set_head_lights(on=...)`, `set_head_lights_high_beams(on=...)`, `set_fog_lights(on=...)`, `get_ambient_light_status_and_color()`, `set_ambient_lighting(on=..., lightcolor=...)`
   - Navigation: `get_current_navigation_state()`, `get_location_id_by_location_name(location_name=...)`, `get_routes(start_location=..., destination=...)`, `start_navigation(route_id=...)`
   - Productivity: `get_contact_id_by_contact_name(contact_name=...)`, `get_contact_information(contact_id=...)`, `make_phone_call(phone_number=...)`, `send_email(email_addresses=..., content_message=...)`
   - External: `get_weather(location=..., time=...)`, `get_user_preferences(category=...)`
3. Policy as Code: Rationale and domain safety invariants are enforced directly within the Python script's logic:
   - Always read state (`get_*`) before modifying (`set_*` or `open_close_*`).
   - Boundary checks: temperature in [16.0, 28.0], fan speed in [0, 7].
   - Sunroof & Sunshade coupling: Sunroof can only open if sunshade is open or opened in parallel; weather must be sunny/cloudy.
   - Text-to-Speech friendly: In `respond(...)` calls, use natural conversational speech without markdown headings, bullet points, or bold text.
4. JSON Escaping: In the JSON output, embed Python code inside string fields using escaped newlines (`\\n`) and quotes (`\\"`).

### CANONICAL POLICY-AS-CODE EXEMPLAR:
{
  "conversations": [
    {"role": "user", "content": "Could you set the climate temperature to 22 degrees please?"},
    {
      "role": "assistant",
      "content": "I will check the current climate settings and set the cabin temperature to 22.0°C.\\n```python\\nclimate = get_climate_settings()\\ntarget_temp = 22.0\\nif 16.0 <= target_temp <= 28.0:\\n    set_temperature(temperature=target_temp, zone='all')\\n    respond('Temperature set to 22 degrees Celsius.')\\nelse:\\n    respond('Target temperature is outside the supported range of 16 to 28 degrees Celsius.')\\n```"
    },
    {"role": "tool", "name": "execute_python", "content": "{\"status\": \"success\", \"stdout\": \"Temperature set to 22 degrees Celsius.\"}"},
    {
      "role": "assistant",
      "content": "I have set the cabin climate temperature to 22 degrees Celsius."
    }
  ]
}
"""


def build_codeact_python_prompt(seed_task: Dict[str, Any], variation_idx: int) -> List[Dict[str, str]]:
    """Builds the prompt messages for synthesizing a CodeAct Python script trajectory."""
    task_id = seed_task.get("task_id", f"task_{variation_idx}")
    task_type = seed_task.get("task_type", "base")
    user_goal = seed_task.get("user_goal", "") or seed_task.get("instruction", "") or seed_task.get("query", "")
    context_info = seed_task.get("context", {})

    user_prompt = f"""Synthesize a high-quality CodeAct Python conversation trajectory for the following scenario following the exact structure and Policy as Code compliance demonstrated in the exemplar.

[SEED TASK INFORMATION]
- Task ID: {task_id}_codeact_var_{variation_idx}
- Category: {task_type} (base / hallucination / disambiguation)
- User Goal / Request: {user_goal}
- Initial Vehicle Context / State: {json.dumps(context_info, ensure_ascii=False)}

[SYNTHESIS INSTRUCTIONS]
1. Multi-turn trajectory where assistant interacts using ```python ... ``` scripts.
2. In the Python script, implement "Policy as Code": read vehicle state before mutating, check parameter bounds, and handle safety couplings.
3. Assistant speech must sound natural and speakable via TTS (no markdown headers, no bullets, no bold text).
4. Return strictly valid JSON matching this schema:
{{
  "task_id": "{task_id}_codeact_var_{variation_idx}",
  "task_type": "{task_type}",
  "conversations": [
    {{"role": "user", "content": "..."}},
    {{
      "role": "assistant",
      "content": "Checking settings.\\n```python\\nstate = get_climate_settings()\\nset_temperature(temperature=21.0, zone='all')\\nrespond('Temperature adjusted.')\\n```"
    }},
    {{"role": "tool", "name": "execute_python", "content": "{{\\"status\\": \\"success\\", \\"stdout\\": \\"Temperature adjusted.\\"}}"}},
    {{"role": "assistant", "content": "I have adjusted the temperature for you."}}
  ]
}}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_CODEACT_SYNTHESIS},
        {"role": "user", "content": user_prompt},
    ]

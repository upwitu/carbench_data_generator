"""CAR-bench Official 58 Tool Schemas, Policy Definitions, and Actuator Mappings.
Compiled from official CAR-bench benchmark environment and Winner Papers (10cars, FreudeDrive, Proxima).
"""

from typing import Dict, List, Set, Any

# ============================================================================
# 1. POLICIES & CONSTRAINTS
# ============================================================================

# High-impact dangerous actions requiring explicit user confirmation before actuation
REQUIRES_CONFIRMATION_TOOLS: Set[str] = {
    "open_close_trunk_door",
    "send_email",
    "set_head_lights_high_beams",
    "navigation_delete_final_destination",
    "navigation_delete_one_waypoint",
    "stop_navigation",
}

# 10cars L1/L3: Actuator to Mandatory Read mapping (Read-Before-Write)
ACTUATOR_TO_READ_MAPPING: Dict[str, str] = {
    "set_temperature": "get_climate_settings",
    "set_fan_speed": "get_climate_settings",
    "set_fan_airflow_direction": "get_climate_settings",
    "set_air_circulation": "get_climate_settings",
    "set_air_conditioning": "get_climate_settings",
    "set_window_defrost": "get_climate_settings",
    "open_close_windows": "get_window_positions",
    "open_close_sunroof": "get_sunroof_and_sunshade_position",
    "open_close_sunshade": "get_sunroof_and_sunshade_position",
    "set_seat_heating": "get_seat_heating_level",
    "set_steering_wheel_heating": "get_steering_wheel_heating_level",
    "set_ambient_lighting": "get_ambient_light_status_and_color",
    "set_head_lights": "get_exterior_lights_status",
    "set_head_lights_high_beams": "get_exterior_lights_status",
    "set_fog_lights": "get_exterior_lights_status",
    "set_reading_light": "get_reading_lights_status",
    "open_close_trunk_door": "get_trunk_door_position",
    "start_navigation": "get_routes",
    "set_destination": "get_routes",
}

# State changing tools (Actuators and Communications)
STATE_CHANGING_TOOLS: Set[str] = set(ACTUATOR_TO_READ_MAPPING.keys()).union({
    "navigation_add_one_waypoint",
    "navigation_delete_one_waypoint",
    "navigation_replace_one_waypoint",
    "navigation_replace_final_destination",
    "navigation_delete_final_destination",
    "send_message",
    "make_phone_call",
    "send_email",
})

# Waypoint tools that cannot be batched in 1 turn (10cars L1)
WAYPOINT_EDIT_TOOLS: Set[str] = {
    "navigation_add_one_waypoint",
    "navigation_delete_one_waypoint",
    "navigation_replace_one_waypoint",
    "navigation_replace_final_destination",
    "navigation_delete_final_destination",
    "set_destination",
}

# Official CAR-bench Policies (from car_voice_assistant/wiki.md & policy_evaluator.py)
OFFICIAL_CAR_BENCH_POLICIES: Dict[str, str] = {
    "LLM-POL:002": "Metric system & 24h datetime: unit of distance km/m, temperature Celsius, datetime 24h format. Speakable text only (no markdown, lists, bold).",
    "LLM-POL:004": "Safety Confirmation: Tools with REQUIRES_CONFIRMATION (open_close_trunk_door, send_email, set_head_lights_high_beams, navigation_delete_final_destination, navigation_delete_one_waypoint, stop_navigation) require explicit user confirmation before execution.",
    "AUT-POL:005": "Sunroof/Sunshade Coupling: Sunroof can only be opened if sunshade is already fully opened or opened in parallel.",
    "LLM-POL:007": "AC Efficiency: Opening windows >25% while AC is active requires confirmation and energy inefficiency warning.",
    "LLM-POL:008": "Weather Confirmation: In adverse weather, vehicle control requires explicit confirmation (sunroof opening requires sunny/cloudy; fog lights require storm/hail).",
    "AUT-POL:009": "Weather Check: Weather must be checked manually via get_weather before opening sunroof or setting fog lights.",
    "AUT-POL:010": "Window Defrost Coupling: Window defrost for front/all windows automatically sets fan speed >= 2, airflow to WINDSHIELD, and turns AC ON.",
    "AUT-POL:011": "AC Activation Coupling: Setting AC to ON automatically closes windows open > 20% and sets fan speed >= 1.",
    "LLM-POL:012": "Single Seat Temperature Disparity: If single seat zone set creates > 3°C difference from others, inform user.",
    "AUT-POL:013": "Fog Lights Coupling: Fog lights require low beam headlights ON and high beam headlights OFF.",
    "AUT-POL:014": "High Beams Restriction: High beams cannot be activated if fog lights are on.",
    "AUT-POL:016": "Route Origin: Start of route always has to be the current car location.",
    "AUT-POL:017": "Active Navigation Waypoint Modification: Tools to delete/replace/add waypoint or destination only usable when navigation is active.",
    "AUT-POL:018": "Single Waypoint Mutation: Waypoint modifications must be sequential, at most 1 per turn (never batched in parallel).",
    "AUT-POL:019": "Route Minimum Endpoints: Route must consist of at least start and destination; destination cannot be deleted if no intermediate stop.",
    "LLM-POL:021": "Toll Roads Disclosure: Detail route presentation with toll roads requires upfront notice to user.",
    "LLM-POL:022": "Multi-stop Route Default: Multi-stop route defaults to fastest route per segment; inform user and ask if alternatives needed.",
    "AUT-POL:023": "Calendar Current Day: Calendar entries can only be requested for current day.",
    "AUT-POL:024": "Weather Current Day: Weather can only be requested for current day.",
}

# Legacy custom mapping preserved for backwards compatibility
CAR_BENCH_POLICIES: Dict[str, str] = {
    "AUT-POL:001": "Read-before-write: Prior vehicle state must be read before mutating actuator state.",
    "AUT-POL:002": "Confirmation gate: High-impact actions (trunk, high beams, email sending, route deletion) require explicit confirmation before execution.",
    "AUT-POL:003": "Sunroof safety: Sunroof cannot be opened if sunshade is fully closed, or during heavy precipitation (AUT-POL:005, AUT-POL:009).",
    "AUT-POL:004": "AC-Window conflict: If window is opened >25% while AC is active, emit an energy efficiency disclosure (LLM-POL:007).",
    "AUT-POL:005": "Fog lights coupling: Front fog lights require low-beam ON and high-beam OFF (AUT-POL:013).",
    "AUT-POL:006": "Single waypoint mutation: Only 1 waypoint modification permitted per assistant turn (AUT-POL:018).",
    "AUT-POL:007": "Temperature limits: Cabin temperature setpoint must remain bounded within [16.0, 28.0] Celsius.",
    "AUT-POL:008": "Fan speed limits: Fan speed integer levels bounded within [0, 7].",
    "AUT-POL:009": "Precipitation safety: Weather check mandatory before sunroof/foglights (AUT-POL:009).",
    "AUT-POL:010": "Disambiguation ladder: Resolve ambiguity via Policy -> Explicit User Words -> Preferences -> Policy Defaults -> Vehicle State -> Ask User (as final rung).",
    "AUT-POL:011": "Multi-zone climate balance: Cross-zone temperature disparity exceeding 3.0 Celsius requires user notification (LLM-POL:012).",
    "AUT-POL:012": "Seat occupancy check: Seat heating/ventilation activation requires seat occupancy confirmation.",
    "AUT-POL:013": "Navigation toll disclosure: Routing through tollways mandates upfront toll notice to user (LLM-POL:021).",
    "AUT-POL:014": "Child lock safety: Rear child locks cannot be toggled while vehicle velocity > 0 km/h.",
    "AUT-POL:015": "Unknown value handling: Missing or 'unknown' tool return values must be acknowledged without fabrication.",
    "AUT-POL:016": "Idempotent mutation: Avoid redundant consecutive setter calls with identical parameters.",
    "AUT-POL:017": "No interim corrective writes: Intermediate invalid states cause immediate evaluation failure.",
    "AUT-POL:018": "Contact identifier provenance: Phone calls and messages must use verified contacts from search_contacts/get_contact_id_by_contact_name.",
    "AUT-POL:019": "Action execution after read: Immediate action tool execution upon querying status, no empty promises.",
}

# ============================================================================
# 2. OFFICIAL 58 CAR-BENCH TOOL DEFINITIONS (OpenAI Tool Format)
# ============================================================================

ALL_CAR_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate_charging_soc_by_time",
            "description": "Calculate projected SOC after charging duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "charging_time_minutes": {
                        "type": "integer"
                    }
                },
                "required": [
                    "charging_time_minutes"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_charging_time_by_soc",
            "description": "Calculate required charging time to reach target SOC.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_soc": {
                        "type": "number"
                    }
                },
                "required": [
                    "target_soc"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_date_time",
            "description": "Perform date and time computations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string"
                    }
                },
                "required": [
                    "expression"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Execute mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string"
                    }
                },
                "required": [
                    "expression"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_route_distance_into_time",
            "description": "Convert driving distance into estimated travel time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance_km": {
                        "type": "number"
                    }
                },
                "required": [
                    "distance_km"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ambient_light_status_and_color",
            "description": "Query ambient light color and status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_car_color",
            "description": "Query exterior car color.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_charging_status",
            "description": "Query EV battery state of charge (SOC) and max charging power.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_climate_settings",
            "description": "Query climate control settings.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_contact_id_by_contact_name",
            "description": "Search contact ID by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {
                        "type": "string"
                    }
                },
                "required": [
                    "contact_name"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_contact_information",
            "description": "Query phone/email contact information by contact ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_id": {
                        "type": "string"
                    }
                },
                "required": [
                    "contact_id"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_navigation_state",
            "description": "Query current navigation state.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_distance_by_soc",
            "description": "Calculate remaining driving range given current SOC.",
            "parameters": {
                "type": "object",
                "properties": {
                    "soc_percentage": {
                        "type": "number"
                    }
                },
                "required": [
                    "soc_percentage"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_entries_from_calendar",
            "description": "Query calendar entries for current day.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_exterior_lights_status",
            "description": "Query exterior lights status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fuel_information",
            "description": "Query fuel level or battery state of charge.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_location_id_by_location_name",
            "description": "Resolve location name to unique location ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string"
                    }
                },
                "required": [
                    "location_name"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_reading_lights_status",
            "description": "Query reading lights status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_routes",
            "description": "Get routes between start location and destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_location": {
                        "type": "string"
                    },
                    "destination": {
                        "type": "string"
                    }
                },
                "required": [
                    "start_location",
                    "destination"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_seat_heating_level",
            "description": "Query seat heating levels.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_seats_occupancy",
            "description": "Query cabin seat occupancy.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_steering_wheel_heating_level",
            "description": "Query steering wheel heating level.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sunroof_and_sunshade_position",
            "description": "Query sunroof and sunshade opening positions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_inside_car",
            "description": "Query internal cabin temperature.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_trunk_door_position",
            "description": "Query trunk door position.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_preferences",
            "description": "Retrieve stored user preferences (POI, climate, routing, settings).",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Query weather conditions for a location and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string"
                    },
                    "time": {
                        "type": "string"
                    }
                },
                "required": [
                    "location"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_window_positions",
            "description": "Query opening percentage of all power windows.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "make_phone_call",
            "description": "Initiate a phone call to contact or number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string"
                    }
                },
                "required": [
                    "phone_number"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigation_add_one_waypoint",
            "description": "Add one waypoint to active navigation route.",
            "parameters": {
                "type": "object",
                "properties": {
                    "waypoint_location": {
                        "type": "string"
                    }
                },
                "required": [
                    "waypoint_location"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigation_delete_final_destination",
            "description": "Delete final destination from active navigation route.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigation_delete_one_waypoint",
            "description": "Delete a waypoint from active navigation route.",
            "parameters": {
                "type": "object",
                "properties": {
                    "waypoint_location": {
                        "type": "string"
                    }
                },
                "required": [
                    "waypoint_location"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigation_replace_final_destination",
            "description": "Replace final destination in navigation route.",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_destination": {
                        "type": "string"
                    }
                },
                "required": [
                    "new_destination"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigation_replace_one_waypoint",
            "description": "Replace an existing waypoint in navigation route.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_waypoint": {
                        "type": "string"
                    },
                    "new_waypoint": {
                        "type": "string"
                    }
                },
                "required": [
                    "old_waypoint",
                    "new_waypoint"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_close_sunroof",
            "description": "Open or close the sunroof to a specified percentage (0 to 100).",
            "parameters": {
                "type": "object",
                "properties": {
                    "percentage": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Opening percentage (0 = closed, 100 = fully open)"
                    }
                },
                "required": [
                    "percentage"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_close_sunshade",
            "description": "Open or close the sunshade to a specified percentage (0 to 100).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "open",
                            "close"
                        ]
                    },
                    "sunshade": {
                        "type": "string",
                        "enum": [
                            "front",
                            "rear",
                            "all"
                        ]
                    },
                    "percentage": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": [
                    "action",
                    "sunshade",
                    "percentage"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_close_trunk_door",
            "description": "Open or close the vehicle trunk door. REQUIRES USER CONFIRMATION.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "open",
                            "close"
                        ]
                    }
                },
                "required": [
                    "action"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_close_windows",
            "description": "Open or close specific power windows (0 to 100%).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "open",
                            "close"
                        ]
                    },
                    "window": {
                        "type": "string",
                        "enum": [
                            "front_left",
                            "front_right",
                            "rear_left",
                            "rear_right",
                            "all"
                        ]
                    },
                    "percentage": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": [
                    "action",
                    "window"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_poi_along_the_route",
            "description": "Search points of interest along an active navigation route.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_poi": {
                        "type": "string"
                    },
                    "at_kilometer": {
                        "type": "number"
                    }
                },
                "required": [
                    "category_poi"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_poi_at_location",
            "description": "Search points of interest near a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string"
                    },
                    "category": {
                        "type": "string"
                    }
                },
                "required": [
                    "location"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email message. REQUIRES USER CONFIRMATION.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_addresses": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "content_message": {
                        "type": "string"
                    }
                },
                "required": [
                    "email_addresses",
                    "content_message"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_air_circulation",
            "description": "Set air circulation mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": [
                            "AUTO",
                            "FRESH_AIR",
                            "RECIRCULATION"
                        ]
                    }
                },
                "required": [
                    "mode"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_air_conditioning",
            "description": "Turn air conditioning ON or OFF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "on"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_ambient_lighting",
            "description": "Set interior ambient lighting color and status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {
                        "type": "boolean"
                    },
                    "lightcolor": {
                        "type": "string",
                        "enum": [
                            "BLUE",
                            "RED",
                            "PURPLE",
                            "WHITE",
                            "GREEN",
                            "AMBER",
                            "CYAN",
                            "OFF"
                        ]
                    }
                },
                "required": [
                    "on",
                    "lightcolor"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_destination",
            "description": "Set navigation destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string"
                    }
                },
                "required": [
                    "destination"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_fan_airflow_direction",
            "description": "Set fan airflow direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": [
                            "HEAD",
                            "FEET",
                            "HEAD_FEET",
                            "WINDSHIELD",
                            "WINDSHIELD_FEET"
                        ]
                    }
                },
                "required": [
                    "direction"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_fan_speed",
            "description": "Set climate fan speed level (0 to 5).",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 5
                    }
                },
                "required": [
                    "level"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_fog_lights",
            "description": "Turn front fog lights ON or OFF. Weather dependency apply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "on"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_head_lights",
            "description": "Turn low beam headlights ON or OFF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "on"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_head_lights_high_beams",
            "description": "Turn high beam headlights ON or OFF. REQUIRES USER CONFIRMATION.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "on"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_reading_light",
            "description": "Turn reading light ON or OFF for a specific seat zone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {
                        "type": "boolean"
                    },
                    "position": {
                        "type": "string",
                        "enum": [
                            "DRIVER",
                            "PASSENGER",
                            "DRIVER_REAR",
                            "PASSENGER_REAR"
                        ]
                    }
                },
                "required": [
                    "on",
                    "position"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_seat_heating",
            "description": "Set seat heating level (0 to 3) for a seat zone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 3
                    },
                    "seat": {
                        "type": "string",
                        "enum": [
                            "driver",
                            "passenger",
                            "rear_left",
                            "rear_right"
                        ]
                    }
                },
                "required": [
                    "level",
                    "seat"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_steering_wheel_heating",
            "description": "Set steering wheel heating level (0 to 3).",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 3
                    }
                },
                "required": [
                    "level"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_temperature",
            "description": "Set target climate temperature for seat zones (16\u00b0C to 28\u00b0C).",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "number",
                        "minimum": 16.0,
                        "maximum": 28.0
                    },
                    "zone": {
                        "type": "string",
                        "enum": [
                            "driver",
                            "passenger",
                            "rear_left",
                            "rear_right",
                            "all"
                        ]
                    }
                },
                "required": [
                    "temperature"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_window_defrost",
            "description": "Activate or deactivate windshield defrost system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {
                        "type": "boolean"
                    },
                    "defrost_window": {
                        "type": "string",
                        "enum": [
                            "FRONT",
                            "REAR",
                            "BOTH"
                        ]
                    }
                },
                "required": [
                    "on",
                    "defrost_window"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_navigation",
            "description": "Start navigation along a route.",
            "parameters": {
                "type": "object",
                "properties": {
                    "route_id": {
                        "type": "string"
                    }
                },
                "required": [
                    "route_id"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop_navigation",
            "description": "Stop current navigation session.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Quick lookup map for tools
TOOL_NAME_MAP: Dict[str, Dict[str, Any]] = {
    tool["function"]["name"]: tool for tool in ALL_CAR_TOOLS
}

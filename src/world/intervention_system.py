"""
Intervention System - Player interaction and influence system.

This module handles various ways players can interact with and influence
the game world and characters without directly controlling them.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class InterventionType(Enum):
    """Types of player interventions."""
    WHISPER_SUGGESTION = "whisper_suggestion"
    ENVIRONMENT_CHANGE = "environment_change"
    ITEM_PLACEMENT = "item_placement"
    MOOD_INFLUENCE = "mood_influence"
    EVENT_TRIGGER = "event_trigger"
    DIRECT_COMMUNICATION = "direct_communication"
    CHAOS_BUTTON = "chaos_button"


class InterventionImpact(Enum):
    """Impact levels of interventions."""
    SUBTLE = "subtle"
    NOTICEABLE = "noticeable"
    OBVIOUS = "obvious"
    DRAMATIC = "dramatic"


@dataclass
class InterventionRecord:
    """Record of a player intervention."""
    intervention_id: str
    intervention_type: InterventionType
    target: str  # character_id or "world"
    description: str
    impact_level: InterventionImpact
    timestamp: float
    success_rate: float
    actual_outcome: Optional[str] = None
    character_awareness: float = 0.0  # How aware character is of intervention


class WhisperSuggestion:
    """System for whispering suggestions to characters."""

    def __init__(self):
        self.suggestion_history: List[Dict[str, Any]] = []
        self.character_receptivity: Dict[str, float] = {}  # character_id -> receptivity

    async def whisper_to_character(self, character_id: str, suggestion: str,
                                 character_traits: Dict[str, Any],
                                 current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Whisper a suggestion to a character."""

        # Calculate receptivity based on character traits
        receptivity = self._calculate_receptivity(character_traits, suggestion)

        # Modify receptivity based on context
        context_modifier = self._get_context_modifier(current_context)
        final_receptivity = max(0.0, min(1.0, receptivity + context_modifier))

        # Determine if suggestion is accepted
        import random
        suggestion_accepted = random.random() < final_receptivity

        # Create intervention record
        intervention = {
            "intervention_id": f"whisper_{int(time.time())}_{character_id}",
            "target": character_id,
            "suggestion": suggestion,
            "receptivity": final_receptivity,
            "accepted": suggestion_accepted,
            "timestamp": time.time(),
            "context": current_context.copy()
        }

        self.suggestion_history.append(intervention)

        # Update character receptivity (characters become more/less receptive over time)
        current_receptivity = self.character_receptivity.get(character_id, 0.5)
        if suggestion_accepted:
            # Successful suggestions slightly increase future receptivity
            self.character_receptivity[character_id] = min(1.0, current_receptivity + 0.05)
        else:
            # Failed suggestions slightly decrease future receptivity
            self.character_receptivity[character_id] = max(0.0, current_receptivity - 0.02)

        return {
            "success": suggestion_accepted,
            "receptivity": final_receptivity,
            "intervention_id": intervention["intervention_id"],
            "character_response": self._generate_character_response(
                character_traits, suggestion, suggestion_accepted
            )
        }

    def _calculate_receptivity(self, character_traits: Dict[str, Any], suggestion: str) -> float:
        """Calculate how receptive a character is to a suggestion."""
        base_receptivity = 0.5

        # Trait-based modifiers
        trait_modifiers = {
            "open_minded": 0.2,
            "stubborn": -0.3,
            "curious": 0.15,
            "skeptical": -0.2,
            "trusting": 0.25,
            "paranoid": -0.35,
            "adventurous": 0.1,
            "cautious": -0.1,
            "impulsive": 0.2,
            "analytical": -0.05
        }

        receptivity = base_receptivity
        traits = character_traits.get("base_traits", [])

        for trait in traits:
            modifier = trait_modifiers.get(trait, 0.0)
            receptivity += modifier

        # Analyze suggestion content for additional modifiers
        suggestion_lower = suggestion.lower()

        if any(word in suggestion_lower for word in ["dangerous", "risky", "bold"]):
            if "adventurous" in traits:
                receptivity += 0.1
            if "cautious" in traits:
                receptivity -= 0.15

        if any(word in suggestion_lower for word in ["help", "kind", "nice"]):
            if "empathetic" in traits:
                receptivity += 0.1
            if "selfish" in traits:
                receptivity -= 0.1

        if any(word in suggestion_lower for word in ["fun", "party", "social"]):
            if "extroverted" in traits:
                receptivity += 0.1
            if "introverted" in traits:
                receptivity -= 0.05

        return max(0.0, min(1.0, receptivity))

    def _get_context_modifier(self, context: Dict[str, Any]) -> float:
        """Get context-based modifier for receptivity."""
        modifier = 0.0

        # Mood affects receptivity
        mood = context.get("mood", "neutral")
        mood_modifiers = {
            "happy": 0.1,
            "excited": 0.15,
            "sad": -0.1,
            "angry": -0.2,
            "confused": 0.05,
            "bored": 0.2
        }
        modifier += mood_modifiers.get(mood, 0.0)

        # Energy level affects receptivity
        energy = context.get("energy", 0.5)
        if energy > 0.8:
            modifier += 0.1  # High energy = more open to suggestions
        elif energy < 0.3:
            modifier -= 0.1  # Low energy = less receptive

        # Social situation affects receptivity
        if context.get("alone", False):
            modifier += 0.05  # Slightly more receptive when alone
        elif context.get("in_group", False):
            modifier -= 0.05  # Less receptive in group settings

        # Current activity affects receptivity
        activity = context.get("current_activity", "")
        if "stuck" in activity or "confused" in activity:
            modifier += 0.2  # More receptive when struggling
        elif "focused" in activity or "working" in activity:
            modifier -= 0.1  # Less receptive when concentrated

        return modifier

    def _generate_character_response(self, character_traits: Dict[str, Any],
                                   suggestion: str, accepted: bool) -> str:
        """Generate a character's internal response to a suggestion."""
        traits = character_traits.get("base_traits", [])

        if accepted:
            if "analytical" in traits:
                return "That's actually a reasonable idea..."
            elif "impulsive" in traits:
                return "Yes! Let's do that right now!"
            elif "cautious" in traits:
                return "Well, I suppose that could work..."
            else:
                return "Hmm, that sounds like a good idea."
        else:
            if "stubborn" in traits:
                return "No way, I'm doing this my way."
            elif "skeptical" in traits:
                return "That doesn't sound right to me..."
            elif "paranoid" in traits:
                return "Where did that thought come from? Suspicious..."
            else:
                return "Nah, I don't think I want to do that."

    def get_character_suggestion_history(self, character_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent suggestion history for a character."""
        character_suggestions = [
            s for s in self.suggestion_history
            if s["target"] == character_id
        ]
        return character_suggestions[-limit:] if character_suggestions else []


class EnvironmentManipulator:
    """System for manipulating environmental factors."""

    def __init__(self):
        self.manipulation_history: List[InterventionRecord] = []
        self.active_manipulations: Dict[str, Any] = {}

    def change_lighting(self, location: str, new_lighting: str) -> InterventionRecord:
        """Change lighting in a location."""
        intervention = InterventionRecord(
            intervention_id=f"lighting_{int(time.time())}",
            intervention_type=InterventionType.ENVIRONMENT_CHANGE,
            target=location,
            description=f"Changed lighting to {new_lighting}",
            impact_level=InterventionImpact.SUBTLE,
            timestamp=time.time(),
            success_rate=0.95,
            actual_outcome=f"Lighting in {location} is now {new_lighting}"
        )

        self.active_manipulations[f"{location}_lighting"] = {
            "type": "lighting",
            "value": new_lighting,
            "start_time": time.time()
        }

        self.manipulation_history.append(intervention)
        return intervention

    def change_temperature(self, location: str, temperature_change: float) -> InterventionRecord:
        """Change temperature in a location."""
        intervention = InterventionRecord(
            intervention_id=f"temp_{int(time.time())}",
            intervention_type=InterventionType.ENVIRONMENT_CHANGE,
            target=location,
            description=f"Changed temperature by {temperature_change}°C",
            impact_level=InterventionImpact.NOTICEABLE,
            timestamp=time.time(),
            success_rate=0.9,
            actual_outcome=f"Temperature in {location} adjusted"
        )

        self.active_manipulations[f"{location}_temperature"] = {
            "type": "temperature",
            "change": temperature_change,
            "start_time": time.time()
        }

        self.manipulation_history.append(intervention)
        return intervention

    def trigger_weather_change(self, new_weather: str) -> InterventionRecord:
        """Change the weather."""
        impact_level = InterventionImpact.OBVIOUS if new_weather in ["stormy", "snowy"] else InterventionImpact.NOTICEABLE

        intervention = InterventionRecord(
            intervention_id=f"weather_{int(time.time())}",
            intervention_type=InterventionType.ENVIRONMENT_CHANGE,
            target="world",
            description=f"Changed weather to {new_weather}",
            impact_level=impact_level,
            timestamp=time.time(),
            success_rate=0.8,
            actual_outcome=f"Weather is now {new_weather}"
        )

        self.active_manipulations["global_weather"] = {
            "type": "weather",
            "value": new_weather,
            "start_time": time.time()
        }

        self.manipulation_history.append(intervention)
        return intervention

    def place_item(self, item_type: str, location: str, item_properties: Dict[str, Any]) -> InterventionRecord:
        """Place an item in the world."""
        intervention = InterventionRecord(
            intervention_id=f"item_{int(time.time())}",
            intervention_type=InterventionType.ITEM_PLACEMENT,
            target=location,
            description=f"Placed {item_type} in {location}",
            impact_level=InterventionImpact.NOTICEABLE,
            timestamp=time.time(),
            success_rate=1.0,
            actual_outcome=f"{item_type} appeared in {location}"
        )

        item_id = f"placed_{item_type}_{int(time.time())}"
        self.active_manipulations[item_id] = {
            "type": "item_placement",
            "item_type": item_type,
            "location": location,
            "properties": item_properties,
            "start_time": time.time()
        }

        self.manipulation_history.append(intervention)
        return intervention


class MoodInfluencer:
    """System for subtly influencing character moods."""

    def __init__(self):
        self.influence_history: List[InterventionRecord] = []
        self.mood_resistance: Dict[str, float] = {}  # Character resistance to mood changes

    def influence_mood(self, character_id: str, target_mood: str,
                      intensity: float = 0.3) -> InterventionRecord:
        """Attempt to influence a character's mood."""

        # Check resistance
        resistance = self.mood_resistance.get(character_id, 0.0)
        effective_intensity = max(0.0, intensity - resistance)

        # Success rate based on intensity and resistance
        success_rate = min(0.9, effective_intensity * 2)

        import random
        success = random.random() < success_rate

        intervention = InterventionRecord(
            intervention_id=f"mood_{int(time.time())}_{character_id}",
            intervention_type=InterventionType.MOOD_INFLUENCE,
            target=character_id,
            description=f"Attempted to make {character_id} feel {target_mood}",
            impact_level=InterventionImpact.SUBTLE,
            timestamp=time.time(),
            success_rate=success_rate,
            actual_outcome=f"Mood influence {'succeeded' if success else 'failed'}",
            character_awareness=0.1  # Characters rarely notice mood influences
        )

        # Increase resistance slightly after each attempt
        self.mood_resistance[character_id] = min(0.8, resistance + 0.05)

        self.influence_history.append(intervention)

        return intervention


class DirectCommunication:
    """System for direct player-character communication."""

    def __init__(self):
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}  # character_id -> messages

    async def send_message_to_character(self, character_id: str, message: str,
                                      character_ai_interface) -> Dict[str, Any]:
        """Send a direct message to a character and get their response."""

        # Initialize conversation history if needed
        if character_id not in self.conversation_history:
            self.conversation_history[character_id] = []

        # Add player message to history
        player_message = {
            "sender": "player",
            "message": message,
            "timestamp": time.time()
        }
        self.conversation_history[character_id].append(player_message)

        # Get character response through AI interface
        try:
            response = await character_ai_interface.respond_to_dialogue("Player", message)

            # Add character response to history
            character_message = {
                "sender": character_id,
                "message": response,
                "timestamp": time.time()
            }
            self.conversation_history[character_id].append(character_message)

            return {
                "success": True,
                "response": response,
                "conversation_id": f"conv_{character_id}_{int(time.time())}"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_response": "I'm sorry, I don't know what to say to that."
            }

    def get_conversation_history(self, character_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history with a character."""
        history = self.conversation_history.get(character_id, [])
        return history[-limit:] if history else []


class InterventionSystem:
    """Main intervention system managing all player interactions."""

    def __init__(self):
        self.whisper_system = WhisperSuggestion()
        self.environment_system = EnvironmentManipulator()
        self.mood_system = MoodInfluencer()
        self.communication_system = DirectCommunication()
        self.chaos_events_system = None  # Will be injected

        self.intervention_limits = {
            InterventionType.WHISPER_SUGGESTION: {"per_hour": 10, "per_day": 50},
            InterventionType.ENVIRONMENT_CHANGE: {"per_hour": 5, "per_day": 20},
            InterventionType.MOOD_INFLUENCE: {"per_hour": 3, "per_day": 15},
            InterventionType.CHAOS_BUTTON: {"per_hour": 2, "per_day": 8},
        }

        self.intervention_usage = {}  # Track usage for limits

    def set_chaos_system(self, chaos_system):
        """Inject chaos events system."""
        self.chaos_events_system = chaos_system

    async def whisper_suggestion(self, character_id: str, suggestion: str,
                                character_traits: Dict[str, Any],
                                current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Whisper a suggestion to a character."""
        if not self._check_intervention_limits(InterventionType.WHISPER_SUGGESTION):
            return {"success": False, "error": "Intervention limit exceeded"}

        result = await self.whisper_system.whisper_to_character(
            character_id, suggestion, character_traits, current_context
        )

        self._record_intervention_usage(InterventionType.WHISPER_SUGGESTION)
        return result

    def manipulate_environment(self, manipulation_type: str, target: str, **kwargs) -> InterventionRecord:
        """Manipulate environmental factors."""
        if not self._check_intervention_limits(InterventionType.ENVIRONMENT_CHANGE):
            return InterventionRecord(
                intervention_id="failed",
                intervention_type=InterventionType.ENVIRONMENT_CHANGE,
                target=target,
                description="Failed - limit exceeded",
                impact_level=InterventionImpact.SUBTLE,
                timestamp=time.time(),
                success_rate=0.0
            )

        if manipulation_type == "lighting":
            result = self.environment_system.change_lighting(target, kwargs.get("new_lighting", "normal"))
        elif manipulation_type == "temperature":
            result = self.environment_system.change_temperature(target, kwargs.get("temperature_change", 0))
        elif manipulation_type == "weather":
            result = self.environment_system.trigger_weather_change(kwargs.get("new_weather", "sunny"))
        elif manipulation_type == "place_item":
            result = self.environment_system.place_item(
                kwargs.get("item_type", "mystery_box"),
                target,
                kwargs.get("item_properties", {})
            )
        else:
            return InterventionRecord(
                intervention_id="invalid",
                intervention_type=InterventionType.ENVIRONMENT_CHANGE,
                target=target,
                description=f"Invalid manipulation type: {manipulation_type}",
                impact_level=InterventionImpact.SUBTLE,
                timestamp=time.time(),
                success_rate=0.0
            )

        self._record_intervention_usage(InterventionType.ENVIRONMENT_CHANGE)
        return result

    def influence_mood(self, character_id: str, target_mood: str, intensity: float = 0.3) -> InterventionRecord:
        """Influence a character's mood."""
        if not self._check_intervention_limits(InterventionType.MOOD_INFLUENCE):
            return InterventionRecord(
                intervention_id="failed",
                intervention_type=InterventionType.MOOD_INFLUENCE,
                target=character_id,
                description="Failed - limit exceeded",
                impact_level=InterventionImpact.SUBTLE,
                timestamp=time.time(),
                success_rate=0.0
            )

        result = self.mood_system.influence_mood(character_id, target_mood, intensity)
        self._record_intervention_usage(InterventionType.MOOD_INFLUENCE)
        return result

    async def communicate_with_character(self, character_id: str, message: str,
                                       character_ai_interface) -> Dict[str, Any]:
        """Communicate directly with a character."""
        return await self.communication_system.send_message_to_character(
            character_id, message, character_ai_interface
        )

    def trigger_chaos_event(self, simulation_mode: str, character_states: Dict[str, Any],
                          world_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Trigger a chaos event."""
        if not self._check_intervention_limits(InterventionType.CHAOS_BUTTON):
            return {"success": False, "error": "Chaos button limit exceeded"}

        if not self.chaos_events_system:
            return {"success": False, "error": "Chaos system not available"}

        result = self.chaos_events_system.trigger_chaos_event(
            simulation_mode, character_states, world_state
        )

        self._record_intervention_usage(InterventionType.CHAOS_BUTTON)
        return result

    def _check_intervention_limits(self, intervention_type: InterventionType) -> bool:
        """Check if intervention type is within limits."""
        current_time = time.time()
        limits = self.intervention_limits.get(intervention_type, {})

        if intervention_type not in self.intervention_usage:
            self.intervention_usage[intervention_type] = []

        usage_times = self.intervention_usage[intervention_type]

        # Remove old usage records
        hour_ago = current_time - 3600
        day_ago = current_time - 86400

        usage_times[:] = [t for t in usage_times if t > day_ago]

        # Check limits
        recent_hour = [t for t in usage_times if t > hour_ago]
        recent_day = usage_times

        per_hour_limit = limits.get("per_hour", float('inf'))
        per_day_limit = limits.get("per_day", float('inf'))

        return len(recent_hour) < per_hour_limit and len(recent_day) < per_day_limit

    def _record_intervention_usage(self, intervention_type: InterventionType):
        """Record usage of an intervention type."""
        if intervention_type not in self.intervention_usage:
            self.intervention_usage[intervention_type] = []

        self.intervention_usage[intervention_type].append(time.time())

    def get_intervention_summary(self) -> Dict[str, Any]:
        """Get summary of intervention system state."""
        current_time = time.time()
        hour_ago = current_time - 3600

        usage_summary = {}
        for intervention_type, times in self.intervention_usage.items():
            recent_usage = [t for t in times if t > hour_ago]
            limits = self.intervention_limits.get(intervention_type, {})

            usage_summary[intervention_type.value] = {
                "used_last_hour": len(recent_usage),
                "limit_per_hour": limits.get("per_hour", "unlimited"),
                "remaining": max(0, limits.get("per_hour", float('inf')) - len(recent_usage))
            }

        return {
            "usage_summary": usage_summary,
            "total_interventions": sum(len(times) for times in self.intervention_usage.values()),
            "active_environmental_changes": len(self.environment_system.active_manipulations),
            "conversation_participants": len(self.communication_system.conversation_history)
        }
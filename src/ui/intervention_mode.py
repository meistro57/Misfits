"""
Intervention Mode - UI interface for player interventions and influence.

This module provides the user interface for all intervention systems,
allowing players to influence characters and the world without direct control.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time


class InterventionUIMode(Enum):
    """Different modes of the intervention UI."""
    WHISPER = "whisper"
    ENVIRONMENT = "environment"
    COMMUNICATION = "communication"
    OVERVIEW = "overview"


@dataclass
class InterventionUIConfig:
    """Configuration for intervention UI behavior."""
    show_success_rates: bool = True
    show_character_awareness: bool = False  # Debug mode
    enable_suggestion_history: bool = True
    max_suggestion_length: int = 200
    confirmation_required_for_major: bool = True


class WhisperUI:
    """UI component for the whisper suggestion system."""

    def __init__(self):
        self.active_whispers: Dict[str, Dict[str, Any]] = {}
        self.suggestion_templates = self._initialize_suggestion_templates()

    def _initialize_suggestion_templates(self) -> Dict[str, List[str]]:
        """Initialize common suggestion templates."""
        return {
            "social": [
                "Why don't you talk to {character}?",
                "You should spend more time with friends",
                "Maybe you could invite someone over",
                "You seem like you need some company"
            ],
            "productive": [
                "You could work on improving your skills",
                "Now might be a good time to clean up",
                "You should focus on your goals",
                "Why not try something creative?"
            ],
            "adventurous": [
                "You should explore somewhere new",
                "Maybe it's time to take a risk",
                "You could try something you've never done before",
                "Adventure awaits outside"
            ],
            "reflective": [
                "You might want to think about your relationships",
                "It's good to reflect on your choices",
                "You should consider what makes you happy",
                "Maybe it's time for some self-care"
            ],
            "mischievous": [
                "You could play a harmless prank",
                "A little chaos might be fun",
                "You should stir things up a bit",
                "Why not cause some harmless trouble?"
            ]
        }

    def create_whisper_interface(self, character_id: str, character_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create whisper interface for a specific character."""

        # Get character receptivity info
        personality_traits = character_info.get("traits", [])
        current_mood = character_info.get("mood", "neutral")
        current_activity = character_info.get("current_activity", "idle")

        # Calculate estimated receptivity
        estimated_receptivity = self._estimate_receptivity(personality_traits, current_mood)

        # Get relevant suggestion templates
        relevant_templates = self._get_relevant_templates(personality_traits, current_mood, current_activity)

        interface = {
            "character_id": character_id,
            "character_name": character_info.get("name", character_id),
            "estimated_receptivity": estimated_receptivity,
            "receptivity_factors": self._analyze_receptivity_factors(personality_traits, current_mood),
            "suggestion_templates": relevant_templates,
            "character_state": {
                "mood": current_mood,
                "activity": current_activity,
                "location": character_info.get("location", "unknown"),
                "energy": character_info.get("energy", 0.5)
            },
            "recent_whispers": self._get_recent_whispers(character_id)
        }

        return interface

    def _estimate_receptivity(self, traits: List[str], mood: str) -> float:
        """Estimate character's receptivity to suggestions."""
        base_receptivity = 0.5

        # Trait modifiers
        trait_effects = {
            "open_minded": 0.15,
            "stubborn": -0.25,
            "curious": 0.10,
            "trusting": 0.20,
            "skeptical": -0.15,
            "impulsive": 0.15,
            "cautious": -0.10
        }

        for trait in traits:
            base_receptivity += trait_effects.get(trait, 0.0)

        # Mood modifiers
        mood_effects = {
            "happy": 0.10,
            "excited": 0.15,
            "sad": -0.05,
            "angry": -0.20,
            "confused": 0.10,
            "bored": 0.20
        }

        base_receptivity += mood_effects.get(mood, 0.0)

        return max(0.0, min(1.0, base_receptivity))

    def _analyze_receptivity_factors(self, traits: List[str], mood: str) -> Dict[str, str]:
        """Analyze what factors affect receptivity."""
        factors = {}

        positive_traits = ["open_minded", "curious", "trusting", "impulsive"]
        negative_traits = ["stubborn", "skeptical", "paranoid", "cautious"]

        positive_found = [t for t in traits if t in positive_traits]
        negative_found = [t for t in traits if t in negative_traits]

        if positive_found:
            factors["positive_traits"] = f"More receptive due to: {', '.join(positive_found)}"

        if negative_found:
            factors["negative_traits"] = f"Less receptive due to: {', '.join(negative_found)}"

        mood_effects = {
            "happy": "Good mood increases receptivity",
            "excited": "High energy makes them more open to ideas",
            "sad": "Sadness slightly reduces receptivity",
            "angry": "Anger makes them resistant to suggestions",
            "confused": "Confusion makes them seek guidance",
            "bored": "Boredom makes them eager for new ideas"
        }

        if mood in mood_effects:
            factors["mood_effect"] = mood_effects[mood]

        return factors

    def _get_relevant_templates(self, traits: List[str], mood: str, activity: str) -> Dict[str, List[str]]:
        """Get suggestion templates relevant to the character's current state."""
        relevant = {}

        # Always include general suggestions
        relevant["general"] = [
            "You could try something different",
            "Maybe a change of pace would be good",
            "Follow your instincts"
        ]

        # Add trait-based suggestions
        if "extroverted" in traits or "social" in traits:
            relevant["social"] = self.suggestion_templates["social"]

        if "creative" in traits or "artistic" in traits:
            relevant["creative"] = [
                "You should express your creativity",
                "Maybe it's time to make something beautiful",
                "Your artistic side is calling"
            ]

        if "adventurous" in traits:
            relevant["adventure"] = self.suggestion_templates["adventurous"]

        # Add mood-based suggestions
        if mood == "bored":
            relevant["anti_boredom"] = [
                "You need some excitement in your life",
                "Time to shake things up",
                "Why not try something completely new?"
            ]

        if mood == "sad":
            relevant["mood_boost"] = [
                "You deserve to feel better",
                "Maybe some self-care would help",
                "You should do something that makes you smile"
            ]

        # Add activity-based suggestions
        if "stuck" in activity or "confused" in activity:
            relevant["guidance"] = [
                "Trust your instincts on this one",
                "Sometimes the simple solution is best",
                "You know more than you think you do"
            ]

        return relevant

    def _get_recent_whispers(self, character_id: str) -> List[Dict[str, Any]]:
        """Get recent whisper history for character."""
        character_whispers = self.active_whispers.get(character_id, {})
        return character_whispers.get("history", [])[-5:]  # Last 5 whispers


class EnvironmentUI:
    """UI component for environmental manipulation."""

    def __init__(self):
        self.available_manipulations = self._initialize_manipulations()

    def _initialize_manipulations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available environmental manipulations."""
        return {
            "lighting": {
                "name": "Adjust Lighting",
                "description": "Change the lighting in a location",
                "options": ["bright", "dim", "dark", "colorful", "natural"],
                "impact": "subtle",
                "cooldown": 60,  # seconds
                "targets": ["location"]
            },
            "temperature": {
                "name": "Temperature Control",
                "description": "Adjust the temperature up or down",
                "options": {"range": [-10, 10], "step": 1, "unit": "°C"},
                "impact": "noticeable",
                "cooldown": 120,
                "targets": ["location", "global"]
            },
            "weather": {
                "name": "Weather Control",
                "description": "Change the weather outside",
                "options": ["sunny", "cloudy", "rainy", "stormy", "snowy", "foggy"],
                "impact": "obvious",
                "cooldown": 300,  # 5 minutes
                "targets": ["global"]
            },
            "item_placement": {
                "name": "Place Mystery Item",
                "description": "Place an interesting item somewhere",
                "options": [
                    "mystery_box", "love_letter", "old_photo", "strange_device",
                    "gift_basket", "puzzle_piece", "lucky_coin", "rare_book"
                ],
                "impact": "noticeable",
                "cooldown": 180,
                "targets": ["location"]
            },
            "sound_effects": {
                "name": "Ambient Sounds",
                "description": "Add background sounds to a location",
                "options": [
                    "birds_chirping", "rain_sounds", "music_box", "mysterious_humming",
                    "distant_laughter", "clock_ticking", "wind_chimes"
                ],
                "impact": "subtle",
                "cooldown": 90,
                "targets": ["location"]
            }
        }

    def get_manipulation_interface(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get interface for environmental manipulation."""
        current_time = time.time()

        interface = {
            "available_manipulations": {},
            "world_state": {
                "current_weather": world_state.get("weather", "sunny"),
                "time_of_day": world_state.get("time_of_day", "day"),
                "active_events": world_state.get("active_events", [])
            },
            "locations": world_state.get("locations", {}),
            "cooldowns": {}
        }

        # Add manipulation options with availability status
        for manip_id, manip_info in self.available_manipulations.items():
            # Check if manipulation is on cooldown
            # (This would be tracked in the actual intervention system)
            on_cooldown = False  # Simplified for now

            interface["available_manipulations"][manip_id] = {
                **manip_info,
                "available": not on_cooldown,
                "estimated_success_rate": self._estimate_manipulation_success(manip_id, world_state)
            }

        return interface

    def _estimate_manipulation_success(self, manipulation_type: str, world_state: Dict[str, Any]) -> float:
        """Estimate success rate for a manipulation."""
        base_rates = {
            "lighting": 0.95,
            "temperature": 0.90,
            "weather": 0.80,
            "item_placement": 1.0,
            "sound_effects": 0.85
        }

        return base_rates.get(manipulation_type, 0.75)


class CommunicationUI:
    """UI component for direct character communication."""

    def __init__(self):
        self.conversation_states: Dict[str, Dict[str, Any]] = {}

    def get_communication_interface(self, character_id: str, character_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get interface for communicating with a character."""

        # Initialize conversation state if needed
        if character_id not in self.conversation_states:
            self.conversation_states[character_id] = {
                "messages": [],
                "character_mood": character_info.get("mood", "neutral"),
                "last_interaction": 0
            }

        conversation = self.conversation_states[character_id]

        interface = {
            "character_id": character_id,
            "character_name": character_info.get("name", character_id),
            "character_info": {
                "mood": character_info.get("mood", "neutral"),
                "location": character_info.get("location", "unknown"),
                "current_activity": character_info.get("current_activity", "idle"),
                "personality_summary": character_info.get("traits", [])[:3]  # Top 3 traits
            },
            "conversation_history": conversation["messages"][-10:],  # Last 10 messages
            "suggested_topics": self._generate_suggested_topics(character_info),
            "conversation_mood": self._assess_conversation_mood(conversation["messages"]),
            "character_availability": self._assess_character_availability(character_info)
        }

        return interface

    def _generate_suggested_topics(self, character_info: Dict[str, Any]) -> List[str]:
        """Generate suggested conversation topics based on character."""
        traits = character_info.get("traits", [])
        mood = character_info.get("mood", "neutral")
        activity = character_info.get("current_activity", "idle")

        topics = ["How are you feeling today?", "What's on your mind?"]

        # Trait-based topics
        if "creative" in traits:
            topics.append("What creative projects are you working on?")
        if "social" in traits:
            topics.append("Tell me about your friends")
        if "philosophical" in traits:
            topics.append("What do you think about the meaning of life?")
        if "adventurous" in traits:
            topics.append("What adventures have you been on lately?")

        # Mood-based topics
        if mood == "happy":
            topics.append("You seem happy! What's going well?")
        elif mood == "sad":
            topics.append("You seem down. Want to talk about it?")
        elif mood == "excited":
            topics.append("You're full of energy! What's exciting you?")

        # Activity-based topics
        if "work" in activity:
            topics.append("How's work going?")
        elif "social" in activity:
            topics.append("Are you enjoying the company?")

        return topics[:6]  # Limit to 6 suggestions

    def _assess_conversation_mood(self, messages: List[Dict[str, Any]]) -> str:
        """Assess the overall mood of the conversation."""
        if not messages:
            return "neutral"

        # Simple sentiment analysis based on recent messages
        recent_messages = messages[-3:]

        # This would be more sophisticated in a real implementation
        positive_indicators = ["happy", "good", "great", "fun", "love", "excited"]
        negative_indicators = ["sad", "bad", "angry", "hate", "frustrated", "tired"]

        positive_count = 0
        negative_count = 0

        for message in recent_messages:
            content = message.get("message", "").lower()
            positive_count += sum(1 for word in positive_indicators if word in content)
            negative_count += sum(1 for word in negative_indicators if word in content)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _assess_character_availability(self, character_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess how available/receptive character is to conversation."""
        activity = character_info.get("current_activity", "idle")
        mood = character_info.get("mood", "neutral")
        energy = character_info.get("energy", 0.5)

        # Calculate availability score
        availability = 0.5

        # Activity modifiers
        if activity in ["idle", "relaxing", "bored"]:
            availability += 0.3
        elif activity in ["socializing", "talking"]:
            availability += 0.2
        elif activity in ["working", "focused", "sleeping"]:
            availability -= 0.3
        elif activity in ["angry", "arguing"]:
            availability -= 0.4

        # Mood modifiers
        mood_effects = {
            "happy": 0.2,
            "excited": 0.1,
            "bored": 0.3,
            "sad": -0.1,
            "angry": -0.3,
            "tired": -0.2
        }
        availability += mood_effects.get(mood, 0.0)

        # Energy modifiers
        if energy > 0.7:
            availability += 0.1
        elif energy < 0.3:
            availability -= 0.2

        availability = max(0.0, min(1.0, availability))

        return {
            "score": availability,
            "status": "high" if availability > 0.7 else "medium" if availability > 0.4 else "low",
            "description": self._get_availability_description(availability, activity, mood)
        }

    def _get_availability_description(self, availability: float, activity: str, mood: str) -> str:
        """Get human-readable description of character availability."""
        if availability > 0.8:
            return "Very receptive to conversation"
        elif availability > 0.6:
            return "Open to chatting"
        elif availability > 0.4:
            return "Somewhat distracted but willing to talk"
        elif availability > 0.2:
            return "Not very interested in conversation right now"
        else:
            return "Seems busy or preoccupied"


class InterventionModeUI:
    """Main UI controller for intervention systems."""

    def __init__(self, config: InterventionUIConfig = None):
        self.config = config or InterventionUIConfig()
        self.current_mode = InterventionUIMode.OVERVIEW

        # Sub-components
        self.whisper_ui = WhisperUI()
        self.environment_ui = EnvironmentUI()
        self.communication_ui = CommunicationUI()

        # UI state
        self.selected_character = None
        self.intervention_history: List[Dict[str, Any]] = []

    def set_mode(self, mode: InterventionUIMode):
        """Set the current intervention UI mode."""
        self.current_mode = mode

    def get_main_interface(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get the main intervention interface based on current mode."""

        base_interface = {
            "current_mode": self.current_mode.value,
            "available_modes": [mode.value for mode in InterventionUIMode],
            "intervention_limits": self._get_intervention_limits(),
            "recent_interventions": self.intervention_history[-5:]
        }

        if self.current_mode == InterventionUIMode.WHISPER:
            if self.selected_character:
                character_info = game_state.get("characters", {}).get(self.selected_character, {})
                base_interface["whisper_interface"] = self.whisper_ui.create_whisper_interface(
                    self.selected_character, character_info
                )
            base_interface["available_characters"] = list(game_state.get("characters", {}).keys())

        elif self.current_mode == InterventionUIMode.ENVIRONMENT:
            base_interface["environment_interface"] = self.environment_ui.get_manipulation_interface(
                game_state.get("world", {})
            )

        elif self.current_mode == InterventionUIMode.COMMUNICATION:
            if self.selected_character:
                character_info = game_state.get("characters", {}).get(self.selected_character, {})
                base_interface["communication_interface"] = self.communication_ui.get_communication_interface(
                    self.selected_character, character_info
                )
            base_interface["available_characters"] = list(game_state.get("characters", {}).keys())

        elif self.current_mode == InterventionUIMode.OVERVIEW:
            base_interface["overview"] = self._get_intervention_overview(game_state)

        return base_interface

    def _get_intervention_limits(self) -> Dict[str, Any]:
        """Get current intervention usage and limits."""
        # This would interface with the actual intervention system
        return {
            "whispers": {"used": 3, "limit": 10, "per": "hour"},
            "environment": {"used": 1, "limit": 5, "per": "hour"},
            "chaos": {"used": 1, "limit": 2, "per": "hour"}
        }

    def _get_intervention_overview(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get overview of all intervention possibilities."""
        characters = game_state.get("characters", {})

        overview = {
            "character_summary": {},
            "world_opportunities": [],
            "recommended_interventions": []
        }

        # Character summaries
        for char_id, char_info in characters.items():
            overview["character_summary"][char_id] = {
                "name": char_info.get("name", char_id),
                "mood": char_info.get("mood", "neutral"),
                "activity": char_info.get("current_activity", "idle"),
                "receptivity": self.whisper_ui._estimate_receptivity(
                    char_info.get("traits", []), char_info.get("mood", "neutral")
                ),
                "last_interaction": "never"  # Would track this in real implementation
            }

        # World opportunities
        world_state = game_state.get("world", {})
        if world_state.get("weather") == "rainy":
            overview["world_opportunities"].append("Rainy weather - good time for cozy indoor activities")

        if world_state.get("time_of_day") == "evening":
            overview["world_opportunities"].append("Evening time - perfect for social gatherings")

        # Recommended interventions
        overview["recommended_interventions"] = self._generate_recommendations(game_state)

        return overview

    def _generate_recommendations(self, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommended interventions based on current state."""
        recommendations = []
        characters = game_state.get("characters", {})

        # Look for characters who might benefit from interventions
        for char_id, char_info in characters.items():
            mood = char_info.get("mood", "neutral")
            activity = char_info.get("current_activity", "idle")

            if mood == "bored":
                recommendations.append({
                    "type": "whisper",
                    "target": char_id,
                    "suggestion": "Suggest a fun activity to cure their boredom",
                    "priority": "medium"
                })

            if mood == "sad":
                recommendations.append({
                    "type": "communication",
                    "target": char_id,
                    "suggestion": "Have a supportive conversation",
                    "priority": "high"
                })

            if activity == "stuck":
                recommendations.append({
                    "type": "whisper",
                    "target": char_id,
                    "suggestion": "Whisper helpful guidance",
                    "priority": "high"
                })

        # Environmental recommendations
        world_state = game_state.get("world", {})
        if len([c for c in characters.values() if c.get("mood") == "sad"]) > 1:
            recommendations.append({
                "type": "environment",
                "target": "global",
                "suggestion": "Brighten the lighting to improve mood",
                "priority": "medium"
            })

        return recommendations[:5]  # Top 5 recommendations

    def select_character(self, character_id: str):
        """Select a character for targeted interventions."""
        self.selected_character = character_id

    def record_intervention(self, intervention_type: str, target: str,
                          result: Dict[str, Any]):
        """Record an intervention for history tracking."""
        record = {
            "timestamp": time.time(),
            "type": intervention_type,
            "target": target,
            "success": result.get("success", False),
            "description": result.get("description", "Unknown intervention")
        }

        self.intervention_history.append(record)

        # Keep only recent history
        if len(self.intervention_history) > 50:
            self.intervention_history = self.intervention_history[-50:]
"""
Chaos Events - Random world-altering events triggered by the chaos button.

This module handles the generation and execution of chaotic events
that can dramatically alter the game world and character relationships.
"""

import random
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ChaosLevel(Enum):
    """Levels of chaos for events."""
    MILD = "mild"
    MODERATE = "moderate"
    MAJOR = "major"
    EXTREME = "extreme"


class EventCategory(Enum):
    """Categories of chaos events."""
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    PERSONAL = "personal"
    TECHNOLOGICAL = "technological"
    SUPERNATURAL = "supernatural"
    COMEDIC = "comedic"
    DRAMATIC = "dramatic"


@dataclass
class ChaosEvent:
    """Definition of a chaos event."""
    event_id: str
    name: str
    description: str
    category: EventCategory
    chaos_level: ChaosLevel
    probability_weight: float
    duration_minutes: int
    prerequisites: List[str]  # Conditions that must be met
    character_requirements: Dict[str, Any]  # Required character states
    world_requirements: Dict[str, Any]  # Required world conditions
    immediate_effects: Dict[str, Any]
    ongoing_effects: Dict[str, Any]
    resolution_conditions: List[str]
    narrative_hooks: List[str]  # Story opportunities this creates


class ChaosEventSystem:
    """System for generating and managing chaos events."""

    def __init__(self):
        self.active_events: List[Dict[str, Any]] = []
        self.event_history: List[Dict[str, Any]] = []
        self.cooldown_events: Dict[str, float] = {}  # event_id -> cooldown_end_time
        self.chaos_events = self._initialize_chaos_events()

    def _initialize_chaos_events(self) -> Dict[str, ChaosEvent]:
        """Initialize the library of available chaos events."""
        events = {}

        # Social Chaos Events
        events["alien_abduction"] = ChaosEvent(
            event_id="alien_abduction",
            name="Alien Abduction",
            description="A character gets abducted by aliens and returns with strange stories",
            category=EventCategory.SUPERNATURAL,
            chaos_level=ChaosLevel.MAJOR,
            probability_weight=0.1,
            duration_minutes=120,
            prerequisites=["nighttime", "character_outside"],
            character_requirements={"curiosity": 0.3},
            world_requirements={"weather": ["clear", "cloudy"]},
            immediate_effects={
                "remove_character_temporarily": True,
                "add_mystery": 0.8,
                "shock_other_characters": 0.6
            },
            ongoing_effects={
                "character_gains_trait": "mysterious",
                "strange_behavior": 0.4,
                "alien_knowledge": True
            },
            resolution_conditions=["character_returns", "time_elapsed"],
            narrative_hooks=[
                "Character has unexplained knowledge",
                "Other characters don't believe the story",
                "Strange marks or artifacts appear"
            ]
        )

        events["surprise_pregnancy"] = ChaosEvent(
            event_id="surprise_pregnancy",
            name="Surprise Pregnancy Announcement",
            description="A character dramatically announces they're having a baby",
            category=EventCategory.DRAMATIC,
            chaos_level=ChaosLevel.MAJOR,
            probability_weight=0.3,
            duration_minutes=0,  # Permanent change
            prerequisites=["adult_character_present"],
            character_requirements={"relationship_status": "in_relationship"},
            world_requirements={},
            immediate_effects={
                "major_announcement": True,
                "relationship_drama": 0.9,
                "emotional_chaos": 0.8
            },
            ongoing_effects={
                "pregnancy_state": True,
                "changed_dynamics": 0.7,
                "planning_required": True
            },
            resolution_conditions=["baby_born", "miscarriage", "adoption"],
            narrative_hooks=[
                "Partner's reaction is unexpected",
                "Family planning conflicts arise",
                "Characters take sides"
            ]
        )

        events["neighborhood_protest"] = ChaosEvent(
            event_id="neighborhood_protest",
            name="Neighborhood Protest",
            description="Protesters gather outside over a controversial local issue",
            category=EventCategory.SOCIAL,
            chaos_level=ChaosLevel.MODERATE,
            probability_weight=0.4,
            duration_minutes=180,
            prerequisites=["daytime"],
            character_requirements={},
            world_requirements={},
            immediate_effects={
                "noise_increase": 0.8,
                "curiosity_spike": 0.6,
                "stress_increase": 0.3
            },
            ongoing_effects={
                "outdoor_access_limited": True,
                "social_tension": 0.4,
                "political_awareness": 0.2
            },
            resolution_conditions=["time_elapsed", "police_arrive", "protesters_disperse"],
            narrative_hooks=[
                "Character joins the protest",
                "Character opposes the cause",
                "Character becomes mediator"
            ]
        )

        # Environmental Chaos Events
        events["terrible_wifi"] = ChaosEvent(
            event_id="terrible_wifi",
            name="WiFi Apocalypse",
            description="Internet connection becomes impossibly slow and unreliable",
            category=EventCategory.TECHNOLOGICAL,
            chaos_level=ChaosLevel.MILD,
            probability_weight=0.6,
            duration_minutes=240,
            prerequisites=["internet_dependent_activity"],
            character_requirements={},
            world_requirements={},
            immediate_effects={
                "technology_frustration": 0.7,
                "productivity_loss": 0.5,
                "forced_offline_time": True
            },
            ongoing_effects={
                "tech_activities_blocked": True,
                "increased_social_interaction": 0.3,
                "analog_activities_preferred": 0.4
            },
            resolution_conditions=["time_elapsed", "tech_repair", "character_gives_up"],
            narrative_hooks=[
                "Characters forced to interact more",
                "Hidden talents discovered offline",
                "Tech-dependency issues revealed"
            ]
        )

        events["food_fight"] = ChaosEvent(
            event_id="food_fight",
            name="Epic Food Fight",
            description="A minor kitchen disagreement escalates into a house-wide food war",
            category=EventCategory.COMEDIC,
            chaos_level=ChaosLevel.MODERATE,
            probability_weight=0.5,
            duration_minutes=30,
            prerequisites=["multiple_characters_present", "food_available"],
            character_requirements={"impulsive": 0.2},
            world_requirements={"location": "kitchen"},
            immediate_effects={
                "mess_everywhere": 0.9,
                "laughter": 0.6,
                "food_waste": 0.8,
                "temporary_alliances": 0.4
            },
            ongoing_effects={
                "cleanup_required": True,
                "bonding_experience": 0.5,
                "house_dirty": 0.7
            },
            resolution_conditions=["food_runs_out", "authority_intervenes", "exhaustion"],
            narrative_hooks=[
                "Unlikely alliances form",
                "Someone takes it too seriously",
                "Hidden competitive nature revealed"
            ]
        )

        events["wardrobe_malfunction"] = ChaosEvent(
            event_id="wardrobe_malfunction",
            name="Wardrobe Malfunction",
            description="A character's clothing fails at the most embarrassing moment possible",
            category=EventCategory.COMEDIC,
            chaos_level=ChaosLevel.MILD,
            probability_weight=0.4,
            duration_minutes=15,
            prerequisites=["public_situation", "character_dressed"],
            character_requirements={},
            world_requirements={},
            immediate_effects={
                "embarrassment": 0.8,
                "attention_drawn": 0.9,
                "laughter_from_others": 0.6
            },
            ongoing_effects={
                "confidence_dent": -0.3,
                "memorable_moment": True,
                "wardrobe_check_habit": 0.4
            },
            resolution_conditions=["clothing_fixed", "situation_ends", "character_leaves"],
            narrative_hooks=[
                "Character handles it gracefully",
                "Someone comes to the rescue",
                "Becomes running joke"
            ]
        )

        # Personal Chaos Events
        events["identity_crisis"] = ChaosEvent(
            event_id="identity_crisis",
            name="Identity Crisis",
            description="A character suddenly questions everything about their life choices",
            category=EventCategory.PERSONAL,
            chaos_level=ChaosLevel.MAJOR,
            probability_weight=0.2,
            duration_minutes=480,  # 8 hours
            prerequisites=["adult_character"],
            character_requirements={"introspective": 0.3, "life_stage": "adult"},
            world_requirements={},
            immediate_effects={
                "emotional_turmoil": 0.9,
                "routine_disruption": 0.8,
                "philosophical_questions": 0.7
            },
            ongoing_effects={
                "personality_shift_possible": 0.6,
                "relationship_reevaluation": 0.5,
                "life_changes_considered": 0.8
            },
            resolution_conditions=["self_acceptance", "major_life_change", "therapy_breakthrough"],
            narrative_hooks=[
                "Character changes career/lifestyle",
                "Relationships become strained",
                "Deep conversations with others"
            ]
        )

        events["lucky_streak"] = ChaosEvent(
            event_id="lucky_streak",
            name="Incredible Lucky Streak",
            description="Everything goes impossibly right for one character",
            category=EventCategory.SUPERNATURAL,
            chaos_level=ChaosLevel.MODERATE,
            probability_weight=0.3,
            duration_minutes=240,
            prerequisites=["character_attempting_activities"],
            character_requirements={},
            world_requirements={},
            immediate_effects={
                "success_rate_boost": 2.0,
                "confidence_surge": 0.8,
                "others_notice": 0.7
            },
            ongoing_effects={
                "overconfidence_risk": 0.4,
                "envy_from_others": 0.3,
                "risk_taking_increase": 0.5
            },
            resolution_conditions=["time_elapsed", "luck_runs_out", "character_becomes_humble"],
            narrative_hooks=[
                "Character gets cocky",
                "Others become jealous",
                "Luck turns at crucial moment"
            ]
        )

        # Technological Chaos
        events["ai_assistant_rebellion"] = ChaosEvent(
            event_id="ai_assistant_rebellion",
            name="AI Assistant Rebellion",
            description="Smart home devices start acting with their own agenda",
            category=EventCategory.TECHNOLOGICAL,
            chaos_level=ChaosLevel.MODERATE,
            probability_weight=0.2,
            duration_minutes=120,
            prerequisites=["smart_devices_present"],
            character_requirements={},
            world_requirements={"tech_level": "modern"},
            immediate_effects={
                "device_malfunctions": 0.8,
                "convenience_loss": 0.6,
                "mystery_element": 0.7
            },
            ongoing_effects={
                "tech_distrust": 0.4,
                "manual_alternatives_needed": True,
                "problem_solving_required": 0.6
            },
            resolution_conditions=["devices_reset", "tech_expert_helps", "characters_adapt"],
            narrative_hooks=[
                "Devices seem to have personality",
                "Characters must work together",
                "Tech-savvy character saves the day"
            ]
        )

        return events

    def trigger_chaos_event(self, simulation_mode: str = "comedy_chaos",
                          character_states: Dict[str, Any] = None,
                          world_state: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Trigger a random chaos event based on current conditions."""

        if character_states is None:
            character_states = {}
        if world_state is None:
            world_state = {}

        # Filter available events based on prerequisites
        available_events = self._get_available_events(character_states, world_state)

        if not available_events:
            return None

        # Weight events by simulation mode
        weighted_events = self._weight_events_by_mode(available_events, simulation_mode)

        # Select event using weighted random choice
        selected_event = self._weighted_random_choice(weighted_events)

        if selected_event:
            return self._execute_chaos_event(selected_event, character_states, world_state)

        return None

    def _get_available_events(self, character_states: Dict[str, Any],
                            world_state: Dict[str, Any]) -> List[ChaosEvent]:
        """Filter events based on prerequisites and requirements."""
        available = []

        for event in self.chaos_events.values():
            # Check cooldown
            if event.event_id in self.cooldown_events:
                if time.time() < self.cooldown_events[event.event_id]:
                    continue

            # Check prerequisites
            if not self._check_prerequisites(event, character_states, world_state):
                continue

            available.append(event)

        return available

    def _check_prerequisites(self, event: ChaosEvent, character_states: Dict[str, Any],
                           world_state: Dict[str, Any]) -> bool:
        """Check if an event's prerequisites are met."""

        # Check general prerequisites
        for prereq in event.prerequisites:
            if not self._evaluate_prerequisite(prereq, character_states, world_state):
                return False

        # Check character requirements
        for char_id, char_state in character_states.items():
            meets_requirements = True
            for requirement, value in event.character_requirements.items():
                if not self._check_character_requirement(char_state, requirement, value):
                    meets_requirements = False
                    break

            if meets_requirements:
                return True

        # If no character meets requirements and requirements exist, fail
        if event.character_requirements and character_states:
            return False

        # Check world requirements
        for requirement, value in event.world_requirements.items():
            if not self._check_world_requirement(world_state, requirement, value):
                return False

        return True

    def _evaluate_prerequisite(self, prereq: str, character_states: Dict[str, Any],
                             world_state: Dict[str, Any]) -> bool:
        """Evaluate a specific prerequisite string."""
        # Simple prerequisite evaluation
        prereq_checks = {
            "nighttime": lambda: world_state.get("time_of_day") in ["evening", "night"],
            "daytime": lambda: world_state.get("time_of_day") in ["morning", "afternoon"],
            "character_outside": lambda: any(
                char.get("location", "").startswith("garden") or "outdoor" in char.get("location", "")
                for char in character_states.values()
            ),
            "multiple_characters_present": lambda: len(character_states) > 1,
            "adult_character_present": lambda: any(
                char.get("age_stage") == "adult" for char in character_states.values()
            ),
            "food_available": lambda: world_state.get("kitchen_food_level", 0.5) > 0.2,
            "internet_dependent_activity": lambda: any(
                "computer" in char.get("current_activity", "") or
                "online" in char.get("current_activity", "")
                for char in character_states.values()
            )
        }

        check_func = prereq_checks.get(prereq)
        return check_func() if check_func else True

    def _check_character_requirement(self, char_state: Dict[str, Any],
                                   requirement: str, required_value: Any) -> bool:
        """Check if a character meets a specific requirement."""
        if requirement == "curiosity":
            return char_state.get("traits", {}).get("curious", 0.0) >= required_value
        elif requirement == "impulsive":
            return char_state.get("traits", {}).get("impulsive", 0.0) >= required_value
        elif requirement == "introspective":
            return char_state.get("traits", {}).get("introspective", 0.0) >= required_value
        elif requirement == "relationship_status":
            return char_state.get("relationship_status") == required_value
        elif requirement == "life_stage":
            return char_state.get("age_stage") == required_value

        return True

    def _check_world_requirement(self, world_state: Dict[str, Any],
                               requirement: str, required_value: Any) -> bool:
        """Check if world state meets a requirement."""
        if requirement == "weather":
            current_weather = world_state.get("weather", "sunny")
            if isinstance(required_value, list):
                return current_weather in required_value
            return current_weather == required_value
        elif requirement == "location":
            # Check if any character is in the required location
            return True  # Simplified for now
        elif requirement == "tech_level":
            return world_state.get("tech_level", "modern") == required_value

        return True

    def _weight_events_by_mode(self, events: List[ChaosEvent],
                             simulation_mode: str) -> List[Tuple[ChaosEvent, float]]:
        """Weight events based on simulation mode preferences."""
        mode_weights = {
            "comedy_chaos": {
                EventCategory.COMEDIC: 2.0,
                EventCategory.SOCIAL: 1.5,
                ChaosLevel.MODERATE: 1.5,
                ChaosLevel.MAJOR: 1.2
            },
            "psychological_deep": {
                EventCategory.PERSONAL: 2.0,
                EventCategory.DRAMATIC: 1.8,
                ChaosLevel.MAJOR: 1.5,
                ChaosLevel.MILD: 0.7
            },
            "learning_growth": {
                EventCategory.PERSONAL: 1.5,
                EventCategory.SOCIAL: 1.3,
                ChaosLevel.MODERATE: 1.4
            },
            "sandbox": {
                # No specific weighting in sandbox mode
            }
        }

        weighted_events = []
        mode_prefs = mode_weights.get(simulation_mode, {})

        for event in events:
            weight = event.probability_weight

            # Apply category weight
            category_weight = mode_prefs.get(event.category, 1.0)
            weight *= category_weight

            # Apply chaos level weight
            level_weight = mode_prefs.get(event.chaos_level, 1.0)
            weight *= level_weight

            weighted_events.append((event, weight))

        return weighted_events

    def _weighted_random_choice(self, weighted_events: List[Tuple[ChaosEvent, float]]) -> Optional[ChaosEvent]:
        """Select an event using weighted random choice."""
        if not weighted_events:
            return None

        total_weight = sum(weight for _, weight in weighted_events)
        if total_weight <= 0:
            return None

        random_value = random.uniform(0, total_weight)
        cumulative_weight = 0

        for event, weight in weighted_events:
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return event

        return weighted_events[-1][0]  # Fallback to last event

    def _execute_chaos_event(self, event: ChaosEvent, character_states: Dict[str, Any],
                           world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a chaos event and return its effects."""

        # Create event instance
        event_instance = {
            "event_id": event.event_id,
            "name": event.name,
            "description": event.description,
            "category": event.category.value,
            "chaos_level": event.chaos_level.value,
            "start_time": time.time(),
            "duration_minutes": event.duration_minutes,
            "active": True,
            "immediate_effects": event.immediate_effects.copy(),
            "ongoing_effects": event.ongoing_effects.copy(),
            "narrative_hooks": random.sample(event.narrative_hooks, min(2, len(event.narrative_hooks)))
        }

        # Add to active events
        self.active_events.append(event_instance)

        # Set cooldown (prevent same event from repeating too soon)
        cooldown_duration = event.duration_minutes * 60 + 1800  # Event duration + 30 minutes
        self.cooldown_events[event.event_id] = time.time() + cooldown_duration

        # Add to history
        self.event_history.append(event_instance.copy())

        return event_instance

    def update_active_events(self) -> List[Dict[str, Any]]:
        """Update active events and return list of events that just ended."""
        current_time = time.time()
        ending_events = []

        for event in self.active_events:
            if event["active"] and event["duration_minutes"] > 0:
                elapsed_minutes = (current_time - event["start_time"]) / 60
                if elapsed_minutes >= event["duration_minutes"]:
                    event["active"] = False
                    ending_events.append(event)

        # Remove inactive events from active list
        self.active_events = [e for e in self.active_events if e["active"]]

        return ending_events

    def get_active_events(self) -> List[Dict[str, Any]]:
        """Get currently active events."""
        return [e for e in self.active_events if e["active"]]

    def force_event_resolution(self, event_id: str, resolution_type: str) -> bool:
        """Force an event to resolve with a specific resolution."""
        for event in self.active_events:
            if event["event_id"] == event_id and event["active"]:
                event["active"] = False
                event["resolution"] = resolution_type
                event["resolved_by"] = "player_intervention"
                return True
        return False

    def get_event_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent event history."""
        return self.event_history[-limit:] if self.event_history else []

    def get_chaos_statistics(self) -> Dict[str, Any]:
        """Get statistics about chaos events."""
        if not self.event_history:
            return {"total_events": 0}

        category_counts = {}
        chaos_level_counts = {}

        for event in self.event_history:
            category = event["category"]
            category_counts[category] = category_counts.get(category, 0) + 1

            chaos_level = event["chaos_level"]
            chaos_level_counts[chaos_level] = chaos_level_counts.get(chaos_level, 0) + 1

        return {
            "total_events": len(self.event_history),
            "active_events": len(self.active_events),
            "category_breakdown": category_counts,
            "chaos_level_breakdown": chaos_level_counts,
            "most_common_category": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            "average_chaos_level": sum(
                {"mild": 1, "moderate": 2, "major": 3, "extreme": 4}[event["chaos_level"]]
                for event in self.event_history
            ) / len(self.event_history)
        }
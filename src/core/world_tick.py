"""
World Tick System - Core game loop that processes character actions.

This module handles the main game loop, processing each character's
AI-driven decisions and updating the world state accordingly.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .ai_personality_engine import WorldContext, ActionDecision, AIPersonalityEngine
from .memory_system import MemorySystem


class ActionType(Enum):
    """Types of actions characters can perform."""
    IDLE = "idle"
    MOVE = "move"
    TALK = "talk"
    INTERACT = "interact"
    WORK = "work"
    EAT = "eat"
    SLEEP = "sleep"
    SOCIALIZE = "socialize"
    PRANK = "prank"
    ARGUE = "argue"
    FLIRT = "flirt"
    HIDE = "hide"
    EXPLORE = "explore"


@dataclass
class WorldState:
    """Current state of the game world."""
    current_time: float
    time_of_day: str
    weather: str
    active_events: List[Dict[str, Any]]
    locations: Dict[str, Dict[str, Any]]
    global_mood: str


@dataclass
class CharacterState:
    """Current state of a character in the world."""
    character_id: str
    location: str
    energy: float
    mood: str
    current_action: Optional[str]
    action_target: Optional[str]
    last_action_time: float
    needs: Dict[str, float]  # hunger, social, fun, hygiene, etc.


class ActionProcessor:
    """Processes character actions and updates world state."""

    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system

    async def process_action(self, character_state: CharacterState,
                           decision: ActionDecision,
                           world_state: WorldState,
                           all_characters: Dict[str, CharacterState]) -> List[str]:
        """Process a character's action and return list of events generated."""

        events = []
        char_id = character_state.character_id

        try:
            if decision.action_type == ActionType.IDLE.value:
                events.append(await self._process_idle(character_state, decision))

            elif decision.action_type == ActionType.MOVE.value:
                events.append(await self._process_move(character_state, decision, world_state))

            elif decision.action_type == ActionType.TALK.value:
                events.extend(await self._process_talk(
                    character_state, decision, all_characters
                ))

            elif decision.action_type == ActionType.SOCIALIZE.value:
                events.extend(await self._process_socialize(
                    character_state, decision, all_characters
                ))

            elif decision.action_type == ActionType.PRANK.value:
                events.extend(await self._process_prank(
                    character_state, decision, all_characters
                ))

            elif decision.action_type == ActionType.WORK.value:
                events.append(await self._process_work(character_state, decision))

            else:
                # Default processing for unknown actions
                events.append(f"{char_id} {decision.action_type}")

            # Update character state
            character_state.current_action = decision.action_type
            character_state.action_target = decision.target
            character_state.last_action_time = time.time()
            character_state.mood = decision.emotional_state

            # Store memory of the action
            if events:
                participants = [char_id]
                if decision.target:
                    participants.append(decision.target)

                await self.memory_system.store_event(
                    char_id,
                    events[0],  # Store the main event
                    self._calculate_emotional_weight(decision),
                    participants,
                    character_state.location
                )

        except Exception as e:
            print(f"Error processing action for {char_id}: {e}")
            events.append(f"{char_id} looks confused and does nothing")

        return [event for event in events if event]  # Filter out None events

    async def _process_idle(self, character_state: CharacterState,
                          decision: ActionDecision) -> str:
        """Process idle action."""
        if decision.dialogue:
            return f"{character_state.character_id} says: '{decision.dialogue}'"
        return f"{character_state.character_id} {decision.internal_monologue.lower()}"

    async def _process_move(self, character_state: CharacterState,
                          decision: ActionDecision,
                          world_state: WorldState) -> Optional[str]:
        """Process movement to a new location."""
        target_location = decision.target

        if not target_location or target_location not in world_state.locations:
            return f"{character_state.character_id} wanders around aimlessly"

        old_location = character_state.location
        character_state.location = target_location

        return f"{character_state.character_id} moves from {old_location} to {target_location}"

    async def _process_talk(self, character_state: CharacterState,
                          decision: ActionDecision,
                          all_characters: Dict[str, CharacterState]) -> List[str]:
        """Process talking to another character."""
        events = []
        target = decision.target

        if not target or target not in all_characters:
            return [f"{character_state.character_id} talks to themselves"]

        target_character = all_characters[target]

        # Check if characters are in same location
        if character_state.location != target_character.location:
            return [f"{character_state.character_id} tries to talk to {target}, but they're not nearby"]

        # Create conversation event
        if decision.dialogue:
            events.append(f"{character_state.character_id} says to {target}: '{decision.dialogue}'")

            # Store memory for both participants
            await self.memory_system.store_event(
                target,
                f"{character_state.character_id} said: '{decision.dialogue}'",
                0.2,  # Mild emotional weight for being talked to
                [character_state.character_id, target],
                character_state.location
            )

        return events

    async def _process_socialize(self, character_state: CharacterState,
                               decision: ActionDecision,
                               all_characters: Dict[str, CharacterState]) -> List[str]:
        """Process socializing with nearby characters."""
        events = []
        char_id = character_state.character_id

        # Find characters in same location
        nearby_characters = [
            other_id for other_id, other_char in all_characters.items()
            if other_char.location == character_state.location and other_id != char_id
        ]

        if not nearby_characters:
            return [f"{char_id} tries to socialize but finds no one around"]

        # Create social interaction
        if len(nearby_characters) == 1:
            target = nearby_characters[0]
            events.append(f"{char_id} socializes with {target}")
        else:
            events.append(f"{char_id} socializes with {', '.join(nearby_characters)}")

        # Boost social need
        character_state.needs['social'] = min(1.0, character_state.needs.get('social', 0.5) + 0.2)

        return events

    async def _process_prank(self, character_state: CharacterState,
                           decision: ActionDecision,
                           all_characters: Dict[str, CharacterState]) -> List[str]:
        """Process prank action."""
        events = []
        target = decision.target
        char_id = character_state.character_id

        if not target or target not in all_characters:
            return [f"{char_id} sets up a prank but no one falls for it"]

        target_character = all_characters[target]

        # Check if characters are in same location
        if character_state.location != target_character.location:
            return [f"{char_id} tries to prank {target}, but they're not nearby"]

        # Execute prank
        prank_descriptions = [
            f"{char_id} puts plastic wrap over the toilet seat, and {target} falls for it!",
            f"{char_id} hides {target}'s belongings as a prank",
            f"{char_id} pretends to be a ghost to scare {target}",
            f"{char_id} puts salt in {target}'s coffee",
            f"{char_id} changes all of {target}'s clocks to different times"
        ]

        import random
        prank_event = random.choice(prank_descriptions)
        events.append(prank_event)

        # Store memory for target (negative emotional weight)
        await self.memory_system.store_event(
            target,
            f"{char_id} pranked me: {prank_event}",
            -0.3,  # Negative emotional weight
            [char_id, target],
            character_state.location
        )

        return events

    async def _process_work(self, character_state: CharacterState,
                          decision: ActionDecision) -> str:
        """Process work action."""
        work_activities = [
            "works on their computer",
            "organizes their belongings",
            "reads a book",
            "practices a skill",
            "cleans their space"
        ]

        import random
        activity = random.choice(work_activities)

        # Reduce energy, increase skill (hypothetically)
        character_state.energy = max(0.0, character_state.energy - 0.1)

        return f"{character_state.character_id} {activity}"

    def _calculate_emotional_weight(self, decision: ActionDecision) -> float:
        """Calculate emotional weight of an action for memory storage."""
        base_weight = 0.1

        # Increase weight for certain emotions
        emotion_weights = {
            'angry': 0.8,
            'love': 0.9,
            'fear': 0.7,
            'joy': 0.6,
            'sadness': 0.7,
            'surprise': 0.5,
            'disgust': 0.4
        }

        emotion = decision.emotional_state.lower()
        weight_multiplier = emotion_weights.get(emotion, 1.0)

        return base_weight * weight_multiplier * decision.confidence


class WorldTicker:
    """Main world tick system that processes the game loop."""

    def __init__(self, ai_engine: AIPersonalityEngine, memory_system: MemorySystem):
        self.ai_engine = ai_engine
        self.memory_system = memory_system
        self.action_processor = ActionProcessor(memory_system)

        self.world_state = WorldState(
            current_time=time.time(),
            time_of_day="morning",
            weather="sunny",
            active_events=[],
            locations={
                "living_room": {"description": "A cozy living room with a TV"},
                "kitchen": {"description": "A modern kitchen with appliances"},
                "bedroom": {"description": "A comfortable bedroom"},
                "bathroom": {"description": "A clean bathroom"},
                "garden": {"description": "A small garden outside"}
            },
            global_mood="neutral"
        )

        self.character_states: Dict[str, CharacterState] = {}
        self.tick_count = 0
        self.is_running = False

    def add_character(self, character_id: str, initial_location: str = "living_room"):
        """Add a character to the world."""
        self.character_states[character_id] = CharacterState(
            character_id=character_id,
            location=initial_location,
            energy=1.0,
            mood="neutral",
            current_action=None,
            action_target=None,
            last_action_time=time.time(),
            needs={
                'hunger': 0.5,
                'social': 0.5,
                'fun': 0.5,
                'hygiene': 0.8,
                'energy': 1.0
            }
        )

    async def tick(self) -> List[str]:
        """Process one world tick - returns list of events that occurred."""
        self.tick_count += 1
        events = []

        # Update world state
        self._update_world_state()

        # Process each character
        for char_id, char_state in self.character_states.items():
            try:
                # Build context for this character
                context = self._build_world_context(char_state)

                # Get AI decision
                character = self.ai_engine.get_character(char_id)
                if character:
                    decision = await character.process_world_tick(context)

                    # Process the action
                    char_events = await self.action_processor.process_action(
                        char_state, decision, self.world_state, self.character_states
                    )

                    events.extend(char_events)

                # Update character needs over time
                self._update_character_needs(char_state)

            except Exception as e:
                print(f"Error processing character {char_id} in tick {self.tick_count}: {e}")
                events.append(f"{char_id} seems distracted and confused")

        return events

    def _update_world_state(self):
        """Update global world state."""
        current_time = time.time()
        self.world_state.current_time = current_time

        # Simple time of day calculation
        hour = int((current_time / 3600) % 24)
        if 6 <= hour < 12:
            self.world_state.time_of_day = "morning"
        elif 12 <= hour < 18:
            self.world_state.time_of_day = "afternoon"
        elif 18 <= hour < 22:
            self.world_state.time_of_day = "evening"
        else:
            self.world_state.time_of_day = "night"

    def _build_world_context(self, character_state: CharacterState) -> WorldContext:
        """Build world context for a specific character."""
        # Find characters in same location
        nearby_characters = [
            other_id for other_id, other_char in self.character_states.items()
            if (other_char.location == character_state.location and
                other_id != character_state.character_id)
        ]

        # Get recent events (simplified)
        recent_events = self.world_state.active_events[-5:] if self.world_state.active_events else []

        return WorldContext(
            current_location=character_state.location,
            nearby_characters=nearby_characters,
            recent_events=recent_events,
            time_of_day=self.world_state.time_of_day,
            environment_state={
                'weather': self.world_state.weather,
                'global_mood': self.world_state.global_mood,
                'character_needs': character_state.needs
            }
        )

    def _update_character_needs(self, character_state: CharacterState):
        """Update character needs over time."""
        # Needs decay slowly over time
        character_state.needs['hunger'] = max(0.0, character_state.needs.get('hunger', 0.5) - 0.01)
        character_state.needs['energy'] = max(0.0, character_state.needs.get('energy', 1.0) - 0.005)
        character_state.needs['social'] = max(0.0, character_state.needs.get('social', 0.5) - 0.005)
        character_state.needs['fun'] = max(0.0, character_state.needs.get('fun', 0.5) - 0.01)

        # Hygiene decreases more slowly
        character_state.needs['hygiene'] = max(0.0, character_state.needs.get('hygiene', 0.8) - 0.002)

    async def run_continuous(self, tick_interval: float = 5.0):
        """Run the world tick continuously."""
        self.is_running = True

        while self.is_running:
            events = await self.tick()

            # Print events for debugging
            if events:
                print(f"Tick {self.tick_count}:")
                for event in events:
                    print(f"  - {event}")
                print()

            await asyncio.sleep(tick_interval)

    def stop(self):
        """Stop the continuous world tick."""
        self.is_running = False

    def get_world_summary(self) -> Dict[str, Any]:
        """Get a summary of the current world state."""
        return {
            'tick_count': self.tick_count,
            'time_of_day': self.world_state.time_of_day,
            'weather': self.world_state.weather,
            'character_count': len(self.character_states),
            'characters': {
                char_id: {
                    'location': state.location,
                    'mood': state.mood,
                    'energy': state.energy,
                    'current_action': state.current_action
                }
                for char_id, state in self.character_states.items()
            }
        }
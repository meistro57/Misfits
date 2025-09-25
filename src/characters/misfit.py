"""
Misfit Character - Individual character representation and behavior.

This module defines the main character class that combines AI personality,
memory, and physical representation in the game world.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..core.ai_personality_engine import PersonalityCore, PersonalityTraits, ActionDecision
from ..core.memory_system import MisfitMemory


class LifeStage(Enum):
    """Character life stages that affect behavior."""
    CHILD = "child"
    TEEN = "teen"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    ELDERLY = "elderly"


@dataclass
class PhysicalTraits:
    """Physical appearance and characteristics."""
    height: str
    build: str
    hair_color: str
    eye_color: str
    distinctive_features: List[str]
    clothing_style: str
    age_appearance: int


@dataclass
class Skills:
    """Character skills and abilities."""
    cooking: float = 0.5
    charisma: float = 0.5
    creativity: float = 0.5
    logic: float = 0.5
    fitness: float = 0.5
    handiness: float = 0.5
    mischief: float = 0.5
    programming: float = 0.5
    gardening: float = 0.5
    painting: float = 0.5

    def get_skill_dict(self) -> Dict[str, float]:
        """Return skills as dictionary."""
        return {
            'cooking': self.cooking,
            'charisma': self.charisma,
            'creativity': self.creativity,
            'logic': self.logic,
            'fitness': self.fitness,
            'handiness': self.handiness,
            'mischief': self.mischief,
            'programming': self.programming,
            'gardening': self.gardening,
            'painting': self.painting
        }

    def improve_skill(self, skill_name: str, amount: float = 0.1):
        """Improve a specific skill."""
        if hasattr(self, skill_name):
            current_value = getattr(self, skill_name)
            new_value = min(1.0, current_value + amount)
            setattr(self, skill_name, new_value)


@dataclass
class Aspirations:
    """Character goals and aspirations."""
    primary_aspiration: str
    secondary_aspirations: List[str]
    completed_aspirations: List[str]
    progress: Dict[str, float]


class MisfitCharacter:
    """Main character class combining AI, memory, and game mechanics."""

    def __init__(self, character_id: str, name: str, personality_traits: PersonalityTraits,
                 physical_traits: PhysicalTraits, life_stage: LifeStage = LifeStage.ADULT):
        self.character_id = character_id
        self.name = name
        self.personality_traits = personality_traits
        self.physical_traits = physical_traits
        self.life_stage = life_stage

        # AI and memory components (to be injected)
        self.personality_core: Optional[PersonalityCore] = None
        self.memory_system: Optional[MisfitMemory] = None

        # Game mechanics
        self.skills = Skills()
        self.aspirations = Aspirations(
            primary_aspiration="find_meaning",
            secondary_aspirations=["make_friends", "learn_skills"],
            completed_aspirations=[],
            progress={}
        )

        # Current state
        self.current_location = "living_room"
        self.energy = 1.0
        self.mood = "neutral"
        self.relationships: Dict[str, float] = {}  # character_id -> relationship strength

        # Quirks and preferences
        self.quirks = []
        self.favorite_activities = []
        self.dislikes = []
        self.fears = []

        # Character development
        self.character_arc_stage = "introduction"
        self.growth_points = 0
        self.life_events = []

    def set_ai_components(self, personality_core: PersonalityCore, memory_system: MisfitMemory):
        """Inject AI and memory components."""
        self.personality_core = personality_core
        self.memory_system = memory_system

    async def make_decision(self, world_context) -> ActionDecision:
        """Make a decision based on current world state."""
        if not self.personality_core:
            # Fallback behavior without AI
            return ActionDecision(
                action_type="idle",
                target=None,
                dialogue=None,
                internal_monologue="I don't know what to do...",
                emotional_state="confused",
                confidence=0.1
            )

        return await self.personality_core.process_world_tick(world_context)

    async def respond_to_dialogue(self, speaker: str, message: str) -> str:
        """Respond to dialogue from another character."""
        if not self.personality_core:
            return "I'm not sure what to say..."

        response = await self.personality_core.respond_to_dialogue(speaker, message)

        # Store memory of this conversation
        if self.memory_system:
            await self.memory_system.store_memory(
                f"{speaker} said: '{message}' and I responded: '{response}'",
                0.1,  # Mild emotional weight for normal conversation
                [self.character_id, speaker],
                self.current_location,
                ["conversation"]
            )

        return response

    def update_mood(self, new_mood: str, intensity: float = 1.0):
        """Update character's current mood."""
        # Simple mood system - could be expanded with mood persistence, etc.
        self.mood = new_mood

        # Store significant mood changes in memory
        if self.memory_system and intensity > 0.5:
            import asyncio
            asyncio.create_task(self.memory_system.store_memory(
                f"I became {new_mood}",
                intensity * 0.3,
                [self.character_id],
                self.current_location,
                ["mood_change"]
            ))

    def improve_skill_through_action(self, action_type: str):
        """Improve relevant skills based on performed action."""
        skill_mapping = {
            'cook': 'cooking',
            'socialize': 'charisma',
            'create': 'creativity',
            'study': 'logic',
            'exercise': 'fitness',
            'repair': 'handiness',
            'prank': 'mischief',
            'program': 'programming',
            'garden': 'gardening',
            'paint': 'painting'
        }

        skill_name = skill_mapping.get(action_type)
        if skill_name:
            self.skills.improve_skill(skill_name, 0.05)  # Small improvement per action

    def update_relationship(self, other_character_id: str, change: float):
        """Update relationship with another character."""
        current_relationship = self.relationships.get(other_character_id, 0.0)
        new_relationship = max(-1.0, min(1.0, current_relationship + change))
        self.relationships[other_character_id] = new_relationship

    def get_personality_summary(self) -> Dict[str, Any]:
        """Get a summary of the character's personality for display."""
        return {
            'name': self.name,
            'age_stage': self.life_stage.value,
            'base_traits': self.personality_traits.base_traits,
            'hidden_desires': self.personality_traits.hidden_desires,
            'dialogue_style': self.personality_traits.dialogue_style,
            'current_mood': self.mood,
            'appearance': {
                'height': self.physical_traits.height,
                'build': self.physical_traits.build,
                'hair_color': self.physical_traits.hair_color,
                'eye_color': self.physical_traits.eye_color,
                'clothing_style': self.physical_traits.clothing_style
            },
            'top_skills': self._get_top_skills(),
            'quirks': self.quirks,
            'relationships': len(self.relationships)
        }

    def _get_top_skills(self, limit: int = 3) -> List[Dict[str, float]]:
        """Get the character's top skills."""
        skill_dict = self.skills.get_skill_dict()
        sorted_skills = sorted(skill_dict.items(), key=lambda x: x[1], reverse=True)
        return [{'skill': skill, 'level': level} for skill, level in sorted_skills[:limit]]

    def get_character_status(self) -> Dict[str, Any]:
        """Get current character status for UI display."""
        return {
            'character_id': self.character_id,
            'name': self.name,
            'location': self.current_location,
            'mood': self.mood,
            'energy': self.energy,
            'primary_aspiration': self.aspirations.primary_aspiration,
            'skill_summary': self._get_top_skills(1),
            'relationship_count': len(self.relationships)
        }

    def experience_life_event(self, event_type: str, description: str,
                            emotional_impact: float, participants: List[str] = None):
        """Process a significant life event."""
        if participants is None:
            participants = [self.character_id]

        life_event = {
            'event_type': event_type,
            'description': description,
            'emotional_impact': emotional_impact,
            'participants': participants,
            'location': self.current_location,
            'character_arc_stage': self.character_arc_stage
        }

        self.life_events.append(life_event)

        # Grant growth points for significant events
        self.growth_points += abs(emotional_impact) * 10

        # Store in memory system
        if self.memory_system:
            import asyncio
            asyncio.create_task(self.memory_system.store_memory(
                description,
                emotional_impact,
                participants,
                self.current_location,
                [event_type, "life_event"]
            ))

        # Check for character arc progression
        self._check_character_arc_progression()

    def _check_character_arc_progression(self):
        """Check if character should progress to next arc stage."""
        if self.growth_points >= 100 and self.character_arc_stage == "introduction":
            self.character_arc_stage = "development"
        elif self.growth_points >= 300 and self.character_arc_stage == "development":
            self.character_arc_stage = "crisis"
        elif self.growth_points >= 500 and self.character_arc_stage == "crisis":
            self.character_arc_stage = "resolution"

    def add_quirk(self, quirk: str):
        """Add a personality quirk to the character."""
        if quirk not in self.quirks:
            self.quirks.append(quirk)

    def develop_fear(self, fear: str, trigger_event: str):
        """Develop a new fear based on an event."""
        if fear not in self.fears:
            self.fears.append(fear)

            # Store the traumatic memory
            if self.memory_system:
                import asyncio
                asyncio.create_task(self.memory_system.store_memory(
                    f"I developed a fear of {fear} because of: {trigger_event}",
                    -0.8,  # Strong negative emotional weight
                    [self.character_id],
                    self.current_location,
                    ["fear", "trauma"]
                ))

    def overcome_fear(self, fear: str, triumph_event: str):
        """Overcome an existing fear through a positive experience."""
        if fear in self.fears:
            self.fears.remove(fear)
            self.growth_points += 50  # Bonus growth for overcoming fears

            # Store the triumph memory
            if self.memory_system:
                import asyncio
                asyncio.create_task(self.memory_system.store_memory(
                    f"I overcame my fear of {fear} when: {triumph_event}",
                    0.8,  # Strong positive emotional weight
                    [self.character_id],
                    self.current_location,
                    ["triumph", "growth"]
                ))

    def get_legacy_data(self) -> Dict[str, Any]:
        """Get data that should persist across playthroughs."""
        return {
            'character_id': self.character_id,
            'name': self.name,
            'completed_aspirations': self.aspirations.completed_aspirations,
            'major_life_events': [event for event in self.life_events
                                if event['emotional_impact'] > 0.7],
            'final_relationships': {k: v for k, v in self.relationships.items()
                                  if abs(v) > 0.5},
            'personality_evolution': self.personality_traits.behavioral_weights,
            'reputation_score': sum(self.relationships.values()) / max(1, len(self.relationships))
        }

    def __str__(self) -> str:
        """String representation of the character."""
        return f"{self.name} ({self.character_id}) - {self.mood} at {self.current_location}"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"MisfitCharacter(id='{self.character_id}', name='{self.name}', "
                f"traits={self.personality_traits.base_traits}, "
                f"location='{self.current_location}', mood='{self.mood}')")
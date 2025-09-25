"""
AI Personality Engine - Core system for driving character behavior.

This module contains the central AI personality system that generates
dynamic character behavior, dialogue, and decision-making.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PersonalityTraits:
    """Character personality traits and behavioral weights."""
    base_traits: List[str]
    hidden_desires: List[str]
    behavioral_weights: Dict[str, float]
    dialogue_style: str


@dataclass
class WorldContext:
    """Current world state and context for AI decision making."""
    current_location: str
    nearby_characters: List[str]
    recent_events: List[Dict[str, Any]]
    time_of_day: str
    environment_state: Dict[str, Any]


@dataclass
class ActionDecision:
    """AI-generated action decision for a character."""
    action_type: str
    target: Optional[str]
    dialogue: Optional[str]
    internal_monologue: str
    emotional_state: str
    confidence: float


class LLMInterface(ABC):
    """Abstract interface for LLM communication."""

    @abstractmethod
    async def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate AI response based on prompt and context."""
        pass


class PersonalityCore:
    """Core personality system for individual characters."""

    def __init__(self, character_id: str, traits: PersonalityTraits,
                 llm_interface: LLMInterface):
        self.character_id = character_id
        self.traits = traits
        self.llm_interface = llm_interface
        self.memory_system = None  # Will be injected
        self.current_emotional_state = "neutral"
        self.recent_interactions = []

    def set_memory_system(self, memory_system):
        """Inject memory system dependency."""
        self.memory_system = memory_system

    async def process_world_tick(self, context: WorldContext) -> ActionDecision:
        """Process world state and generate character action."""

        # Gather relevant memories
        relevant_memories = []
        if self.memory_system:
            relevant_memories = self.memory_system.retrieve_relevant_memories(
                self.character_id, str(context)
            )

        # Build AI prompt
        prompt = self._build_decision_prompt(context, relevant_memories)

        # Generate AI response
        response = await self.llm_interface.generate_response(prompt, {
            'character_id': self.character_id,
            'traits': self.traits,
            'context': context
        })

        # Parse response into ActionDecision
        return self._parse_ai_response(response)

    async def respond_to_dialogue(self, speaker: str, message: str) -> str:
        """Generate character response to dialogue."""

        # Gather memories about the speaker
        speaker_memories = []
        if self.memory_system:
            speaker_memories = self.memory_system.query_memories(
                self.character_id, f"interactions with {speaker}"
            )

        # Build dialogue prompt
        prompt = self._build_dialogue_prompt(speaker, message, speaker_memories)

        # Generate response
        response = await self.llm_interface.generate_response(prompt, {
            'character_id': self.character_id,
            'traits': self.traits,
            'speaker': speaker,
            'message': message
        })

        return response

    def _build_decision_prompt(self, context: WorldContext,
                              memories: List[Dict]) -> str:
        """Build AI prompt for decision making."""

        prompt = f"""
You are {self.character_id}, a character with the following personality:
- Base traits: {', '.join(self.traits.base_traits)}
- Hidden desires: {', '.join(self.traits.hidden_desires)}
- Dialogue style: {self.traits.dialogue_style}
- Current emotional state: {self.current_emotional_state}

Current situation:
- Location: {context.current_location}
- Nearby characters: {', '.join(context.nearby_characters)}
- Time: {context.time_of_day}
- Recent events: {context.recent_events}

Relevant memories:
{self._format_memories(memories)}

What do you want to do next? Consider your personality traits, hidden desires,
and the current situation. Respond with your decision, including any dialogue
you want to say and your internal thoughts.

Format your response as:
ACTION: [what you want to do]
TARGET: [who or what you're targeting, if applicable]
DIALOGUE: [what you say out loud, if anything]
THOUGHTS: [your internal monologue]
EMOTION: [your current emotional state]
"""

        return prompt

    def _build_dialogue_prompt(self, speaker: str, message: str,
                              memories: List[Dict]) -> str:
        """Build AI prompt for dialogue response."""

        prompt = f"""
You are {self.character_id}, a character with the following personality:
- Base traits: {', '.join(self.traits.base_traits)}
- Hidden desires: {', '.join(self.traits.hidden_desires)}
- Dialogue style: {self.traits.dialogue_style}
- Current emotional state: {self.current_emotional_state}

{speaker} just said to you: "{message}"

Your memories of {speaker}:
{self._format_memories(memories)}

How do you respond? Stay true to your personality and relationship with {speaker}.
Only provide your spoken response, nothing else.
"""

        return prompt

    def _format_memories(self, memories: List[Dict]) -> str:
        """Format memories for inclusion in prompts."""
        if not memories:
            return "No significant memories."

        formatted = []
        for memory in memories[:5]:  # Limit to top 5 memories
            formatted.append(f"- {memory.get('event', 'Unknown event')}")

        return '\n'.join(formatted)

    def _parse_ai_response(self, response: str) -> ActionDecision:
        """Parse AI response into structured ActionDecision."""

        # Simple parsing - in production this would be more robust
        lines = response.strip().split('\n')

        action_type = "idle"
        target = None
        dialogue = None
        thoughts = "I'm thinking..."
        emotion = "neutral"

        for line in lines:
            line = line.strip()
            if line.startswith("ACTION:"):
                action_type = line[7:].strip()
            elif line.startswith("TARGET:"):
                target = line[7:].strip()
            elif line.startswith("DIALOGUE:"):
                dialogue = line[9:].strip()
            elif line.startswith("THOUGHTS:"):
                thoughts = line[9:].strip()
            elif line.startswith("EMOTION:"):
                emotion = line[8:].strip()

        return ActionDecision(
            action_type=action_type,
            target=target,
            dialogue=dialogue,
            internal_monologue=thoughts,
            emotional_state=emotion,
            confidence=0.8  # Default confidence
        )


class AIPersonalityEngine:
    """Main engine managing all character personalities."""

    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface
        self.characters: Dict[str, PersonalityCore] = {}
        self.memory_system = None

    def set_memory_system(self, memory_system):
        """Set memory system for all characters."""
        self.memory_system = memory_system
        for character in self.characters.values():
            character.set_memory_system(memory_system)

    def create_character(self, character_id: str, traits: PersonalityTraits) -> PersonalityCore:
        """Create a new character with given personality traits."""

        character = PersonalityCore(character_id, traits, self.llm_interface)
        if self.memory_system:
            character.set_memory_system(self.memory_system)

        self.characters[character_id] = character
        return character

    def get_character(self, character_id: str) -> Optional[PersonalityCore]:
        """Get character by ID."""
        return self.characters.get(character_id)

    async def process_all_characters(self, context: WorldContext) -> Dict[str, ActionDecision]:
        """Process world tick for all characters."""

        decisions = {}
        for char_id, character in self.characters.items():
            try:
                decision = await character.process_world_tick(context)
                decisions[char_id] = decision
            except Exception as e:
                print(f"Error processing character {char_id}: {e}")
                # Fallback behavior
                decisions[char_id] = ActionDecision(
                    action_type="idle",
                    target=None,
                    dialogue=None,
                    internal_monologue="I'm confused...",
                    emotional_state="confused",
                    confidence=0.1
                )

        return decisions
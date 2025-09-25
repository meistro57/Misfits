"""
Simulation Modes - Different gameplay modes that filter AI behavior.

This module defines the four core simulation modes that change how
AI personalities behave and what kinds of events can occur.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class ModeType(Enum):
    """Available simulation modes."""
    COMEDY_CHAOS = "comedy_chaos"
    PSYCHOLOGICAL_DEEP = "psychological_deep"
    LEARNING_GROWTH = "learning_growth"
    SANDBOX = "sandbox"


@dataclass
class ModeParameters:
    """Parameters that control simulation behavior."""
    chaos_frequency: float  # How often random events occur (0.0 - 1.0)
    exaggeration_factor: float  # How exaggerated personality traits become
    prank_likelihood: float  # Probability of prank actions
    gossip_spread_rate: float  # How quickly rumors spread
    emotional_volatility: float  # How quickly emotions change
    learning_rate: float  # How quickly characters adapt/grow
    realism_level: float  # How realistic vs absurd events are
    consequence_weight: float  # How much past actions affect future
    chaos_button_enabled: bool  # Whether chaos button is available
    intervention_allowed: bool  # Whether player interventions work


@dataclass
class SimulationMode:
    """Complete simulation mode configuration."""
    name: str
    description: str
    parameters: ModeParameters
    available_events: List[str]
    personality_modifiers: Dict[str, float]
    special_rules: List[str]


class ModeModifier(ABC):
    """Abstract base class for mode-specific behavior modifications."""

    @abstractmethod
    def modify_personality_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Modify personality behavioral weights based on simulation mode."""
        pass

    @abstractmethod
    def filter_available_actions(self, actions: List[str]) -> List[str]:
        """Filter which actions are available in this mode."""
        pass

    @abstractmethod
    def modify_dialogue_style(self, base_style: str, character_traits: List[str]) -> str:
        """Modify how characters speak in this mode."""
        pass

    @abstractmethod
    def calculate_event_probability(self, event_type: str, base_probability: float) -> float:
        """Modify probability of events occurring."""
        pass


class ComedyChaosModeModifier(ModeModifier):
    """Modifier for Comedy & Chaos simulation mode."""

    def __init__(self, parameters: ModeParameters):
        self.parameters = parameters

    def modify_personality_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Exaggerate personality traits for comedic effect."""
        modified = base_weights.copy()

        exaggeration = self.parameters.exaggeration_factor

        # Amplify existing traits
        for trait, weight in modified.items():
            if weight > 0.5:  # Amplify strong traits
                modified[trait] = min(1.0, weight * exaggeration)
            elif weight < 0.3:  # Suppress weak traits for clearer personalities
                modified[trait] = max(0.0, weight * 0.7)

        # Boost comedy-relevant traits
        comedy_traits = ['impulsive', 'dramatic', 'mischievous', 'eccentric']
        for trait in comedy_traits:
            if trait in modified:
                modified[trait] = min(1.0, modified[trait] + 0.2)

        return modified

    def filter_available_actions(self, actions: List[str]) -> List[str]:
        """Prioritize fun and chaotic actions."""
        priority_actions = ['prank', 'socialize', 'argue', 'flirt', 'explore']

        # Reorder to prioritize fun actions
        filtered = []
        for action in priority_actions:
            if action in actions:
                filtered.append(action)

        # Add remaining actions
        for action in actions:
            if action not in filtered:
                filtered.append(action)

        return filtered

    def modify_dialogue_style(self, base_style: str, character_traits: List[str]) -> str:
        """Make dialogue more exaggerated and comedic."""
        if "dramatic" in character_traits:
            return "over_the_top_dramatic"
        elif "sarcastic" in character_traits:
            return "extremely_sarcastic"
        elif "cheerful" in character_traits:
            return "hyperactive_positive"
        else:
            return f"exaggerated_{base_style}"

    def calculate_event_probability(self, event_type: str, base_probability: float) -> float:
        """Increase probability of chaotic/funny events."""
        chaos_events = ['prank_backfire', 'mistaken_identity', 'food_fight', 'wardrobe_malfunction']

        if event_type in chaos_events:
            return min(1.0, base_probability * 2.0 * self.parameters.chaos_frequency)
        elif event_type.startswith('chaos_'):
            return min(1.0, base_probability * self.parameters.chaos_frequency)

        return base_probability


class PsychologicalDeepModeModifier(ModeModifier):
    """Modifier for Psychological & Deep simulation mode."""

    def __init__(self, parameters: ModeParameters):
        self.parameters = parameters

    def modify_personality_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Make personality traits more nuanced and realistic."""
        modified = base_weights.copy()

        # Add complexity by introducing contradictions
        if 'confident' in modified and modified['confident'] > 0.7:
            # Highly confident characters get hidden insecurities
            modified['insecure'] = modified.get('insecure', 0.0) + 0.2

        if 'social' in modified and modified['social'] > 0.8:
            # Very social characters sometimes need alone time
            modified['introverted_moments'] = 0.3

        # Reduce exaggeration for more realistic behavior
        for trait, weight in modified.items():
            if weight > 0.9:
                modified[trait] = 0.85  # Cap extreme traits

        return modified

    def filter_available_actions(self, actions: List[str]) -> List[str]:
        """Prioritize meaningful, relationship-building actions."""
        meaningful_actions = ['talk', 'socialize', 'work', 'reflect', 'help']

        filtered = []
        for action in meaningful_actions:
            if action in actions:
                filtered.append(action)

        # Deprioritize shallow actions
        shallow_actions = ['prank', 'show_off']
        for action in actions:
            if action not in filtered and action not in shallow_actions:
                filtered.append(action)

        # Add shallow actions at the end
        for action in shallow_actions:
            if action in actions:
                filtered.append(action)

        return filtered

    def modify_dialogue_style(self, base_style: str, character_traits: List[str]) -> str:
        """Make dialogue more thoughtful and emotionally aware."""
        if "philosophical" in character_traits:
            return "deep_thoughtful"
        elif "empathetic" in character_traits:
            return "emotionally_aware"
        elif "analytical" in character_traits:
            return "introspective"
        else:
            return f"realistic_{base_style}"

    def calculate_event_probability(self, event_type: str, base_probability: float) -> float:
        """Increase probability of meaningful, consequence-heavy events."""
        deep_events = ['relationship_conflict', 'personal_crisis', 'moral_dilemma', 'breakthrough_moment']

        if event_type in deep_events:
            return min(1.0, base_probability * 1.5 * self.parameters.consequence_weight)
        elif event_type.startswith('shallow_'):
            return base_probability * 0.3

        return base_probability


class LearningGrowthModeModifier(ModeModifier):
    """Modifier for Learning & Growth simulation mode."""

    def __init__(self, parameters: ModeParameters):
        self.parameters = parameters

    def modify_personality_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Allow personality traits to evolve based on experiences."""
        modified = base_weights.copy()

        # Add growth-oriented traits
        growth_traits = ['curious', 'adaptive', 'reflective', 'open_minded']
        for trait in growth_traits:
            if trait in modified:
                modified[trait] = min(1.0, modified[trait] + 0.1)
            else:
                modified[trait] = 0.2

        return modified

    def filter_available_actions(self, actions: List[str]) -> List[str]:
        """Prioritize learning and skill-building actions."""
        learning_actions = ['learn', 'practice', 'experiment', 'teach', 'ask_questions']

        filtered = []
        for action in learning_actions:
            if action in actions:
                filtered.append(action)

        for action in actions:
            if action not in filtered:
                filtered.append(action)

        return filtered

    def modify_dialogue_style(self, base_style: str, character_traits: List[str]) -> str:
        """Make dialogue more inquisitive and growth-oriented."""
        return f"growth_minded_{base_style}"

    def calculate_event_probability(self, event_type: str, base_probability: float) -> float:
        """Increase probability of learning opportunities."""
        learning_events = ['skill_challenge', 'teaching_moment', 'failure_lesson', 'discovery']

        if event_type in learning_events:
            return min(1.0, base_probability * 1.8 * self.parameters.learning_rate)

        return base_probability


class SandboxModeModifier(ModeModifier):
    """Modifier for Sandbox simulation mode - highly customizable."""

    def __init__(self, parameters: ModeParameters, custom_rules: Dict[str, Any] = None):
        self.parameters = parameters
        self.custom_rules = custom_rules or {}

    def modify_personality_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply custom personality modifications from configuration."""
        modified = base_weights.copy()

        # Apply custom weight multipliers
        if 'trait_multipliers' in self.custom_rules:
            for trait, multiplier in self.custom_rules['trait_multipliers'].items():
                if trait in modified:
                    modified[trait] = max(0.0, min(1.0, modified[trait] * multiplier))

        return modified

    def filter_available_actions(self, actions: List[str]) -> List[str]:
        """Filter actions based on custom rules."""
        if 'blocked_actions' in self.custom_rules:
            blocked = self.custom_rules['blocked_actions']
            actions = [action for action in actions if action not in blocked]

        if 'priority_actions' in self.custom_rules:
            priority = self.custom_rules['priority_actions']
            filtered = [action for action in priority if action in actions]
            filtered.extend([action for action in actions if action not in filtered])
            return filtered

        return actions

    def modify_dialogue_style(self, base_style: str, character_traits: List[str]) -> str:
        """Apply custom dialogue modifications."""
        if 'dialogue_style_override' in self.custom_rules:
            return self.custom_rules['dialogue_style_override']

        return base_style

    def calculate_event_probability(self, event_type: str, base_probability: float) -> float:
        """Apply custom event probability modifications."""
        if 'event_multipliers' in self.custom_rules:
            multiplier = self.custom_rules['event_multipliers'].get(event_type, 1.0)
            return min(1.0, base_probability * multiplier)

        return base_probability


class SimulationModeManager:
    """Manages simulation modes and applies their modifications."""

    def __init__(self):
        self.current_mode: Optional[SimulationMode] = None
        self.mode_modifier: Optional[ModeModifier] = None
        self.available_modes = self._create_default_modes()

    def _create_default_modes(self) -> Dict[str, SimulationMode]:
        """Create the four default simulation modes."""
        modes = {}

        # Comedy & Chaos Mode
        modes[ModeType.COMEDY_CHAOS.value] = SimulationMode(
            name="Comedy & Chaos",
            description="Maximum humor and absurdity",
            parameters=ModeParameters(
                chaos_frequency=0.8,
                exaggeration_factor=1.5,
                prank_likelihood=0.7,
                gossip_spread_rate=2.0,
                emotional_volatility=1.3,
                learning_rate=0.3,
                realism_level=0.2,
                consequence_weight=0.4,
                chaos_button_enabled=True,
                intervention_allowed=True
            ),
            available_events=[
                "alien_abduction", "surprise_pregnancy", "neighborhood_protest",
                "terrible_wifi", "food_fight", "wardrobe_malfunction"
            ],
            personality_modifiers={
                "impulsive": 1.4,
                "dramatic": 1.6,
                "mischievous": 1.5
            },
            special_rules=[
                "Failures become comedic rather than tragic",
                "Characters are more resilient to negative events",
                "Pranks always have unexpected consequences"
            ]
        )

        # Psychological & Deep Mode
        modes[ModeType.PSYCHOLOGICAL_DEEP.value] = SimulationMode(
            name="Psychological & Deep",
            description="Realistic emotions and long-term consequences",
            parameters=ModeParameters(
                chaos_frequency=0.2,
                exaggeration_factor=0.8,
                prank_likelihood=0.1,
                gossip_spread_rate=0.6,
                emotional_volatility=0.7,
                learning_rate=0.8,
                realism_level=0.9,
                consequence_weight=1.2,
                chaos_button_enabled=False,
                intervention_allowed=False
            ),
            available_events=[
                "relationship_conflict", "personal_crisis", "moral_dilemma",
                "breakthrough_moment", "family_visit", "job_loss"
            ],
            personality_modifiers={
                "empathetic": 1.3,
                "introspective": 1.4,
                "complex": 1.2
            },
            special_rules=[
                "Actions have lasting emotional consequences",
                "Characters develop and change over time",
                "Relationships are central to story development"
            ]
        )

        # Learning & Growth Mode
        modes[ModeType.LEARNING_GROWTH.value] = SimulationMode(
            name="Learning & Growth",
            description="Adaptive AI with skill development",
            parameters=ModeParameters(
                chaos_frequency=0.4,
                exaggeration_factor=1.0,
                prank_likelihood=0.3,
                gossip_spread_rate=1.0,
                emotional_volatility=0.9,
                learning_rate=1.5,
                realism_level=0.7,
                consequence_weight=0.8,
                chaos_button_enabled=True,
                intervention_allowed=True
            ),
            available_events=[
                "skill_challenge", "teaching_moment", "failure_lesson",
                "discovery", "mentor_appears", "competition"
            ],
            personality_modifiers={
                "curious": 1.4,
                "adaptive": 1.5,
                "perseverant": 1.3
            },
            special_rules=[
                "Characters gain skills from repeated actions",
                "Failures become learning opportunities",
                "Characters mentor each other"
            ]
        )

        # Sandbox Mode
        modes[ModeType.SANDBOX.value] = SimulationMode(
            name="Multi-Use Sandbox",
            description="Customizable parameters for experimentation",
            parameters=ModeParameters(
                chaos_frequency=0.5,
                exaggeration_factor=1.0,
                prank_likelihood=0.5,
                gossip_spread_rate=1.0,
                emotional_volatility=1.0,
                learning_rate=1.0,
                realism_level=0.5,
                consequence_weight=1.0,
                chaos_button_enabled=True,
                intervention_allowed=True
            ),
            available_events=[
                "custom_event", "player_defined", "experimental"
            ],
            personality_modifiers={},
            special_rules=[
                "All parameters can be modified at runtime",
                "Custom events can be added",
                "Experimental features enabled"
            ]
        )

        return modes

    def set_mode(self, mode_type: str, custom_config: Dict[str, Any] = None):
        """Set the current simulation mode."""
        if mode_type not in self.available_modes:
            raise ValueError(f"Unknown mode type: {mode_type}")

        self.current_mode = self.available_modes[mode_type]

        # Create appropriate modifier
        if mode_type == ModeType.COMEDY_CHAOS.value:
            self.mode_modifier = ComedyChaosModeModifier(self.current_mode.parameters)
        elif mode_type == ModeType.PSYCHOLOGICAL_DEEP.value:
            self.mode_modifier = PsychologicalDeepModeModifier(self.current_mode.parameters)
        elif mode_type == ModeType.LEARNING_GROWTH.value:
            self.mode_modifier = LearningGrowthModeModifier(self.current_mode.parameters)
        elif mode_type == ModeType.SANDBOX.value:
            self.mode_modifier = SandboxModeModifier(self.current_mode.parameters, custom_config)

    def apply_mode_to_personality(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply current mode modifications to personality weights."""
        if not self.mode_modifier:
            return base_weights

        return self.mode_modifier.modify_personality_weights(base_weights)

    def filter_actions(self, available_actions: List[str]) -> List[str]:
        """Filter available actions based on current mode."""
        if not self.mode_modifier:
            return available_actions

        return self.mode_modifier.filter_available_actions(available_actions)

    def modify_dialogue_style(self, base_style: str, character_traits: List[str]) -> str:
        """Modify dialogue style based on current mode."""
        if not self.mode_modifier:
            return base_style

        return self.mode_modifier.modify_dialogue_style(base_style, character_traits)

    def get_event_probability(self, event_type: str, base_probability: float = 0.1) -> float:
        """Get modified probability for an event type."""
        if not self.mode_modifier:
            return base_probability

        return self.mode_modifier.calculate_event_probability(event_type, base_probability)

    def is_chaos_button_enabled(self) -> bool:
        """Check if chaos button is enabled in current mode."""
        return self.current_mode.parameters.chaos_button_enabled if self.current_mode else False

    def is_intervention_allowed(self) -> bool:
        """Check if player interventions are allowed in current mode."""
        return self.current_mode.parameters.intervention_allowed if self.current_mode else True

    def get_mode_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current mode."""
        if not self.current_mode:
            return None

        return {
            'name': self.current_mode.name,
            'description': self.current_mode.description,
            'parameters': self.current_mode.parameters.__dict__,
            'special_rules': self.current_mode.special_rules
        }
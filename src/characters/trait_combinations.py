"""
Trait Combinations - Complex personality trait mashups and generators.

This module handles the creation of interesting personality combinations
and manages how different traits interact with each other.
"""

import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class TraitCategory(Enum):
    """Categories of personality traits."""
    SOCIAL = "social"
    EMOTIONAL = "emotional"
    INTELLECTUAL = "intellectual"
    CREATIVE = "creative"
    PHYSICAL = "physical"
    MORAL = "moral"
    QUIRKY = "quirky"


@dataclass
class TraitConflict:
    """Represents conflicting traits that create interesting tension."""
    trait1: str
    trait2: str
    conflict_description: str
    narrative_potential: float  # How much story potential this conflict creates


@dataclass
class TraitSynergy:
    """Represents traits that work well together."""
    traits: List[str]
    synergy_description: str
    bonus_behaviors: List[str]


class TraitLibrary:
    """Library of all available personality traits and their properties."""

    def __init__(self):
        self.traits_by_category = {
            TraitCategory.SOCIAL: [
                "extroverted", "introverted", "charismatic", "shy", "empathetic",
                "manipulative", "loyal", "flirtatious", "gossipy", "supportive",
                "competitive", "collaborative", "leader", "follower", "social_butterfly"
            ],
            TraitCategory.EMOTIONAL: [
                "optimistic", "pessimistic", "dramatic", "stoic", "sensitive",
                "emotionally_stable", "moody", "passionate", "calm", "anxious",
                "confident", "insecure", "jealous", "forgiving", "vengeful"
            ],
            TraitCategory.INTELLECTUAL: [
                "intelligent", "simple_minded", "curious", "closed_minded", "logical",
                "intuitive", "academic", "street_smart", "philosophical", "practical",
                "analytical", "creative_thinker", "skeptical", "gullible", "wise"
            ],
            TraitCategory.CREATIVE: [
                "artistic", "musical", "literary", "innovative", "traditional",
                "imaginative", "realistic", "experimental", "perfectionist",
                "spontaneous", "methodical", "inspired", "blocked", "versatile"
            ],
            TraitCategory.PHYSICAL: [
                "athletic", "sedentary", "energetic", "lazy", "adventurous",
                "cautious", "graceful", "clumsy", "strong", "delicate",
                "health_conscious", "indulgent", "outdoorsy", "homebody"
            ],
            TraitCategory.MORAL: [
                "honest", "deceitful", "altruistic", "selfish", "responsible",
                "irresponsible", "principled", "flexible", "justice_oriented",
                "pragmatic", "idealistic", "cynical", "trustworthy", "suspicious"
            ],
            TraitCategory.QUIRKY: [
                "eccentric", "conventional", "superstitious", "scientific",
                "collector", "minimalist", "night_owl", "early_bird", "messy",
                "organized", "technophobic", "tech_savvy", "nostalgic", "futuristic"
            ]
        }

        self.trait_conflicts = self._define_trait_conflicts()
        self.trait_synergies = self._define_trait_synergies()
        self.hidden_desire_templates = self._define_hidden_desires()

    def get_all_traits(self) -> List[str]:
        """Get a flat list of all available traits."""
        all_traits = []
        for category_traits in self.traits_by_category.values():
            all_traits.extend(category_traits)
        return all_traits

    def get_traits_by_category(self, category: TraitCategory) -> List[str]:
        """Get traits for a specific category."""
        return self.traits_by_category.get(category, [])

    def _define_trait_conflicts(self) -> List[TraitConflict]:
        """Define interesting trait conflicts."""
        return [
            TraitConflict("extroverted", "shy", "Wants to socialize but feels awkward", 0.8),
            TraitConflict("honest", "manipulative", "Believes in truth but uses deception", 0.9),
            TraitConflict("optimistic", "anxious", "Hopeful outlook battles constant worry", 0.7),
            TraitConflict("altruistic", "selfish", "Wants to help others but prioritizes self", 0.8),
            TraitConflict("perfectionist", "spontaneous", "Plans everything but craves freedom", 0.6),
            TraitConflict("logical", "superstitious", "Rational mind with irrational beliefs", 0.7),
            TraitConflict("confident", "insecure", "Projects confidence while doubting self", 0.9),
            TraitConflict("charismatic", "introverted", "Magnetic personality but needs alone time", 0.8),
            TraitConflict("principled", "pragmatic", "Strong morals but willing to compromise", 0.7),
            TraitConflict("adventurous", "anxious", "Seeks excitement but fears consequences", 0.8),
            TraitConflict("empathetic", "cynical", "Feels others' pain but expects the worst", 0.9),
            TraitConflict("creative", "perfectionist", "Innovative ideas stifled by high standards", 0.7)
        ]

    def _define_trait_synergies(self) -> List[TraitSynergy]:
        """Define traits that work well together."""
        return [
            TraitSynergy(
                ["charismatic", "manipulative"],
                "Master of social influence",
                ["convince_others", "lead_groups", "create_alliances"]
            ),
            TraitSynergy(
                ["creative", "eccentric"],
                "Unique artistic vision",
                ["create_unusual_art", "inspire_others", "think_outside_box"]
            ),
            TraitSynergy(
                ["analytical", "curious"],
                "Relentless investigator",
                ["solve_mysteries", "research_topics", "discover_secrets"]
            ),
            TraitSynergy(
                ["empathetic", "wise"],
                "Natural counselor",
                ["give_advice", "mediate_conflicts", "comfort_others"]
            ),
            TraitSynergy(
                ["athletic", "competitive"],
                "Natural champion",
                ["excel_at_sports", "motivate_teams", "push_limits"]
            ),
            TraitSynergy(
                ["optimistic", "supportive"],
                "Inspirational friend",
                ["boost_morale", "encourage_others", "spread_positivity"]
            ),
            TraitSynergy(
                ["philosophical", "introspective"],
                "Deep thinker",
                ["contemplate_existence", "offer_wisdom", "question_everything"]
            ),
            TraitSynergy(
                ["adventurous", "confident"],
                "Fearless explorer",
                ["take_risks", "lead_adventures", "inspire_courage"]
            )
        ]

    def _define_hidden_desires(self) -> Dict[str, List[str]]:
        """Define hidden desires that conflict with apparent traits."""
        return {
            "extroverted": ["solitude", "deep_connection", "authenticity"],
            "introverted": ["recognition", "leadership", "adventure"],
            "confident": ["acceptance", "vulnerability", "support"],
            "logical": ["magic", "mystery", "faith"],
            "altruistic": ["recognition", "power", "luxury"],
            "honest": ["deception", "secrets", "mystery"],
            "perfectionist": ["chaos", "spontaneity", "imperfection"],
            "responsible": ["freedom", "irresponsibility", "rebellion"],
            "optimistic": ["pessimism", "realism", "dark_thoughts"],
            "calm": ["excitement", "passion", "chaos"],
            "traditional": ["innovation", "rebellion", "change"],
            "organized": ["mess", "spontaneity", "chaos"],
            "athletic": ["laziness", "indulgence", "comfort"],
            "practical": ["dreams", "fantasy", "impracticality"]
        }


class TraitMashupGenerator:
    """Generates interesting personality combinations with conflicts and synergies."""

    def __init__(self, trait_library: TraitLibrary):
        self.trait_library = trait_library

    def generate_random_mashup(self, num_traits: int = 4) -> Dict[str, Any]:
        """Generate a random personality mashup."""
        all_traits = self.trait_library.get_all_traits()
        selected_traits = random.sample(all_traits, min(num_traits, len(all_traits)))

        return self.analyze_trait_combination(selected_traits)

    def generate_conflicted_personality(self) -> Dict[str, Any]:
        """Generate a personality with interesting internal conflicts."""
        conflicts = self.trait_library.trait_conflicts
        chosen_conflict = random.choice(conflicts)

        # Add 2-3 additional traits
        all_traits = self.trait_library.get_all_traits()
        additional_traits = random.sample(
            [t for t in all_traits if t not in [chosen_conflict.trait1, chosen_conflict.trait2]],
            random.randint(2, 3)
        )

        base_traits = [chosen_conflict.trait1, chosen_conflict.trait2] + additional_traits

        mashup = self.analyze_trait_combination(base_traits)
        mashup['primary_conflict'] = {
            'traits': [chosen_conflict.trait1, chosen_conflict.trait2],
            'description': chosen_conflict.conflict_description,
            'narrative_potential': chosen_conflict.narrative_potential
        }

        return mashup

    def generate_synergistic_personality(self) -> Dict[str, Any]:
        """Generate a personality with strong trait synergies."""
        synergies = self.trait_library.trait_synergies
        chosen_synergy = random.choice(synergies)

        # Add 1-2 additional traits
        all_traits = self.trait_library.get_all_traits()
        additional_traits = random.sample(
            [t for t in all_traits if t not in chosen_synergy.traits],
            random.randint(1, 2)
        )

        base_traits = chosen_synergy.traits + additional_traits

        mashup = self.analyze_trait_combination(base_traits)
        mashup['primary_synergy'] = {
            'traits': chosen_synergy.traits,
            'description': chosen_synergy.synergy_description,
            'bonus_behaviors': chosen_synergy.bonus_behaviors
        }

        return mashup

    def generate_themed_personality(self, theme: str) -> Dict[str, Any]:
        """Generate a personality around a specific theme."""
        themes = {
            "artist": ["creative", "sensitive", "eccentric", "passionate"],
            "scientist": ["logical", "curious", "analytical", "skeptical"],
            "rebel": ["independent", "defiant", "adventurous", "unconventional"],
            "caregiver": ["empathetic", "nurturing", "selfless", "patient"],
            "leader": ["confident", "charismatic", "decisive", "ambitious"],
            "mysterious": ["secretive", "intuitive", "complex", "enigmatic"],
            "comedian": ["humorous", "spontaneous", "optimistic", "attention_seeking"],
            "philosopher": ["wise", "introspective", "questioning", "idealistic"]
        }

        if theme not in themes:
            return self.generate_random_mashup()

        base_traits = themes[theme].copy()

        # Add some random variation
        all_traits = self.trait_library.get_all_traits()
        additional_trait = random.choice([t for t in all_traits if t not in base_traits])
        base_traits.append(additional_trait)

        mashup = self.analyze_trait_combination(base_traits)
        mashup['theme'] = theme

        return mashup

    def analyze_trait_combination(self, traits: List[str]) -> Dict[str, Any]:
        """Analyze a combination of traits for conflicts, synergies, and hidden desires."""

        # Find conflicts within the trait set
        conflicts = []
        for conflict in self.trait_library.trait_conflicts:
            if conflict.trait1 in traits and conflict.trait2 in traits:
                conflicts.append({
                    'trait1': conflict.trait1,
                    'trait2': conflict.trait2,
                    'description': conflict.conflict_description,
                    'narrative_potential': conflict.narrative_potential
                })

        # Find synergies within the trait set
        synergies = []
        for synergy in self.trait_library.trait_synergies:
            if all(trait in traits for trait in synergy.traits):
                synergies.append({
                    'traits': synergy.traits,
                    'description': synergy.synergy_description,
                    'bonus_behaviors': synergy.bonus_behaviors
                })

        # Generate hidden desires
        hidden_desires = []
        for trait in traits:
            possible_desires = self.trait_library.hidden_desire_templates.get(trait, [])
            if possible_desires:
                hidden_desires.extend(random.sample(possible_desires, min(1, len(possible_desires))))

        # Remove duplicates from hidden desires
        hidden_desires = list(set(hidden_desires))

        # Generate behavioral weights
        behavioral_weights = self._calculate_behavioral_weights(traits)

        # Generate dialogue style
        dialogue_style = self._determine_dialogue_style(traits)

        return {
            'base_traits': traits,
            'hidden_desires': hidden_desires,
            'behavioral_weights': behavioral_weights,
            'dialogue_style': dialogue_style,
            'conflicts': conflicts,
            'synergies': synergies,
            'complexity_score': len(conflicts) * 0.3 + len(synergies) * 0.2 + len(hidden_desires) * 0.1,
            'narrative_potential': sum(c['narrative_potential'] for c in conflicts) / max(1, len(conflicts))
        }

    def _calculate_behavioral_weights(self, traits: List[str]) -> Dict[str, float]:
        """Calculate behavioral weights based on trait combination."""
        weights = {
            'social_seeking': 0.5,
            'risk_taking': 0.5,
            'emotional_expression': 0.5,
            'intellectual_pursuits': 0.5,
            'creative_expression': 0.5,
            'physical_activity': 0.5,
            'moral_behavior': 0.5,
            'routine_following': 0.5,
            'leadership': 0.5,
            'independence': 0.5
        }

        # Adjust weights based on traits
        trait_adjustments = {
            'extroverted': {'social_seeking': 0.3},
            'introverted': {'social_seeking': -0.3, 'independence': 0.2},
            'adventurous': {'risk_taking': 0.4},
            'cautious': {'risk_taking': -0.4, 'routine_following': 0.2},
            'dramatic': {'emotional_expression': 0.4},
            'stoic': {'emotional_expression': -0.3},
            'intelligent': {'intellectual_pursuits': 0.3},
            'creative': {'creative_expression': 0.4},
            'athletic': {'physical_activity': 0.4},
            'principled': {'moral_behavior': 0.3},
            'charismatic': {'leadership': 0.3, 'social_seeking': 0.2},
            'organized': {'routine_following': 0.3}
        }

        for trait in traits:
            if trait in trait_adjustments:
                for weight_type, adjustment in trait_adjustments[trait].items():
                    weights[weight_type] = max(0.0, min(1.0, weights[weight_type] + adjustment))

        return weights

    def _determine_dialogue_style(self, traits: List[str]) -> str:
        """Determine dialogue style based on trait combination."""
        style_mapping = {
            ('dramatic', 'emotional'): 'theatrical',
            ('logical', 'analytical'): 'precise',
            ('sarcastic', 'witty'): 'sardonic',
            ('empathetic', 'supportive'): 'nurturing',
            ('confident', 'charismatic'): 'commanding',
            ('shy', 'sensitive'): 'hesitant',
            ('philosophical', 'wise'): 'contemplative',
            ('eccentric', 'creative'): 'whimsical',
            ('honest', 'direct'): 'straightforward',
            ('mysterious', 'secretive'): 'cryptic'
        }

        # Check for exact trait pairs
        for trait_pair, style in style_mapping.items():
            if all(trait in traits for trait in trait_pair):
                return style

        # Check for individual trait styles
        individual_styles = {
            'dramatic': 'expressive',
            'logical': 'rational',
            'empathetic': 'caring',
            'confident': 'assertive',
            'shy': 'quiet',
            'philosophical': 'thoughtful',
            'creative': 'imaginative',
            'honest': 'direct',
            'sarcastic': 'witty'
        }

        for trait in traits:
            if trait in individual_styles:
                return individual_styles[trait]

        return 'conversational'  # Default style


def create_personality_from_template(template_name: str, trait_library: TraitLibrary,
                                   generator: TraitMashupGenerator) -> Dict[str, Any]:
    """Create a personality from a named template."""

    templates = {
        "romantic_nihilist": {
            "base_traits": ["romantic", "pessimistic", "philosophical", "sensitive"],
            "forced_conflicts": [("romantic", "pessimistic")],
            "theme": "tortured_romantic"
        },
        "anxious_overachiever": {
            "base_traits": ["perfectionist", "anxious", "intelligent", "competitive"],
            "forced_conflicts": [("perfectionist", "anxious")],
            "theme": "high_achiever"
        },
        "charismatic_loner": {
            "base_traits": ["charismatic", "introverted", "mysterious", "independent"],
            "forced_conflicts": [("charismatic", "introverted")],
            "theme": "enigmatic_leader"
        },
        "creative_critic": {
            "base_traits": ["creative", "critical", "passionate", "perfectionist"],
            "forced_conflicts": [("creative", "critical")],
            "theme": "artistic_perfectionist"
        }
    }

    if template_name not in templates:
        return generator.generate_random_mashup()

    template = templates[template_name]
    mashup = generator.analyze_trait_combination(template["base_traits"])
    mashup["template_name"] = template_name
    mashup["theme"] = template.get("theme", "custom")

    return mashup
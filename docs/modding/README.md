# Misfits! Modding Guide

Welcome to the Misfits! modding system! This game is built from the ground up to support extensive customization and modification.

## Getting Started

### Custom Personalities
Create new personality templates by adding JSON files to `data/personalities/`:

```json
{
  "my_custom_personality": {
    "description": "Your personality description",
    "base_traits": ["trait1", "trait2", "trait3"],
    "hidden_desires": ["desire1", "desire2"],
    "behavioral_weights": {
      "social_seeking": 0.7,
      "risk_taking": 0.3,
      "creative_expression": 0.8
    },
    "dialogue_style": "your_style_name"
  }
}
```

### Custom Events
Add chaos events by modifying `data/events/chaos_events.json`:

```json
{
  "your_event_id": {
    "name": "Your Event Name",
    "description": "What happens during this event",
    "category": "comedic",
    "chaos_level": "moderate",
    "duration_minutes": 60,
    "probability_weight": 0.4,
    "prerequisites": ["daytime"],
    "immediate_effects": {
      "fun_increase": 0.5
    },
    "narrative_hooks": [
      "Something interesting happens"
    ]
  }
}
```

### Custom Simulation Modes
Create new simulation modes in `data/simulation_modes/`:

```yaml
name: "Your Mode Name"
description: "Mode description"
parameters:
  chaos_frequency: 0.6
  exaggeration_factor: 1.2
  learning_rate: 0.8
special_rules:
  - "Custom rule 1"
  - "Custom rule 2"
```

## Advanced Modding

### Custom Python Modules
You can extend the game by adding Python modules to the `src/` directory. Follow the existing patterns and use dependency injection.

### Custom LLM Providers
Implement the `BaseLLMProvider` interface to add support for new LLM services.

### Custom UI Components
Extend the UI system by creating new components that follow the existing patterns in `src/ui/`.

## Modding Best Practices

1. **Follow the existing code style** - Use the same patterns as the core game
2. **Test your modifications** - Use `python scripts/run_tests.py`
3. **Document your changes** - Add comments and documentation
4. **Share with the community** - Submit pull requests for useful mods

## Community Resources

- GitHub Repository: [Misfits Game Repository]
- Discord Community: [Coming Soon]
- Mod Sharing Platform: [Coming Soon]
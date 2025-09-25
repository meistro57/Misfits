# Misfits! - AI-Driven Life Simulation Game

## Project Overview

**Tagline:** "No scripts. Just chaos."

Misfits! is an open-source, AI-powered life simulation game where quirky characters live, love, and create emergent narratives through their own AI-driven personalities. Players act as architects of chaos rather than micromanagers, setting the stage for unpredictable digital soap operas.

## Core Philosophy

- **AI-Powered Personalities**: Each character driven by independent AI models with evolving behaviors
- **Emergent Drama Over Scripts**: No pre-set goals; compelling stories unfold organically
- **Open-Source Community-Driven**: Fully open-source with extensive modding support

## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Game Engine** | Godot 4.x | Open-source, lightweight, flexible |
| **AI Personalities** | Local LLMs (Ollama, LM Studio, GPT4All) | Character behavior and dialogue |
| **Memory System** | SQLite + FAISS | Vector database for character memories |
| **Audio** | TTS Engine + Lip-sync | Generated character voices |
| **Configuration** | JSON/YAML | Personality and mode definitions |

## Project Structure

```
misfits/
├── src/
│   ├── core/
│   │   ├── ai_personality_engine.py
│   │   ├── memory_system.py
│   │   ├── world_tick.py
│   │   └── simulation_modes.py
│   ├── characters/
│   │   ├── misfit.py
│   │   ├── personality_templates/
│   │   └── trait_combinations.py
│   ├── world/
│   │   ├── environment.py
│   │   ├── chaos_events.py
│   │   └── intervention_system.py
│   ├── ui/
│   │   ├── chaos_button.py
│   │   ├── intervention_mode.py
│   │   └── observation_panel.py
│   └── utils/
│       ├── llm_interface.py
│       ├── vector_db.py
│       └── config_loader.py
├── data/
│   ├── personalities/
│   │   ├── base_templates.json
│   │   └── trait_mashups.json
│   ├── simulation_modes/
│   │   ├── comedy_chaos.yaml
│   │   ├── psychological_deep.yaml
│   │   ├── learning_growth.yaml
│   │   └── sandbox.yaml
│   └── events/
│       ├── chaos_events.json
│       └── world_events.json
├── assets/
│   ├── models/
│   ├── textures/
│   ├── audio/
│   └── ui/
├── tests/
├── docs/
│   ├── api/
│   ├── modding/
│   └── deployment/
├── scripts/
│   ├── setup.py
│   ├── run_tests.py
│   └── build.py
└── README.md
```

## Core Systems Implementation

### 1. AI Personality Engine

**File**: `src/core/ai_personality_engine.py`

The central system driving character behavior with the following components:

- **Personality Core**: LLM-based personality module
- **Trait Mashups**: Complex trait combinations (e.g., "Paranoid + Charismatic")
- **Hidden Desires**: Secret motivations that conflict with apparent behavior
- **Control Layer**: Hidden need states that ground AI decisions

**Key Features**:
- Dynamic dialogue generation (no pre-written scripts)
- Personality evolution based on experiences
- Context-aware decision making

### 2. Persistent Memory System

**File**: `src/core/memory_system.py`

Vector database implementation for character memory storage:

- **Memory Storage**: Significant events, relationships, grudges
- **Contextual Retrieval**: AI queries memories for decision-making
- **Legacy Saves**: "Ghost data" persisting across playthroughs
- **Gossip Network**: Memory sharing between characters

**Technical Implementation**:
```python
# Example memory structure
class MisfitMemory:
    def __init__(self, character_id):
        self.character_id = character_id
        self.vector_store = FAISSVectorStore()
        self.recent_memories = deque(maxlen=50)
        
    def store_memory(self, event, emotional_weight, participants):
        embedding = self.embed_event(event, emotional_weight)
        self.vector_store.add(embedding, metadata={
            'event': event,
            'timestamp': time.now(),
            'participants': participants,
            'emotional_weight': emotional_weight
        })
    
    def retrieve_relevant_memories(self, context, limit=10):
        query_embedding = self.embed_context(context)
        return self.vector_store.similarity_search(query_embedding, limit)
```

### 3. World Tick System

**File**: `src/core/world_tick.py`

The core game loop that processes each character's actions:

1. **Context Gathering**: Combine need states, memories, and social web data
2. **AI Processing**: Send context to LLM for decision generation
3. **Action Execution**: Process AI output into game world changes
4. **Memory Update**: Store new experiences and interactions

### 4. Simulation Modes

**File**: `src/core/simulation_modes.py`

Four distinct gameplay modes that filter AI behavior:

- **Comedy & Chaos**: Exaggerated personalities, frequent pranks
- **Psychological & Deep**: Realistic emotions, long-term consequences
- **Learning & Growth**: Adaptive AI with skill development
- **Multi-Use Sandbox**: Customizable parameters for experimentation

## Player Interaction Systems

### Chaos Button
**File**: `src/ui/chaos_button.py`
- Triggers random world-altering events
- Event pool configurable via JSON
- Integration with simulation mode for appropriate chaos level

### Intervention Mode
**File**: `src/ui/intervention_mode.py`
- Whisper suggestions directly to character AI
- Preserves character agency while allowing influence
- Tracks intervention history for narrative consequences

### Direct Communication
**File**: `src/characters/dialogue_system.py`
- Chat interface with any character
- AI responds in-character
- Conversations stored in memory system

## Configuration and Modding

### Personality Templates
**File**: `data/personalities/base_templates.json`

```json
{
  "romantic_nihilist": {
    "base_traits": ["romantic", "nihilist"],
    "hidden_desires": ["connection", "meaning"],
    "behavioral_weights": {
      "social_seeking": 0.7,
      "philosophical": 0.8,
      "impulsive": 0.6
    },
    "dialogue_style": "poetic_pessimistic"
  }
}
```

### Simulation Mode Configuration
**File**: `data/simulation_modes/comedy_chaos.yaml`

```yaml
name: "Comedy & Chaos"
description: "Maximum humor and absurdity"
parameters:
  chaos_frequency: 0.8
  exaggeration_factor: 1.5
  prank_likelihood: 0.7
  gossip_spread_rate: 2.0
  emotional_volatility: 1.3
chaos_button_enabled: true
available_events:
  - "alien_abduction"
  - "surprise_pregnancy"
  - "neighborhood_protest"
  - "terrible_wifi"
```

## Development Setup

### Requirements
- Python 3.10+
- Godot 4.x
- Local LLM setup (Ollama recommended)
- SQLite with FAISS
- TTS engine (espeak/piper)

### Installation
```bash
# Clone repository
git clone https://github.com/your-org/misfits-game.git
cd misfits-game

# Install dependencies
pip install -r requirements.txt

# Setup local LLM
ollama pull llama2:7b-chat

# Initialize database
python scripts/setup.py

# Run tests
python scripts/run_tests.py

# Launch game
python main.py
```

### Development Commands

```bash
# Run specific tests
python -m pytest tests/test_ai_engine.py

# Generate new personality template
python scripts/generate_personality.py --traits "anxious,creative"

# Validate simulation mode config
python scripts/validate_config.py data/simulation_modes/

# Build distribution
python scripts/build.py --platform all
```

## API Documentation

### Character AI Interface
```python
class MisfitCharacter:
    def process_world_tick(self, context: WorldContext) -> ActionDecision
    def receive_player_message(self, message: str) -> str
    def add_memory(self, event: Event) -> None
    def get_personality_state(self) -> PersonalityState
```

### Memory System Interface
```python
class MemorySystem:
    def store_event(self, character_id: str, event: Event) -> None
    def query_memories(self, character_id: str, context: str) -> List[Memory]
    def create_gossip_chain(self, initial_event: Event) -> GossipChain
```

### Intervention System Interface
```python
class InterventionSystem:
    def whisper_suggestion(self, character_id: str, suggestion: str) -> None
    def trigger_chaos_event(self, event_type: str = "random") -> Event
    def modify_environment(self, changes: EnvironmentChanges) -> None
```

## Modding Support

### Creating Custom Personalities
1. Add JSON template to `data/personalities/`
2. Define traits, hidden desires, and behavioral weights
3. Test with `python scripts/test_personality.py --personality your_personality`

### Adding Chaos Events
1. Define event in `data/events/chaos_events.json`
2. Implement handler in `src/world/chaos_events.py`
3. Add to appropriate simulation mode configurations

### Custom Objects and Interactions
1. Create object definition in `data/objects/`
2. Implement behavior in `src/world/interactive_objects.py`
3. Add AI interaction rules in personality templates

## Testing Strategy

### Unit Tests
- AI personality generation and consistency
- Memory storage and retrieval accuracy
- Event system functionality
- Configuration validation

### Integration Tests
- Multi-character interaction scenarios
- Long-term memory persistence
- Cross-system communication
- Performance under load

### AI Behavior Tests
- Personality consistency over time
- Appropriate responses to interventions
- Memory-influenced decision making
- Emergent narrative quality

## Performance Considerations

### LLM Optimization
- Batch character processing where possible
- Cache common personality responses
- Optimize prompt engineering for speed
- Implement fallback behaviors for LLM failures

### Memory System Scaling
- Regular vector database optimization
- Memory pruning for long-running games
- Efficient similarity search implementations
- Backup and recovery procedures

## Deployment and Distribution

### Build Targets
- Windows (x64)
- Linux (x64)
- macOS (ARM64/x64)
- Web (WASM) - Limited AI capabilities

### Packaging
- Standalone executables with embedded LLM
- Docker containers for server deployment
- Flatpak/Snap packages for Linux
- Steam Workshop integration for mods

## Future Roadmap

### Version 1.0 (Core Release)
- Basic AI personality system
- Memory persistence
- Four simulation modes
- Modding framework

### Version 1.1 (Community Features)
- Enhanced modding tools
- Personality sharing marketplace
- Advanced chaos events
- Multi-language support

### Version 2.0 (Expansion Content)
- Misfits: Apocalypse mode
- Advanced learning systems
- Multiplayer neighborhoods
- VR support

## Contributing

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Document all public APIs
- Include type hints

### Pull Request Process
1. Fork repository and create feature branch
2. Implement changes with tests
3. Update documentation
4. Submit PR with detailed description

### Community Guidelines
- Be respectful and inclusive
- Share knowledge and help others
- Test thoroughly before submitting
- Follow open source etiquette

## License

MIT License - See LICENSE file for details

This project is fully open source and community-driven. All contributions are welcome!

---

*"In Misfits!, chaos isn't a bug - it's the feature that makes every story unique."*

#!/usr/bin/env python3
"""
Misfits! - AI-Driven Life Simulation Game

Main entry point for the Misfits game. Initializes all systems and starts
the game loop with the configured simulation mode.

Usage:
    python main.py [--config CONFIG_FILE] [--mode SIMULATION_MODE] [--debug]
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from typing import Dict

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.ai_personality_engine import AIPersonalityEngine, PersonalityTraits
from src.core.memory_system import MemorySystem
from src.core.world_tick import WorldTicker
from src.core.simulation_modes import SimulationModeManager
from src.characters.misfit import MisfitCharacter, PhysicalTraits, LifeStage
from src.characters.trait_combinations import TraitLibrary, TraitMashupGenerator, create_personality_from_template
from src.world.environment import WorldEnvironment
from src.world.chaos_events import ChaosEventSystem
from src.world.intervention_system import InterventionSystem
from src.ui.chaos_button import ChaosButtonUI
from src.ui.intervention_mode import InterventionModeUI
from src.ui.observation_panel import ObservationPanel
from src.utils.llm_interface import LLMInterface, LLMConfig, LLMProvider, MockLLMProvider
from src.utils.vector_db import create_vector_db
from src.utils.config_loader import create_config_loader


class MisfitsGame:
    """Main game class that coordinates all systems."""

    def __init__(self, config_file: str = "game_config.yaml"):
        self.config_loader = create_config_loader()
        self.config = self._load_config(config_file)

        # Core systems
        self.llm_interface: LLMInterface = None
        self.memory_system: MemorySystem = None
        self.ai_engine: AIPersonalityEngine = None
        self.world_ticker: WorldTicker = None
        self.world_environment: WorldEnvironment = None
        self.chaos_system: ChaosEventSystem = None
        self.intervention_system: InterventionSystem = None
        self.simulation_mode_manager: SimulationModeManager = None

        # UI systems
        self.chaos_button: ChaosButtonUI = None
        self.intervention_ui: InterventionModeUI = None
        self.observation_panel: ObservationPanel = None

        # Game state
        self.characters: Dict[str, MisfitCharacter] = {}
        self.is_running = False

        # Initialize logging
        self._setup_logging()

    def _load_config(self, config_file: str):
        """Load and validate configuration."""
        try:
            config_dict = self.config_loader.load_config_file(config_file)
            env_config = self.config_loader.load_environment_config()
            if env_config:
                config_dict = self.config_loader.merge_configs(config_dict, env_config)

            config = self.config_loader.parse_game_config(config_dict)

            if not self.config_loader.validate_config(config):
                errors = self.config_loader.get_validation_errors()
                print("Configuration validation errors:")
                for error in errors:
                    print(f"  - {error}")
                print("\nUsing default configuration...")
                config = self.config_loader.create_default_config()

            return config

        except FileNotFoundError:
            print(f"Configuration file {config_file} not found. Using default configuration.")
            default_config = self.config_loader.create_default_config()
            env_config = self.config_loader.load_environment_config()
            if env_config:
                config_dict = self.config_loader.config_to_dict(default_config)
                config_dict = self.config_loader.merge_configs(config_dict, env_config)
                return self.config_loader.parse_game_config(config_dict)

            return default_config

        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration...")
            return self.config_loader.create_default_config()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('misfits.log')
            ]
        )

        if self.config.debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)

    async def initialize(self):
        """Initialize all game systems."""
        logging.info("Initializing Misfits! game systems...")

        # Initialize LLM interface
        await self._initialize_llm()

        # Initialize memory system
        self._initialize_memory_system()

        # Initialize AI engine
        self._initialize_ai_engine()

        # Initialize world systems
        self._initialize_world_systems()

        # Initialize UI systems
        self._initialize_ui_systems()

        # Initialize simulation mode
        self._initialize_simulation_mode()

        # Create initial characters
        await self._create_initial_characters()

        logging.info("All systems initialized successfully!")

    async def _initialize_llm(self):
        """Initialize LLM interface."""
        llm_configs = []

        for provider_name, llm_config in self.config.llm.items():
            config = LLMConfig(
                provider=LLMProvider(llm_config.provider),
                host=llm_config.host,
                port=llm_config.port,
                model=llm_config.model,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                timeout=llm_config.timeout,
                api_key=llm_config.api_key
            )
            llm_configs.append(config)

        # Add mock provider as fallback if no real providers configured
        if not llm_configs:
            llm_configs.append(LLMConfig(provider=LLMProvider.OLLAMA))  # Will use mock

        primary_provider = LLMProvider.OLLAMA  # Default primary
        if llm_configs:
            primary_provider = llm_configs[0].provider

        self.llm_interface = LLMInterface(llm_configs, primary_provider)
        await self.llm_interface.__aenter__()

        # Test connection
        health = await self.llm_interface.check_all_providers_health()
        logging.info(f"LLM provider health: {health}")

    def _initialize_memory_system(self):
        """Initialize memory system."""
        # Create vector database for semantic memory search
        vector_config = {
            "embedding": {
                "provider": self.config.vector_db.embedding_provider,
                "model": self.config.vector_db.embedding_model,
                "dimension": self.config.vector_db.dimension
            },
            "storage": {
                "type": self.config.vector_db.storage_type,
                "path": self.config.vector_db.storage_path
            }
        }

        vector_db = create_vector_db(vector_config)
        self.memory_system = MemorySystem(self.config.database.path)

        # TODO: Integrate vector DB with memory system for semantic search

    def _initialize_ai_engine(self):
        """Initialize AI personality engine."""
        self.ai_engine = AIPersonalityEngine(self.llm_interface)
        self.ai_engine.set_memory_system(self.memory_system)

    def _initialize_world_systems(self):
        """Initialize world systems."""
        # World environment
        self.world_environment = WorldEnvironment()

        # Chaos events system
        self.chaos_system = ChaosEventSystem()

        # Intervention system
        self.intervention_system = InterventionSystem()
        self.intervention_system.set_chaos_system(self.chaos_system)

        # World ticker (main game loop)
        self.world_ticker = WorldTicker(self.ai_engine, self.memory_system)

    def _initialize_ui_systems(self):
        """Initialize UI systems."""
        # Chaos button
        self.chaos_button = ChaosButtonUI()

        # Set up chaos button callback
        def trigger_chaos():
            character_states = {
                char_id: {
                    "traits": char.personality_traits.base_traits,
                    "mood": char.mood,
                    "location": char.current_location,
                    "age_stage": char.life_stage.value
                }
                for char_id, char in self.characters.items()
            }

            world_state = self.world_environment.get_world_state_summary()
            return self.chaos_system.trigger_chaos_event(
                "comedy_chaos", character_states, world_state
            )

        self.chaos_button.set_callbacks(on_chaos_triggered=trigger_chaos)

        # Intervention UI
        self.intervention_ui = InterventionModeUI()

        # Observation panel
        self.observation_panel = ObservationPanel()

    def _initialize_simulation_mode(self):
        """Initialize simulation mode manager."""
        self.simulation_mode_manager = SimulationModeManager()
        self.simulation_mode_manager.set_mode(self.config.simulation.default_mode)

        # Configure chaos button based on simulation mode
        if self.simulation_mode_manager.is_chaos_button_enabled():
            self.chaos_button.enable_button()
        else:
            self.chaos_button.disable_button()

    async def _create_initial_characters(self):
        """Create initial characters for the game."""
        trait_library = TraitLibrary()
        trait_generator = TraitMashupGenerator(trait_library)

        # Create a few diverse characters
        character_templates = [
            ("Alice", "romantic_nihilist"),
            ("Bob", "anxious_overachiever"),
            ("Carol", "charismatic_loner"),
            ("Dave", None)  # Random generation
        ]

        for name, template in character_templates:
            if template:
                personality_data = create_personality_from_template(
                    template, trait_library, trait_generator
                )
            else:
                personality_data = trait_generator.generate_conflicted_personality()

            # Create personality traits
            personality_traits = PersonalityTraits(
                base_traits=personality_data["base_traits"],
                hidden_desires=personality_data["hidden_desires"],
                behavioral_weights=personality_data["behavioral_weights"],
                dialogue_style=personality_data["dialogue_style"]
            )

            # Create physical traits (randomized for demo)
            import random
            physical_traits = PhysicalTraits(
                height=random.choice(["short", "average", "tall"]),
                build=random.choice(["slim", "average", "athletic", "heavy"]),
                hair_color=random.choice(["brown", "black", "blonde", "red", "gray"]),
                eye_color=random.choice(["brown", "blue", "green", "hazel"]),
                distinctive_features=[],
                clothing_style=random.choice(["casual", "formal", "trendy", "bohemian"]),
                age_appearance=random.randint(22, 45)
            )

            # Create character
            character_id = name.lower()
            character = MisfitCharacter(
                character_id=character_id,
                name=name,
                personality_traits=personality_traits,
                physical_traits=physical_traits,
                life_stage=LifeStage.ADULT
            )

            # Create AI personality core
            personality_core = self.ai_engine.create_character(character_id, personality_traits)

            # Set up memory system for character
            character_memory = self.memory_system.get_character_memory(character_id)

            # Link AI components to character
            character.set_ai_components(personality_core, character_memory)

            # Add to world
            self.characters[character_id] = character
            self.world_ticker.add_character(character_id, "living_room")

            logging.info(f"Created character: {name} ({template or 'random'})")

    async def run(self):
        """Run the main game loop."""
        self.is_running = True
        logging.info("Starting Misfits! game loop...")

        try:
            # Start the world ticker
            tick_task = asyncio.create_task(
                self.world_ticker.run_continuous(self.config.simulation.tick_interval)
            )

            # Start UI update loop
            ui_task = asyncio.create_task(self._ui_update_loop())

            # Wait for tasks to complete
            await asyncio.gather(tick_task, ui_task)

        except KeyboardInterrupt:
            logging.info("Game interrupted by user")
        except Exception as e:
            logging.error(f"Game error: {e}")
        finally:
            await self.shutdown()

    async def _ui_update_loop(self):
        """Update UI systems regularly."""
        while self.is_running:
            try:
                # Update chaos button
                chaos_status = self.chaos_button.update()
                if chaos_status:
                    logging.debug(f"Chaos button status: {chaos_status}")

                # Update observation panel with current game state
                game_state = {
                    "characters": {
                        char_id: {
                            "name": char.name,
                            "mood": char.mood,
                            "location": char.current_location,
                            "current_activity": char.current_location,  # Simplified
                            "traits": char.personality_traits.base_traits,
                            "energy": char.energy
                        }
                        for char_id, char in self.characters.items()
                    },
                    "world": self.world_environment.get_world_state_summary(),
                    "recent_events": []  # Would be populated with recent events
                }

                self.observation_panel.update_game_state(game_state)

                # Check for active chaos events
                active_events = self.chaos_system.get_active_events()
                if active_events:
                    logging.info(f"Active chaos events: {len(active_events)}")

                await asyncio.sleep(self.config.ui.update_frequency)

            except Exception as e:
                logging.error(f"UI update error: {e}")
                await asyncio.sleep(1.0)

    async def shutdown(self):
        """Shutdown game systems gracefully."""
        logging.info("Shutting down Misfits! game...")

        self.is_running = False

        # Stop world ticker
        self.world_ticker.stop()

        # Shutdown LLM interface
        if self.llm_interface:
            await self.llm_interface.__aexit__(None, None, None)

        logging.info("Shutdown complete")

    def get_game_status(self):
        """Get current game status for debugging."""
        return {
            "running": self.is_running,
            "tick_count": self.world_ticker.tick_count if self.world_ticker else 0,
            "character_count": len(self.characters),
            "simulation_mode": self.simulation_mode_manager.current_mode.name if self.simulation_mode_manager.current_mode else "None",
            "llm_stats": self.llm_interface.get_provider_stats() if self.llm_interface else {},
            "chaos_button": self.chaos_button.get_button_display_info() if self.chaos_button else {}
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Misfits! AI-Driven Life Simulation Game")
    parser.add_argument("--config", default="game_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--mode", default=None,
                       help="Simulation mode (comedy_chaos, psychological_deep, learning_growth, sandbox)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")

    args = parser.parse_args()

    # Create and run game
    game = MisfitsGame(args.config)

    # Override debug mode if specified
    if args.debug:
        game.config.debug_mode = True

    try:
        await game.initialize()

        # Override simulation mode if specified
        if args.mode and game.simulation_mode_manager:
            game.simulation_mode_manager.set_mode(args.mode)

        print("\n" + "="*50)
        print("🎲 Welcome to Misfits! 🎲")
        print("AI-Driven Life Simulation Game")
        print("="*50)
        current_mode = (
            game.simulation_mode_manager.current_mode.name
            if game.simulation_mode_manager and game.simulation_mode_manager.current_mode
            else game.config.simulation.default_mode
        )
        print(f"Simulation Mode: {current_mode}")
        print(f"Characters: {len(game.characters)}")
        print(f"Tick Interval: {game.config.simulation.tick_interval}s")
        print("="*50)
        print("\nGame is starting... Press Ctrl+C to exit")
        print()

        await game.run()

    except Exception as e:
        logging.error(f"Failed to start game: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

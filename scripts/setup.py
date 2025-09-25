#!/usr/bin/env python3
"""
Setup Script for Misfits! Game

This script initializes the game database, creates default configuration files,
and sets up the development environment.
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.utils.config_loader import create_config_loader
from src.core.memory_system import SQLiteMemoryStore
from src.utils.vector_db import create_vector_db


def setup_database(db_path: str = "misfits.db"):
    """Initialize the main game database."""
    print(f"Setting up database: {db_path}")

    # Create database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create game state table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_state (
            id INTEGER PRIMARY KEY,
            save_name TEXT UNIQUE NOT NULL,
            game_data TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)

    # Create character profiles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS character_profiles (
            character_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            personality_data TEXT NOT NULL,
            physical_traits TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)

    # Create world events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS world_events (
            event_id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            event_data TEXT NOT NULL,
            participants TEXT,
            timestamp REAL NOT NULL,
            significance REAL DEFAULT 0.5
        )
    """)

    # Create settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print(f"✓ Database initialized: {db_path}")


def setup_memory_database(db_path: str = "memories.db"):
    """Initialize the memory system database."""
    print(f"Setting up memory database: {db_path}")

    # Use the memory store to initialize
    memory_store = SQLiteMemoryStore(db_path)
    print(f"✓ Memory database initialized: {db_path}")


def setup_vector_database(db_path: str = "vectors.db"):
    """Initialize the vector database."""
    print(f"Setting up vector database: {db_path}")

    # Create vector database with mock embedding provider
    config = {
        "embedding": {
            "provider": "mock",
            "dimension": 384
        },
        "storage": {
            "type": "sqlite",
            "path": db_path
        }
    }

    vector_db = create_vector_db(config)
    print(f"✓ Vector database initialized: {db_path}")


def setup_config_files():
    """Create default configuration files if they don't exist."""
    print("Setting up configuration files...")

    config_loader = create_config_loader()
    config_dir = Path(config_loader.config_dir)

    # Create config directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create default game config if it doesn't exist
    game_config_path = config_dir / "game_config.yaml"
    if not game_config_path.exists():
        config_loader.save_default_config("game_config.yaml")
        print(f"✓ Created default game configuration: {game_config_path}")
    else:
        print(f"✓ Game configuration already exists: {game_config_path}")

    # Create additional mode configs if they don't exist
    mode_configs = [
        "learning_growth.yaml",
        "sandbox.yaml"
    ]

    for mode_config in mode_configs:
        mode_path = config_dir / "simulation_modes" / mode_config
        if not mode_path.exists():
            mode_path.parent.mkdir(parents=True, exist_ok=True)
            # Create basic mode config (would be more detailed in production)
            basic_config = {
                "name": mode_config.replace(".yaml", "").replace("_", " ").title(),
                "description": f"Configuration for {mode_config.replace('.yaml', '')} mode",
                "parameters": {
                    "chaos_frequency": 0.5,
                    "exaggeration_factor": 1.0,
                    "learning_rate": 1.0
                }
            }

            import yaml
            with open(mode_path, 'w') as f:
                yaml.dump(basic_config, f, default_flow_style=False)
            print(f"✓ Created mode configuration: {mode_path}")


def setup_directories():
    """Create necessary directories."""
    print("Setting up directories...")

    directories = [
        "data/personalities",
        "data/simulation_modes",
        "data/events",
        "assets/models",
        "assets/textures",
        "assets/audio",
        "assets/ui",
        "tests",
        "docs/api",
        "docs/modding",
        "docs/deployment",
        "logs"
    ]

    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {directory}")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")

    required_modules = [
        ("yaml", "PyYAML"),
        ("numpy", "numpy"),
        ("aiohttp", "aiohttp")
    ]

    missing_modules = []

    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            missing_modules.append(package_name)
            print(f"✗ {package_name} is missing")

    if missing_modules:
        print(f"\nMissing dependencies: {', '.join(missing_modules)}")
        print("Install them with: pip install " + " ".join(missing_modules))
        return False

    return True


def check_llm_providers():
    """Check if LLM providers are available."""
    print("Checking LLM providers...")

    # Check Ollama
    try:
        import aiohttp
        import asyncio

        async def check_ollama():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:11434/api/tags", timeout=5) as response:
                        if response.status == 200:
                            return True
            except:
                pass
            return False

        ollama_available = asyncio.run(check_ollama())
        if ollama_available:
            print("✓ Ollama is running on localhost:11434")
        else:
            print("✗ Ollama not detected (this is optional)")
    except ImportError:
        print("✗ Cannot check Ollama (aiohttp not installed)")

    # Check LM Studio
    try:
        async def check_lm_studio():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:1234/v1/models", timeout=5) as response:
                        if response.status == 200:
                            return True
            except:
                pass
            return False

        lm_studio_available = asyncio.run(check_lm_studio())
        if lm_studio_available:
            print("✓ LM Studio is running on localhost:1234")
        else:
            print("✗ LM Studio not detected (this is optional)")
    except ImportError:
        print("✗ Cannot check LM Studio (aiohttp not installed)")

    print("Note: Mock LLM provider is always available for testing")


def main():
    """Main setup function."""
    print("🎲 Misfits! Game Setup 🎲")
    print("=" * 50)

    # Change to project root directory
    os.chdir(project_root)

    try:
        # Check dependencies first
        if not check_dependencies():
            print("\n⚠️  Please install missing dependencies before continuing")
            return 1

        # Setup directories
        setup_directories()

        # Setup configuration files
        setup_config_files()

        # Setup databases
        setup_database()
        setup_memory_database()
        setup_vector_database()

        # Check LLM providers
        check_llm_providers()

        print("\n" + "=" * 50)
        print("🎉 Setup completed successfully!")
        print("=" * 50)
        print("\nYou can now run the game with:")
        print("  python main.py")
        print("\nFor help:")
        print("  python main.py --help")
        print("\nFor development:")
        print("  python main.py --debug")

        return 0

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
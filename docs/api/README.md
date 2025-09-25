# Misfits! API Documentation

This directory contains API documentation for the Misfits! AI-Driven Life Simulation Game.

## Core APIs

### AI Personality Engine
- **Location**: `src/core/ai_personality_engine.py`
- **Purpose**: Character AI behavior and decision-making
- **Key Classes**: `AIPersonalityEngine`, `PersonalityCore`, `PersonalityTraits`

### Memory System
- **Location**: `src/core/memory_system.py`
- **Purpose**: Character memory storage and retrieval
- **Key Classes**: `MemorySystem`, `MisfitMemory`, `SQLiteMemoryStore`

### World Systems
- **Location**: `src/world/`
- **Purpose**: Game world management and simulation
- **Key Modules**: `environment.py`, `chaos_events.py`, `intervention_system.py`

### Character System
- **Location**: `src/characters/`
- **Purpose**: Character creation and management
- **Key Classes**: `MisfitCharacter`, `TraitMashupGenerator`

## Usage Examples

See the `main.py` file for complete integration examples.

## API Reference

Detailed API documentation will be generated using tools like Sphinx in future versions.
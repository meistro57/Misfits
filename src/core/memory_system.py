"""
Memory System - Vector database implementation for character memory storage.

This module handles persistent storage and retrieval of character memories,
experiences, and relationships using vector embeddings for semantic search.
"""

import time
import sqlite3
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
from abc import ABC, abstractmethod


@dataclass
class Memory:
    """Individual memory record."""
    memory_id: str
    character_id: str
    event: str
    emotional_weight: float
    participants: List[str]
    location: str
    timestamp: float
    tags: List[str]
    embedding: Optional[List[float]] = None


@dataclass
class GossipChain:
    """Represents gossip spreading between characters."""
    original_event: str
    chain: List[Dict[str, Any]]
    distortion_level: float


class VectorStore(ABC):
    """Abstract interface for vector storage and similarity search."""

    @abstractmethod
    def add_memory(self, memory: Memory) -> None:
        """Add a memory with its embedding to the store."""
        pass

    @abstractmethod
    def similarity_search(self, query_embedding: List[float],
                         character_id: str, limit: int = 10) -> List[Memory]:
        """Find similar memories for a character."""
        pass

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID."""
        pass


class EmbeddingProvider(ABC):
    """Abstract interface for generating embeddings."""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        pass


class SQLiteMemoryStore:
    """SQLite-based memory persistence with vector search simulation."""

    def __init__(self, db_path: str = "memories.db"):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        cursor = self.connection.cursor()

        # Main memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                event TEXT NOT NULL,
                emotional_weight REAL NOT NULL,
                participants TEXT NOT NULL,
                location TEXT NOT NULL,
                timestamp REAL NOT NULL,
                tags TEXT NOT NULL,
                embedding TEXT
            )
        """)

        # Relationships table for tracking character connections
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                character_id TEXT NOT NULL,
                other_character TEXT NOT NULL,
                relationship_strength REAL NOT NULL,
                relationship_type TEXT,
                last_interaction REAL NOT NULL,
                PRIMARY KEY (character_id, other_character)
            )
        """)

        # Legacy saves for cross-playthrough persistence
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS legacy_data (
                character_id TEXT PRIMARY KEY,
                ghost_memories TEXT,
                reputation_score REAL,
                legacy_traits TEXT
            )
        """)

        self.connection.commit()

    def store_memory(self, memory: Memory):
        """Store a memory in the database."""
        cursor = self.connection.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO memories
            (memory_id, character_id, event, emotional_weight, participants,
             location, timestamp, tags, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.memory_id,
            memory.character_id,
            memory.event,
            memory.emotional_weight,
            json.dumps(memory.participants),
            memory.location,
            memory.timestamp,
            json.dumps(memory.tags),
            json.dumps(memory.embedding) if memory.embedding else None
        ))

        self.connection.commit()

    def get_memories_by_character(self, character_id: str,
                                 limit: int = 50) -> List[Memory]:
        """Get recent memories for a character."""
        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT * FROM memories
            WHERE character_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (character_id, limit))

        rows = cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search_memories_by_text(self, character_id: str, search_text: str,
                               limit: int = 10) -> List[Memory]:
        """Simple text-based memory search (placeholder for vector search)."""
        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT * FROM memories
            WHERE character_id = ? AND (
                event LIKE ? OR
                location LIKE ? OR
                participants LIKE ?
            )
            ORDER BY emotional_weight DESC, timestamp DESC
            LIMIT ?
        """, (character_id, f"%{search_text}%", f"%{search_text}%",
              f"%{search_text}%", limit))

        rows = cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    def update_relationship(self, character_id: str, other_character: str,
                           interaction_type: str, strength_delta: float):
        """Update relationship strength between characters."""
        cursor = self.connection.cursor()

        # Get current relationship
        cursor.execute("""
            SELECT relationship_strength FROM relationships
            WHERE character_id = ? AND other_character = ?
        """, (character_id, other_character))

        row = cursor.fetchone()
        current_strength = row[0] if row else 0.0
        new_strength = max(-1.0, min(1.0, current_strength + strength_delta))

        cursor.execute("""
            INSERT OR REPLACE INTO relationships
            (character_id, other_character, relationship_strength,
             relationship_type, last_interaction)
            VALUES (?, ?, ?, ?, ?)
        """, (character_id, other_character, new_strength, interaction_type, time.time()))

        self.connection.commit()

    def get_relationships(self, character_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all relationships for a character."""
        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT other_character, relationship_strength, relationship_type, last_interaction
            FROM relationships
            WHERE character_id = ?
            ORDER BY relationship_strength DESC
        """, (character_id,))

        relationships = {}
        for row in cursor.fetchall():
            relationships[row[0]] = {
                'strength': row[1],
                'type': row[2],
                'last_interaction': row[3]
            }

        return relationships

    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            memory_id=row[0],
            character_id=row[1],
            event=row[2],
            emotional_weight=row[3],
            participants=json.loads(row[4]),
            location=row[5],
            timestamp=row[6],
            tags=json.loads(row[7]),
            embedding=json.loads(row[8]) if row[8] else None
        )


class MisfitMemory:
    """Main memory system for individual characters."""

    def __init__(self, character_id: str, store: SQLiteMemoryStore,
                 embedding_provider: Optional[EmbeddingProvider] = None):
        self.character_id = character_id
        self.store = store
        self.embedding_provider = embedding_provider
        self.recent_memories = deque(maxlen=50)
        self.memory_counter = 0

    async def store_memory(self, event: str, emotional_weight: float,
                          participants: List[str], location: str = "unknown",
                          tags: List[str] = None) -> str:
        """Store a new memory with optional embedding."""

        if tags is None:
            tags = []

        memory_id = f"{self.character_id}_memory_{self.memory_counter}"
        self.memory_counter += 1

        # Generate embedding if provider is available
        embedding = None
        if self.embedding_provider:
            try:
                embedding_text = f"{event} at {location} with {', '.join(participants)}"
                embedding = await self.embedding_provider.embed_text(embedding_text)
            except Exception as e:
                print(f"Failed to generate embedding: {e}")

        memory = Memory(
            memory_id=memory_id,
            character_id=self.character_id,
            event=event,
            emotional_weight=emotional_weight,
            participants=participants,
            location=location,
            timestamp=time.time(),
            tags=tags,
            embedding=embedding
        )

        # Store in database
        self.store.store_memory(memory)

        # Keep in recent memory
        self.recent_memories.append(memory)

        # Update relationships based on participants
        for participant in participants:
            if participant != self.character_id:
                relationship_delta = emotional_weight * 0.1  # Scale relationship change
                self.store.update_relationship(
                    self.character_id, participant, "interaction", relationship_delta
                )

        return memory_id

    def retrieve_relevant_memories(self, context: str, limit: int = 10) -> List[Memory]:
        """Retrieve memories relevant to the given context."""

        # For now, use text-based search
        # In production, this would use vector similarity search
        memories = self.store.search_memories_by_text(
            self.character_id, context, limit
        )

        return memories

    def query_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """Query memories using text search."""
        return self.store.search_memories_by_text(self.character_id, query, limit)

    def get_recent_memories(self, limit: int = 10) -> List[Memory]:
        """Get most recent memories."""
        return list(self.recent_memories)[-limit:]

    def get_relationships(self) -> Dict[str, Dict[str, Any]]:
        """Get character's relationships."""
        return self.store.get_relationships(self.character_id)

    def forget_memory(self, memory_id: str):
        """Mark a memory as forgotten (low-priority removal)."""
        # In production, this might move to a separate "forgotten" table
        # or mark with a forgotten flag rather than deleting
        pass


class MemorySystem:
    """Central memory system managing all character memories."""

    def __init__(self, db_path: str = "memories.db"):
        self.store = SQLiteMemoryStore(db_path)
        self.character_memories: Dict[str, MisfitMemory] = {}
        self.embedding_provider = None

    def set_embedding_provider(self, provider: EmbeddingProvider):
        """Set embedding provider for semantic search."""
        self.embedding_provider = provider
        # Update existing character memories
        for memory in self.character_memories.values():
            memory.embedding_provider = provider

    def get_character_memory(self, character_id: str) -> MisfitMemory:
        """Get or create memory system for a character."""
        if character_id not in self.character_memories:
            self.character_memories[character_id] = MisfitMemory(
                character_id, self.store, self.embedding_provider
            )
        return self.character_memories[character_id]

    async def store_event(self, character_id: str, event: str, emotional_weight: float,
                         participants: List[str], location: str = "unknown") -> str:
        """Store an event in a character's memory."""
        memory = self.get_character_memory(character_id)
        return await memory.store_memory(event, emotional_weight, participants, location)

    def query_memories(self, character_id: str, context: str) -> List[Memory]:
        """Query memories for a specific character."""
        memory = self.get_character_memory(character_id)
        return memory.query_memories(context)

    def create_gossip_chain(self, initial_event: str,
                           participants: List[str]) -> GossipChain:
        """Create a gossip chain that spreads information between characters."""

        chain = []
        distortion = 0.0

        # Simple gossip propagation
        for i, participant in enumerate(participants[1:], 1):
            # Each retelling adds some distortion
            distortion += 0.1 * i

            # Modify the event slightly based on character personality
            # In production, this would use AI to generate variations
            modified_event = f"{initial_event} (as told by {participants[i-1]})"

            chain.append({
                'teller': participants[i-1],
                'listener': participant,
                'version': modified_event,
                'distortion_added': 0.1 * i
            })

        return GossipChain(
            original_event=initial_event,
            chain=chain,
            distortion_level=distortion
        )

    def get_character_relationships(self, character_id: str) -> Dict[str, Dict[str, Any]]:
        """Get relationships for a character."""
        memory = self.get_character_memory(character_id)
        return memory.get_relationships()
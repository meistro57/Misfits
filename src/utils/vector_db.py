"""
Vector Database - Embedding and similarity search utilities.

This module provides vector database functionality for semantic
memory search using different embedding providers and storage backends.
"""

import numpy as np
import sqlite3
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path


@dataclass
class EmbeddingVector:
    """Vector embedding with metadata."""
    vector_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    text: str
    timestamp: float


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding vector for text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing and development."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding based on text."""
        # Simple hash-based embedding for testing
        text_hash = hash(text)
        np.random.seed(abs(text_hash) % (2**32))

        # Generate random vector and normalize
        vector = np.random.normal(0, 1, self.dimension)
        return vector / np.linalg.norm(vector)

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


class SentenceTransformersProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider (requires sentence-transformers)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers library not installed")
        return self._model

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding using SentenceTransformers."""
        model = self._get_model()
        embedding = model.encode([text])[0]
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        model = self._get_model()
        embeddings = model.encode(texts)
        return [emb for emb in embeddings]

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        model = self._get_model()
        return model.get_sentence_embedding_dimension()


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace embedding provider using transformers library."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def _get_model_and_tokenizer(self):
        """Lazy load model and tokenizer."""
        if self._model is None or self._tokenizer is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model.to(self._device)
            except ImportError:
                raise ImportError("transformers library not installed")

        return self._model, self._tokenizer

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding using HuggingFace transformers."""
        model, tokenizer = self._get_model_and_tokenizer()

        try:
            import torch

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)

            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()

            return embeddings.cpu().numpy()

        except ImportError:
            raise ImportError("torch library not installed")

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        model, _ = self._get_model_and_tokenizer()
        return model.config.hidden_size


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    async def add_vector(self, vector: EmbeddingVector) -> str:
        """Add a vector to the store."""
        pass

    @abstractmethod
    async def add_vectors(self, vectors: List[EmbeddingVector]) -> List[str]:
        """Add multiple vectors to the store."""
        pass

    @abstractmethod
    async def search_similar(self, query_vector: np.ndarray, k: int = 10,
                           metadata_filter: Dict[str, Any] = None) -> List[Tuple[EmbeddingVector, float]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def get_vector(self, vector_id: str) -> Optional[EmbeddingVector]:
        """Get a vector by ID."""
        pass

    @abstractmethod
    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID."""
        pass

    @abstractmethod
    async def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a vector."""
        pass


class SQLiteVectorStore(VectorStore):
    """SQLite-based vector store with basic similarity search."""

    def __init__(self, db_path: str, embedding_dimension: int):
        self.db_path = db_path
        self.embedding_dimension = embedding_dimension
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    vector_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)

            # Create index on timestamp for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON vectors(timestamp)
            """)

            conn.commit()

    async def add_vector(self, vector: EmbeddingVector) -> str:
        """Add a vector to the SQLite store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Serialize embedding and metadata
            embedding_blob = pickle.dumps(vector.embedding)
            metadata_json = json.dumps(vector.metadata)

            cursor.execute("""
                INSERT OR REPLACE INTO vectors
                (vector_id, embedding, text, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (vector.vector_id, embedding_blob, vector.text,
                  metadata_json, vector.timestamp))

            conn.commit()

        return vector.vector_id

    async def add_vectors(self, vectors: List[EmbeddingVector]) -> List[str]:
        """Add multiple vectors to the store."""
        vector_ids = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for vector in vectors:
                embedding_blob = pickle.dumps(vector.embedding)
                metadata_json = json.dumps(vector.metadata)

                cursor.execute("""
                    INSERT OR REPLACE INTO vectors
                    (vector_id, embedding, text, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (vector.vector_id, embedding_blob, vector.text,
                      metadata_json, vector.timestamp))

                vector_ids.append(vector.vector_id)

            conn.commit()

        return vector_ids

    async def search_similar(self, query_vector: np.ndarray, k: int = 10,
                           metadata_filter: Dict[str, Any] = None) -> List[Tuple[EmbeddingVector, float]]:
        """Search for similar vectors using cosine similarity."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get all vectors (this is not efficient for large datasets)
            cursor.execute("""
                SELECT vector_id, embedding, text, metadata, timestamp
                FROM vectors
            """)

            results = []

            for row in cursor.fetchall():
                vector_id, embedding_blob, text, metadata_json, timestamp = row

                # Deserialize
                embedding = pickle.loads(embedding_blob)
                metadata = json.loads(metadata_json)

                # Apply metadata filter if provided
                if metadata_filter:
                    skip = False
                    for filter_key, filter_value in metadata_filter.items():
                        if metadata.get(filter_key) != filter_value:
                            skip = True
                            break
                    if skip:
                        continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_vector, embedding)

                vector_obj = EmbeddingVector(
                    vector_id=vector_id,
                    embedding=embedding,
                    metadata=metadata,
                    text=text,
                    timestamp=timestamp
                )

                results.append((vector_obj, similarity))

            # Sort by similarity (highest first) and return top k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def get_vector(self, vector_id: str) -> Optional[EmbeddingVector]:
        """Get a vector by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT vector_id, embedding, text, metadata, timestamp
                FROM vectors
                WHERE vector_id = ?
            """, (vector_id,))

            row = cursor.fetchone()
            if not row:
                return None

            vector_id, embedding_blob, text, metadata_json, timestamp = row

            embedding = pickle.loads(embedding_blob)
            metadata = json.loads(metadata_json)

            return EmbeddingVector(
                vector_id=vector_id,
                embedding=embedding,
                metadata=metadata,
                text=text,
                timestamp=timestamp
            )

    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM vectors WHERE vector_id = ?", (vector_id,))
            conn.commit()

            return cursor.rowcount > 0

    async def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a vector."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            metadata_json = json.dumps(metadata)

            cursor.execute("""
                UPDATE vectors
                SET metadata = ?
                WHERE vector_id = ?
            """, (metadata_json, vector_id))

            conn.commit()

            return cursor.rowcount > 0


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for development and testing."""

    def __init__(self):
        self.vectors: Dict[str, EmbeddingVector] = {}

    async def add_vector(self, vector: EmbeddingVector) -> str:
        """Add a vector to memory."""
        self.vectors[vector.vector_id] = vector
        return vector.vector_id

    async def add_vectors(self, vectors: List[EmbeddingVector]) -> List[str]:
        """Add multiple vectors to memory."""
        vector_ids = []
        for vector in vectors:
            self.vectors[vector.vector_id] = vector
            vector_ids.append(vector.vector_id)
        return vector_ids

    async def search_similar(self, query_vector: np.ndarray, k: int = 10,
                           metadata_filter: Dict[str, Any] = None) -> List[Tuple[EmbeddingVector, float]]:
        """Search for similar vectors in memory."""
        results = []

        for vector in self.vectors.values():
            # Apply metadata filter if provided
            if metadata_filter:
                skip = False
                for filter_key, filter_value in metadata_filter.items():
                    if vector.metadata.get(filter_key) != filter_value:
                        skip = True
                        break
                if skip:
                    continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_vector, vector.embedding)
            results.append((vector, similarity))

        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def get_vector(self, vector_id: str) -> Optional[EmbeddingVector]:
        """Get a vector by ID."""
        return self.vectors.get(vector_id)

    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID."""
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            return True
        return False

    async def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a vector."""
        if vector_id in self.vectors:
            self.vectors[vector_id].metadata = metadata
            return True
        return False


class VectorDB:
    """Main vector database interface combining embedding and storage."""

    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def add_text(self, text_id: str, text: str, metadata: Dict[str, Any] = None) -> str:
        """Add text to the database with automatic embedding generation."""
        import time

        if metadata is None:
            metadata = {}

        # Generate embedding
        embedding = await self.embedding_provider.embed_text(text)

        # Create vector object
        vector = EmbeddingVector(
            vector_id=text_id,
            embedding=embedding,
            metadata=metadata,
            text=text,
            timestamp=time.time()
        )

        # Store in vector store
        return await self.vector_store.add_vector(vector)

    async def add_texts(self, texts: List[Tuple[str, str, Dict[str, Any]]]) -> List[str]:
        """Add multiple texts to the database."""
        import time

        # Extract texts for batch embedding
        text_strings = [text for _, text, _ in texts]
        embeddings = await self.embedding_provider.embed_batch(text_strings)

        # Create vector objects
        vectors = []
        for i, (text_id, text, metadata) in enumerate(texts):
            if metadata is None:
                metadata = {}

            vector = EmbeddingVector(
                vector_id=text_id,
                embedding=embeddings[i],
                metadata=metadata,
                text=text,
                timestamp=time.time()
            )
            vectors.append(vector)

        # Store all vectors
        return await self.vector_store.add_vectors(vectors)

    async def search_by_text(self, query_text: str, k: int = 10,
                           metadata_filter: Dict[str, Any] = None) -> List[Tuple[EmbeddingVector, float]]:
        """Search for similar texts using text query."""
        # Generate embedding for query
        query_embedding = await self.embedding_provider.embed_text(query_text)

        # Search in vector store
        return await self.vector_store.search_similar(query_embedding, k, metadata_filter)

    async def search_by_embedding(self, query_embedding: np.ndarray, k: int = 10,
                                metadata_filter: Dict[str, Any] = None) -> List[Tuple[EmbeddingVector, float]]:
        """Search for similar vectors using embedding query."""
        return await self.vector_store.search_similar(query_embedding, k, metadata_filter)

    async def get_text(self, text_id: str) -> Optional[EmbeddingVector]:
        """Get text and its embedding by ID."""
        return await self.vector_store.get_vector(text_id)

    async def delete_text(self, text_id: str) -> bool:
        """Delete text by ID."""
        return await self.vector_store.delete_vector(text_id)

    async def update_text_metadata(self, text_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a text."""
        return await self.vector_store.update_metadata(text_id, metadata)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        return {
            "embedding_provider": self.embedding_provider.__class__.__name__,
            "vector_store": self.vector_store.__class__.__name__,
            "embedding_dimension": self.embedding_provider.get_embedding_dimension()
        }


def create_vector_db(config: Dict[str, Any]) -> VectorDB:
    """Create vector database from configuration."""

    # Create embedding provider
    embedding_config = config.get("embedding", {})
    provider_type = embedding_config.get("provider", "mock")

    if provider_type == "mock":
        embedding_provider = MockEmbeddingProvider(
            dimension=embedding_config.get("dimension", 384)
        )
    elif provider_type == "sentence_transformers":
        model_name = embedding_config.get("model", "all-MiniLM-L6-v2")
        embedding_provider = SentenceTransformersProvider(model_name)
    elif provider_type == "huggingface":
        model_name = embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        embedding_provider = HuggingFaceEmbeddingProvider(model_name)
    else:
        # Default to mock
        embedding_provider = MockEmbeddingProvider()

    # Create vector store
    storage_config = config.get("storage", {})
    storage_type = storage_config.get("type", "memory")

    if storage_type == "sqlite":
        db_path = storage_config.get("path", "vectors.db")
        dimension = embedding_provider.get_embedding_dimension()
        vector_store = SQLiteVectorStore(db_path, dimension)
    else:
        # Default to in-memory
        vector_store = InMemoryVectorStore()

    return VectorDB(embedding_provider, vector_store)
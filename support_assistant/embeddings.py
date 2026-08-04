"""
Zepto Support Assistant (RAG) - Embeddings Manager
Provides HuggingFace / SentenceTransformers embedding pipeline with fallback support.
"""

from typing import List
import numpy as np

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False

from support_assistant.config import DEFAULT_EMBEDDING_MODEL


class FallbackEmbeddings:
    """Lightweight deterministic fallback embedding generator for offline/keyless testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_text(self, text: str) -> List[float]:
        """Maps input string to normalized float vector using hashing."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        for idx, word in enumerate(words):
            hash_val = hash(word) % self.dimension
            vec[hash_val] += 1.0 / (idx + 1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_text(text)


def get_embedding_function():
    """Initializes and returns embedding model with fallback protection."""
    if HAS_HUGGINGFACE:
        try:
            return HuggingFaceEmbeddings(
                model_name=DEFAULT_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            print(f"Notice: HuggingFaceEmbeddings initialization fallback ({e}). Using FallbackEmbeddings.")
            return FallbackEmbeddings()
    else:
        return FallbackEmbeddings()


if __name__ == "__main__":
    emb_fn = get_embedding_function()
    vec = emb_fn.embed_query("What is the refund policy for damaged groceries?")
    print(f"Generated query embedding vector of dimension: {len(vec)}")

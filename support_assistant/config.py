"""
Zepto Support Assistant (RAG) - Configuration Settings
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_STORE_DIR = BASE_DIR / "faiss_index"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Embedding & Vector Database Parameters
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_NAME = "zepto_policies_index"

# LLM Providers Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # Options: gemini, openai, local

# RAG Retrieval Top-K
RETRIEVAL_K = 4

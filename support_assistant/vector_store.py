"""
Zepto Support Assistant (RAG) - FAISS Vector Store Manager
Indexes document chunks, manages vector database persistence, and performs similarity searches.
"""

from pathlib import Path
from typing import List, Tuple

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from dataclasses import dataclass, field

        @dataclass
        class Document:
            page_content: str
            metadata: dict = field(default_factory=dict)

try:
    from langchain_community.vectorstores import FAISS
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from support_assistant.config import FAISS_INDEX_NAME, RETRIEVAL_K, VECTOR_STORE_DIR
from support_assistant.embeddings import get_embedding_function


class SimpleInMemoryVectorStore:
    """Fallback vector index if native FAISS CPU bindings are unavailable."""

    def __init__(self, documents: List[Document] = None, embedding_fn = None):
        self.documents = documents or []
        self.embedding_fn = embedding_fn or get_embedding_function()
        self.doc_embeddings = []
        if self.documents:
            texts = [doc.page_content for doc in self.documents]
            self.doc_embeddings = self.embedding_fn.embed_documents(texts)

    def similarity_search(self, query: str, k: int = RETRIEVAL_K) -> List[Document]:
        if not self.documents:
            return []
        query_vec = self.embedding_fn.embed_query(query)
        scores = []
        for idx, doc_vec in enumerate(self.doc_embeddings):
            # Cosine similarity
            dot = sum(q * d for q, d in zip(query_vec, doc_vec))
            scores.append((dot, self.documents[idx]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:k]]


class VectorStoreManager:
    """Manages creation, loading, saving, and querying of FAISS vector store."""

    def __init__(self, index_dir: Path = VECTOR_STORE_DIR):
        self.index_dir = Path(index_dir)
        self.embedding_fn = get_embedding_function()
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]) -> any:
        """Builds vector database index from list of LangChain Document chunks."""
        if not documents:
            raise ValueError("Cannot build vector store with empty document list.")

        print(f"Indexing {len(documents)} document chunks into vector database...")
        if HAS_FAISS:
            try:
                self.vector_store = FAISS.from_documents(documents, self.embedding_fn)
                self.save_vector_store()
                return self.vector_store
            except Exception as e:
                print(f"FAISS indexing error ({e}). Using SimpleInMemoryVectorStore fallback.")
                self.vector_store = SimpleInMemoryVectorStore(documents, self.embedding_fn)
                return self.vector_store
        else:
            self.vector_store = SimpleInMemoryVectorStore(documents, self.embedding_fn)
            return self.vector_store

    def save_vector_store(self) -> None:
        """Saves FAISS index to disk."""
        if self.vector_store and hasattr(self.vector_store, "save_local"):
            self.vector_store.save_local(folder_path=str(self.index_dir), index_name=FAISS_INDEX_NAME)
            print(f"FAISS index successfully saved to: {self.index_dir}")

    def load_vector_store(self) -> any:
        """Loads index from disk if present."""
        index_file = self.index_dir / f"{FAISS_INDEX_NAME}.faiss"
        if HAS_FAISS and index_file.exists():
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=str(self.index_dir),
                    embeddings=self.embedding_fn,
                    index_name=FAISS_INDEX_NAME,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing FAISS index from: {self.index_dir}")
                return self.vector_store
            except Exception as e:
                print(f"Failed to load FAISS index ({e}). Index re-creation required.")
                return None
        return None

    def search_similar_chunks(self, query: str, k: int = RETRIEVAL_K) -> List[Document]:
        """Retrieves top-K relevant document chunks matching user query."""
        if not self.vector_store:
            self.load_vector_store()

        if not self.vector_store:
            return []

        return self.vector_store.similarity_search(query, k=k)


if __name__ == "__main__":
    sample_docs = [
        Document(page_content="Zepto refund policy guarantees 100% refund for damaged perishable items.", metadata={"source": "refund_policy.txt"}),
        Document(page_content="Standard employee leaves include 12 days sick leave and 18 days privilege leave.", metadata={"source": "leave_policy.txt"}),
    ]
    mgr = VectorStoreManager()
    mgr.create_vector_store(sample_docs)
    results = mgr.search_similar_chunks("How many sick leaves do employees get?")
    print("Found matching chunks:", [r.page_content for r in results])

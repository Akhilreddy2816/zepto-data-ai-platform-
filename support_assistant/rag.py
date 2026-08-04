"""
Zepto Support Assistant (RAG) - Document Extraction & Chunking Engine
Loads PDF, TXT, MD company policy documents and splits text using RecursiveCharacterTextSplitter.
"""

from pathlib import Path
from typing import List, Optional

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
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=500, chunk_overlap=80, **kwargs):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_documents(self, documents: List[Document]) -> List[Document]:
                chunks = []
                for doc in documents:
                    text = doc.page_content
                    start = 0
                    while start < len(text):
                        end = start + self.chunk_size
                        chunk_text = text[start:end]
                        chunks.append(Document(page_content=chunk_text, metadata=dict(doc.metadata)))
                        start += (self.chunk_size - self.chunk_overlap)
                return chunks

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    HAS_LOADERS = True
except ImportError:
    HAS_LOADERS = False

from support_assistant.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENTS_DIR
from support_assistant.vector_store import VectorStoreManager


class RAGDocumentPipeline:
    """Manages document extraction, chunking, and index construction."""

    def __init__(self, docs_dir: Path = DOCUMENTS_DIR):
        self.docs_dir = Path(docs_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_mgr = VectorStoreManager()

    def load_document(self, file_path: Path) -> List[Document]:
        """Loads a single text or PDF file into LangChain Documents."""
        file_path = Path(file_path)
        if not file_path.exists():
            return []

        ext = file_path.suffix.lower()
        docs = []

        if ext == ".pdf" and HAS_LOADERS:
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            except Exception as e:
                print(f"Error reading PDF {file_path}: {e}")
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                docs = [Document(page_content=content, metadata={"source": file_path.name})]
            except Exception as e:
                print(f"Error reading text file {file_path}: {e}")

        return docs


    def load_all_documents(self) -> List[Document]:
        """Scans documents directory and extracts all text & PDF files."""
        all_docs = []
        for file_path in self.docs_dir.glob("*.*"):
            if file_path.suffix.lower() in [".txt", ".md", ".pdf"]:
                docs = self.load_document(file_path)
                all_docs.extend(docs)
        return all_docs

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits raw documents into overlap-managed chunks."""
        chunks = self.text_splitter.split_documents(documents)
        print(f"Chunked {len(documents)} raw documents into {len(chunks)} split chunks.")
        return chunks

    def build_or_refresh_knowledge_base(self) -> int:
        """Loads documents from disk, chunks text, and refreshes vector index."""
        raw_docs = self.load_all_documents()
        if not raw_docs:
            print("No policy documents found in documents directory.")
            return 0

        chunks = self.chunk_documents(raw_docs)
        self.vector_mgr.create_vector_store(chunks)
        return len(chunks)

    def retrieve_relevant_context(self, query: str) -> List[Document]:
        """Retrieves top relevant chunks for a question."""
        return self.vector_mgr.search_similar_chunks(query)


if __name__ == "__main__":
    pipeline = RAGDocumentPipeline()
    num_chunks = pipeline.build_or_refresh_knowledge_base()
    print(f"Knowledge Base Indexing Complete. Total Chunks Indexed: {num_chunks}")

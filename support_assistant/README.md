# Module 3: GenAI Support Assistant (RAG)

Retrieval Augmented Generation (RAG) Support Assistant for Zepto employee and customer policy Q&A.

## Architecture

```
Policy PDFs & Text Files (documents/)
           │
           ▼
    Text Extractor & Recursive Splitter (rag.py)
           │
           ▼
    SentenceTransformers Embeddings (embeddings.py)
           │
           ▼
     FAISS Vector Store Index (vector_store.py)
           │
           ▼
Retriever + Google Gemini API / OpenAI / Heuristic Engine (chatbot.py)
           │
           ▼
   Grounded Answer with Policy Source Citations
```

## Policy Documents Included

1. `employee_handbook.txt`: Work hours, code of conduct, data security rules.
2. `refund_policy.txt`: 100% quick-commerce refund guarantees and processing SLA.
3. `hr_policy.txt`: Probation, learning allowance, annual bonus cycle.
4. `leave_policy.txt`: Sick leave, privilege leave, maternity/paternity entitlements.
5. `safety_guidelines.txt`: Dark store safety & delivery partner road rules.

## Key Features

- **Document Indexing**: Splits text into 500-character chunks with 80-character overlap.
- **FAISS Vector DB**: Fast similarity search for top-K document chunks.
- **Anti-Hallucination Grounding**: Strict system prompt enforcing facts-only responses.
- **Source Attribution**: Automatic listing of matching policy document file names.
- **Multi-LLM & Offline Fallback**: Integrates Google Gemini API with fallback to offline generation.

## How to Run Standalone

```bash
# Refresh knowledge base index
python -m support_assistant.rag

# Run sample query
python -m support_assistant.chatbot
```

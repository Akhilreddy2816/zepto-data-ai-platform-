"""
pytest Unit Tests for Module 3: GenAI Support Assistant (RAG)
"""

import pytest
from support_assistant.rag import Document
from support_assistant.chatbot import ZeptoSupportChatbot
from support_assistant.embeddings import get_embedding_function
from support_assistant.vector_store import VectorStoreManager


def test_embedding_function():
    emb = get_embedding_function()
    vec = emb.embed_query("Zepto refund policy")
    assert isinstance(vec, list)
    assert len(vec) > 0


def test_vector_store_indexing():
    docs = [
        Document(page_content="Zepto delivery guarantee is 15 minutes.", metadata={"source": "test_policy.txt"}),
        Document(page_content="Sick leaves accrue 1 day per month.", metadata={"source": "leave_policy.txt"}),
    ]
    mgr = VectorStoreManager()
    mgr.create_vector_store(docs)
    results = mgr.search_similar_chunks("What is the delivery guarantee time?")
    assert len(results) > 0
    assert "15 minutes" in results[0].page_content


def test_rag_chatbot_query():
    bot = ZeptoSupportChatbot()
    res = bot.ask("What is the refund policy for damaged grocery items?")
    assert "query" in res
    assert "answer" in res
    assert isinstance(res["answer"], str)
    assert len(res["answer"]) > 10

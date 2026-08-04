"""
Zepto Support Assistant (RAG) - Chatbot & Generation Engine
Executes grounded RAG response generation using Google Gemini, OpenAI, or local heuristic fallback.
Includes strict anti-hallucination prompt templates, source citations, and conversation history.
"""

from typing import Dict, List, Optional

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

from support_assistant.config import GEMINI_API_KEY, LLM_PROVIDER, OPENAI_API_KEY
from support_assistant.rag import RAGDocumentPipeline

SYSTEM_PROMPT = """You are the official Zepto Enterprise AI Support Assistant.
Your task is to provide precise, professional, and helpful answers to employees and customers regarding Zepto company policies.

CRITICAL RULES FOR HALLUCINATION PREVENTION:
1. Answer ONLY using the facts present in the retrieved context below.
2. If the context does not contain sufficient information to answer the question, state: "I could not find explicit details regarding this query in the official Zepto policy documents."
3. Do NOT make up rules, dates, or contact emails that are not in the context.
4. Always cite your source policy document names at the end of your response.

RETRIEVED CONTEXT:
------------------
{context}
------------------

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}
ANSWER:"""


class HeuristicRAGLLM:
    """Smart fallback generation engine for offline/keyless testing."""

    def generate_grounded_answer(self, query: str, context_chunks: List[Document]) -> str:
        if not context_chunks:
            return (
                "I could not find explicit details regarding this query in the official Zepto policy documents.\n\n"
                "📌 *Source*: No matching policy chunk found."
            )

        # Synthesize top matched contents
        answer_parts = []
        sources = set()

        for doc in context_chunks:
            content = doc.page_content.strip()
            src = doc.metadata.get("source", "Zepto Policy Document")
            sources.add(src)
            # Simple sentence filtering relevant to query terms
            query_terms = [w.lower() for w in query.split() if len(w) > 3]
            lines = content.split("\n")
            for line in lines:
                if line.strip() and any(term in line.lower() for term in query_terms):
                    if line.strip() not in answer_parts:
                        answer_parts.append(line.strip())

        if not answer_parts:
            # Fallback to presenting the primary excerpt
            primary_text = context_chunks[0].page_content[:350].strip()
            response = f"Based on the official policy documentation:\n\n\"{primary_text}...\"\n"
        else:
            formatted_points = "\n".join([f"- {pt.lstrip('- ')}" for pt in answer_parts[:4]])
            response = f"Here are the relevant details from the official policy:\n\n{formatted_points}\n"

        sources_str = ", ".join(sorted(sources))
        response += f"\n📌 **Sources Cited**: `{sources_str}`"
        return response


class ZeptoSupportChatbot:
    """Enterprise RAG Chatbot managing document retrieval and grounded response generation."""

    def __init__(self):
        self.rag_pipeline = RAGDocumentPipeline()
        self.heuristic_llm = HeuristicRAGLLM()
        self.conversation_history: List[Dict[str, str]] = []
        
        # Ensure knowledge base is initialized
        self.rag_pipeline.build_or_refresh_knowledge_base()

    def _format_history(self) -> str:
        if not self.conversation_history:
            return "None"
        formatted = []
        for msg in self.conversation_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def _call_gemini(self, prompt: str) -> Optional[str]:
        if not GEMINI_API_KEY:
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API Call Exception ({e}). Falling back to heuristic LLM.")
            return None

    def ask(self, query: str) -> Dict[str, any]:
        """Main chat query endpoint."""
        # 1. Retrieve Relevant Chunks
        chunks = self.rag_pipeline.retrieve_relevant_context(query)

        # 2. Extract Context Text & Sources
        context_text = "\n\n".join([c.page_content for c in chunks])
        sources = list(set([c.metadata.get("source", "Policy Document") for c in chunks]))

        # 3. Format Prompt
        history_text = self._format_history()
        prompt = SYSTEM_PROMPT.format(
            context=context_text if context_text else "No relevant context found.",
            history=history_text,
            question=query
        )

        # 4. Generate Response
        answer_text = None
        provider_used = "Heuristic Fallback Engine (Offline)"

        if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
            answer_text = self._call_gemini(prompt)
            if answer_text:
                provider_used = "Google Gemini Pro/Flash API"

        if not answer_text:
            answer_text = self.heuristic_llm.generate_grounded_answer(query, chunks)

        # 5. Record History
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer_text})

        return {
            "query": query,
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks_count": len(chunks),
            "llm_provider": provider_used,
        }

    def clear_history(self) -> None:
        self.conversation_history = []


if __name__ == "__main__":
    bot = ZeptoSupportChatbot()
    res = bot.ask("What is the refund policy for damaged perishable items?")
    print("Chatbot Response:\n", res["answer"])
    print("\nMetadata:", {k: v for k, v in res.items() if k != "answer"})

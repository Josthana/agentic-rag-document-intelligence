from typing import TypedDict

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.config import Settings
from app.models import (
    RAGResult,
    RetrievedChunk,
)
from app.rag import (
    OllamaClient,
    RAGService,
)
from app.retriever import HybridRetriever


# =========================================================
# GRAPH STATE
# =========================================================

class AgentState(TypedDict, total=False):
    """
    State passed between LangGraph nodes.
    """

    # Original user question.
    query: str

    # Query currently being used for retrieval.
    # This may change if the agent rewrites the query.
    current_query: str

    # Retrieved document chunks.
    sources: list[RetrievedChunk]

    # Final generated answer.
    answer: str

    # Number of retrieval attempts.
    attempts: int

    # Query that finally produced the context.
    used_query: str


# =========================================================
# AGENTIC RAG SERVICE
# =========================================================

class AgenticRAGService:
    """
    Agentic RAG pipeline implemented with LangGraph.

    Flow:

        User question
              ↓
          Retrieve
              ↓
      Enough context?
         /         \
       yes          no
        ↓            ↓
     Generate     Rewrite query
        ↓            ↓
       END         Retrieve again
    """

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
    ):
        self.settings = settings
        self.retriever = retriever

        # Normal RAG service used for final generation.
        self.rag_service = RAGService(
            settings=settings,
            retriever=retriever,
        )

        # Direct Ollama access used for query rewriting.
        self.llm = OllamaClient(
            settings
        )

        # Maximum total retrieval attempts.
        self.max_attempts = 2

        # Minimum hybrid score we consider useful.
        #
        # Hybrid scores are approximately 0-1.
        self.minimum_context_score = 0.20

        # Compile LangGraph.
        self.graph = self._build_graph()

    # =====================================================
    # NODE 1 — RETRIEVE
    # =====================================================

    def _retrieve_node(
        self,
        state: AgentState,
    ) -> dict:
        """
        Retrieve document chunks using HybridRetriever.
        """

        query = state.get(
            "current_query"
        ) or state["query"]

        attempts = (
            state.get("attempts", 0)
            + 1
        )

        print(
            f"\n[Agent] Retrieval attempt "
            f"{attempts}"
        )

        print(
            f"[Agent] Search query: {query}"
        )

        sources = self.retriever.retrieve(
            query=query,
            top_k=self.settings.final_top_k,
        )

        print(
            f"[Agent] Retrieved "
            f"{len(sources)} chunk(s)."
        )

        if sources:
            print(
                "[Agent] Best retrieval score:",
                f"{sources[0].score:.4f}",
            )

        return {
            "sources": sources,
            "attempts": attempts,
            "used_query": query,
        }

    # =====================================================
    # ROUTER — CHECK CONTEXT
    # =====================================================

    def _route_after_retrieval(
        self,
        state: AgentState,
    ) -> str:
        """
        Decide whether retrieved context is strong enough.

        Possible routes:
            generate
            rewrite
            no_context
        """

        sources = state.get(
            "sources",
            []
        )

        attempts = state.get(
            "attempts",
            0,
        )

        # No sources at all.
        if not sources:

            if attempts < self.max_attempts:

                print(
                    "[Agent] No useful context. "
                    "Rewriting query..."
                )

                return "rewrite"

            print(
                "[Agent] Retrieval failed "
                "after maximum attempts."
            )

            return "no_context"

        best_score = sources[0].score

        # Good enough context.
        if (
            best_score
            >= self.minimum_context_score
        ):

            print(
                "[Agent] Context is sufficient."
            )

            return "generate"

        # Weak context, but another attempt is allowed.
        if attempts < self.max_attempts:

            print(
                "[Agent] Context score is weak. "
                "Rewriting query..."
            )

            return "rewrite"

        print(
            "[Agent] Context remained weak "
            "after maximum attempts."
        )

        return "no_context"

    # =====================================================
    # NODE 2 — REWRITE QUERY
    # =====================================================

    def _rewrite_query_node(
        self,
        state: AgentState,
    ) -> dict:
        """
        Ask Ollama to rewrite the question into
        a better retrieval query.
        """

        original_query = state["query"]

        current_query = state.get(
            "current_query",
            original_query,
        )

        prompt = f"""
You are improving a search query for a document
retrieval system.

Original user question:

{original_query}

Current search query:

{current_query}

Rewrite it as a short and precise search query.

Rules:
1. Preserve the original meaning.
2. Include important keywords.
3. Do not answer the question.
4. Return only the rewritten query.
""".strip()

        rewritten_query = self.llm.chat(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

        rewritten_query = (
            rewritten_query
            .strip()
            .strip('"')
        )

        print(
            "[Agent] Rewritten query:",
            rewritten_query,
        )

        return {
            "current_query": rewritten_query,
        }

    # =====================================================
    # NODE 3 — GENERATE ANSWER
    # =====================================================

    def _generate_node(
        self,
        state: AgentState,
    ) -> dict:
        """
        Generate the final grounded answer using
        retrieved document chunks.
        """

        print(
            "[Agent] Generating grounded answer..."
        )

        result = (
            self.rag_service.generate_from_context(
                query=state["query"],
                sources=state.get(
                    "sources",
                    [],
                ),
                used_query=state.get(
                    "used_query"
                ),
            )
        )

        return {
            "answer": result.answer,
            "sources": result.sources,
            "used_query": result.used_query,
        }

    # =====================================================
    # NODE 4 — NO CONTEXT
    # =====================================================

    def _no_context_node(
        self,
        state: AgentState,
    ) -> dict:
        """
        Return a safe response when retrieval
        cannot find sufficiently relevant context.
        """

        answer = (
            "I could not find enough relevant "
            "information in the indexed documents "
            "to answer this question reliably."
        )

        return {
            "answer": answer,
            "sources": [],
        }

    # =====================================================
    # BUILD LANGGRAPH
    # =====================================================

    def _build_graph(self):
        """
        Construct and compile the LangGraph workflow.
        """

        workflow = StateGraph(
            AgentState
        )

        # Register nodes.
        workflow.add_node(
            "retrieve",
            self._retrieve_node,
        )

        workflow.add_node(
            "rewrite",
            self._rewrite_query_node,
        )

        workflow.add_node(
            "generate",
            self._generate_node,
        )

        workflow.add_node(
            "no_context",
            self._no_context_node,
        )

        # ---------------------------------------------
        # Start
        # ---------------------------------------------

        workflow.add_edge(
            START,
            "retrieve",
        )

        # ---------------------------------------------
        # Conditional decision after retrieval
        # ---------------------------------------------

        workflow.add_conditional_edges(
            "retrieve",
            self._route_after_retrieval,
            {
                "generate": "generate",
                "rewrite": "rewrite",
                "no_context": "no_context",
            },
        )

        # ---------------------------------------------
        # Rewritten query goes back to retrieval
        # ---------------------------------------------

        workflow.add_edge(
            "rewrite",
            "retrieve",
        )

        # ---------------------------------------------
        # Final states
        # ---------------------------------------------

        workflow.add_edge(
            "generate",
            END,
        )

        workflow.add_edge(
            "no_context",
            END,
        )

        return workflow.compile()

    # =====================================================
    # PUBLIC API
    # =====================================================

    def answer(
        self,
        query: str,
    ) -> RAGResult:
        """
        Run the complete agentic RAG workflow.
        """

        initial_state: AgentState = {
            "query": query,
            "current_query": query,
            "sources": [],
            "answer": "",
            "attempts": 0,
            "used_query": query,
        }

        final_state = self.graph.invoke(
            initial_state
        )

        return RAGResult(
            query=query,
            answer=final_state.get(
                "answer",
                "",
            ),
            sources=final_state.get(
                "sources",
                [],
            ),
            used_query=final_state.get(
                "used_query",
                query,
            ),
        )
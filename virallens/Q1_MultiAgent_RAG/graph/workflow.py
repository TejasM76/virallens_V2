"""
graph/workflow.py — LangGraph StateGraph definition.

Flow:
  retrieve → grade → [rewrite | generate] → validate → [regenerate | END]

Conditional edges implement the retry loops for:
  1. No relevant chunks found → rewrite query → re-retrieve
  2. Hallucination detected   → regenerate answer
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from groq import AsyncGroq

from graph.state import AgentState
from agents.retriever_agent import RetrieverAgent
from agents.grader_agent import GraderAgent
from agents.generator_agent import GeneratorAgent
from agents.validator_agent import ValidatorAgent
from utils.vector_store import HybridVectorStore
from utils.rate_limiter import with_retry
from config import settings

logger = logging.getLogger(__name__)


# ── Query Rewriter ────────────────────────────────────────────────────────────

REWRITER_PROMPT = """You are a query optimizer for a RAG system over research papers.
The original query returned no relevant results. Rewrite it to be more specific and use
domain terminology that might appear in academic papers.
Return ONLY the rewritten query, nothing else."""


@with_retry(max_retries=3)
async def rewrite_query_node(state: AgentState, client: AsyncGroq) -> dict:
    """Rewrites the query when grading finds no relevant chunks."""
    query = state["query"]
    logger.info(f"[QueryRewriter] Rewriting query: '{query}'")
    response = await client.chat.completions.create(
        model=settings.grader_model,  # Use fast model for rewriting
        messages=[
            {"role": "system", "content": REWRITER_PROMPT},
            {"role": "user", "content": f"Original query: {query}"},
        ],
        temperature=0.3,
        max_tokens=100,
    )
    rewritten = response.choices[0].message.content.strip()
    logger.info(f"[QueryRewriter] Rewritten: '{rewritten}'")
    return {"rewritten_query": rewritten}


# ── Conditional Edge Functions ────────────────────────────────────────────────

def should_rewrite_or_generate(
    state: AgentState,
) -> Literal["rewrite_query", "generate"]:
    """After grading: rewrite if no chunks passed, else generate."""
    graded = state.get("graded_chunks", [])
    attempts = state.get("retrieval_attempts", 0)

    if not graded and attempts < settings.max_retrieval_retries:
        logger.info("No relevant chunks — routing to query rewriter")
        return "rewrite_query"

    return "generate"


def should_regenerate_or_end(
    state: AgentState,
) -> Literal["generate", "end"]:
    """After validation: regenerate if hallucination detected, else end."""
    is_grounded = state.get("is_grounded", False)
    gen_attempts = state.get("generation_attempts", 0)

    if not is_grounded and gen_attempts < settings.max_generation_retries:
        logger.info("Hallucination detected — routing to generator for retry")
        return "generate"

    return "end"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_workflow(vector_store: HybridVectorStore, groq_client: AsyncGroq) -> StateGraph:
    """
    Construct and compile the LangGraph workflow.

    Returns a compiled graph that can be invoked with:
        result = await graph.ainvoke(initial_state)
    """
    retriever = RetrieverAgent(vector_store)
    grader = GraderAgent(groq_client)
    generator = GeneratorAgent(groq_client)
    validator = ValidatorAgent(groq_client)

    # Wrap the rewriter to inject the client (LangGraph nodes are single-arg)
    async def rewrite_node(state):
        return await rewrite_query_node(state, groq_client)

    # ── Build graph ───────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve", retriever)
    graph.add_node("grade", grader)
    graph.add_node("rewrite_query", rewrite_node)
    graph.add_node("generate", generator)
    graph.add_node("validate", validator)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Fixed edges
    graph.add_edge("retrieve", "grade")
    graph.add_edge("rewrite_query", "retrieve")  # Loop back after rewrite
    graph.add_edge("generate", "validate")

    # Conditional edges
    graph.add_conditional_edges(
        "grade",
        should_rewrite_or_generate,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )
    graph.add_conditional_edges(
        "validate",
        should_regenerate_or_end,
        {"generate": "generate", "end": END},
    )

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully")
    return compiled


def get_initial_state(query: str) -> AgentState:
    """Construct a clean initial state for a new query."""
    return AgentState(
        query=query,
        rewritten_query=None,
        retrieved_chunks=[],
        graded_chunks=[],
        answer="",
        citations=[],
        is_grounded=False,
        validation_feedback=None,
        retrieval_attempts=0,
        generation_attempts=0,
        error=None,
    )

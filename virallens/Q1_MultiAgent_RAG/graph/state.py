"""
graph/state.py — Shared state that flows through the LangGraph pipeline.

Using TypedDict + Annotated so LangGraph knows how to merge fields
when multiple nodes update the same key.
"""

from typing import Annotated, List, Optional
from typing_extensions import TypedDict
import operator


class DocumentChunk(TypedDict):
    """A single retrieved chunk with metadata."""
    content: str
    source: str          # filename
    page: int
    section: str         # section header if detected
    chunk_id: str
    relevance_score: float  # set by GraderAgent (0.0–1.0)


class AgentState(TypedDict):
    """
    The single source of truth shared across all agents in the graph.

    LangGraph merges list fields using the `operator.add` reducer,
    meaning each agent *appends* to lists rather than overwriting them.
    Scalar fields are simply overwritten.
    """
    # ── Input ────────────────────────────────────────────────────────────
    query: str                              # Original user query
    rewritten_query: Optional[str]          # Set by query rewriter if needed

    # ── Retrieval ────────────────────────────────────────────────────────
    retrieved_chunks: Annotated[List[DocumentChunk], operator.add]
    graded_chunks: List[DocumentChunk]      # Filtered by GraderAgent

    # ── Generation ───────────────────────────────────────────────────────
    answer: str                             # Generated answer
    citations: List[str]                    # Source references

    # ── Validation ───────────────────────────────────────────────────────
    is_grounded: bool                       # ValidatorAgent verdict
    validation_feedback: Optional[str]      # Why it failed (if it did)

    # ── Control ──────────────────────────────────────────────────────────
    retrieval_attempts: int                 # Number of retrieval retries
    generation_attempts: int                # Number of generation retries
    error: Optional[str]                    # Any fatal error message

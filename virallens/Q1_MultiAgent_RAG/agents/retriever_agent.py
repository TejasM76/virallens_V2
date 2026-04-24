"""
agents/retriever_agent.py — Hybrid retrieval with intelligent metadata filtering.

Two-phase retrieval strategy
─────────────────────────────
Phase 1 — Intent-aware metadata filter extraction
  Parse the query for signals that narrow the candidate pool:
    • "in the methods section"   → section_contains="Method"
    • "paper about BERT / GPT"   → source_in={papers mentioning that term}
    • "on page 5"                → page_min=5, page_max=5
  This eliminates false positives from irrelevant papers/sections
  BEFORE any vector operations — directly reducing hallucination risk.

Phase 2 — Hybrid search on filtered pool
  Dense (FAISS) + Sparse (BM25) → Reciprocal Rank Fusion → top-K chunks
  Each returned chunk carries rrf_score, dense_score, bm25_score for
  full traceability in logs and GraderAgent input.
"""

import logging
import re
from typing import Any, Optional

from graph.state import AgentState, DocumentChunk
from utils.vector_store import HybridVectorStore, MetadataFilter
from config import settings

logger = logging.getLogger(__name__)


# Section keywords → canonical section names (substring match)
_SECTION_SIGNALS: dict[str, str] = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "background": "Background",
    "method": "Method",        # matches Methodology, Methods, Method
    "experiment": "Experiment", # matches Experiments, Experimental
    "result": "Result",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "appendix": "Appendix",
}

_PAGE_RE = re.compile(r"\b(?:page|p\.?)\s*(\d+)\b", re.IGNORECASE)


def _extract_metadata_filter(
    query: str,
    available_sources: list[str],
    available_sections: list[str],
) -> Optional[MetadataFilter]:
    """
    Lightweight rule-based filter extraction from natural language queries.

    Intentionally conservative: only applies a filter when there is a
    high-confidence signal. Ambiguous queries run unfiltered (= full corpus).

    Returns None if no filter is warranted.
    """
    q_lower = query.lower()

    section_filter: Optional[str] = None
    source_filter: Optional[set] = None
    page_filter: Optional[tuple] = None

    # ── Section signal ────────────────────────────────────────────────────
    for keyword, section_name in _SECTION_SIGNALS.items():
        if keyword in q_lower:
            # Only apply if that section actually exists in the corpus
            matching_sections = [s for s in available_sections if section_name.lower() in s.lower()]
            if matching_sections:
                section_filter = section_name
                logger.debug(f"Filter: section_contains='{section_name}' (matched '{keyword}')")
            break

    # ── Source signal (paper-specific queries) ────────────────────────────
    # If the query mentions a filename fragment or key word that maps to one paper
    for source in available_sources:
        stem = source.replace(".pdf", "").replace("_", " ").replace("-", " ").lower()
        # Check if ≥2 words from the paper stem appear in the query
        stem_words = [w for w in stem.split() if len(w) > 3]
        matches = sum(1 for w in stem_words if w in q_lower)
        if matches >= 2 or (len(stem_words) == 1 and stem_words[0] in q_lower):
            source_filter = source_filter or set()
            source_filter.add(source)
            logger.debug(f"Filter: source='{source}' (matched stem words)")

    # ── Page signal ───────────────────────────────────────────────────────
    page_match = _PAGE_RE.search(query)
    if page_match:
        page_num = int(page_match.group(1))
        page_filter = (page_num, page_num)
        logger.debug(f"Filter: page={page_num}")

    # ── Build filter object (only if at least one constraint) ─────────────
    if not any([section_filter, source_filter, page_filter]):
        return None

    return MetadataFilter(
        source_in=source_filter,
        section_contains=section_filter,
        page_min=page_filter[0] if page_filter else None,
        page_max=page_filter[1] if page_filter else None,
    )


class RetrieverAgent:
    """
    Hybrid retriever agent with metadata-aware filtering.

    Injected with the shared HybridVectorStore instance.
    All state lives in AgentState — this class is stateless.
    """

    def __init__(self, vector_store: HybridVectorStore):
        self._vs = vector_store

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """LangGraph node — receives full state, returns state update dict."""
        query = state.get("rewritten_query") or state["query"]
        attempts = state.get("retrieval_attempts", 0)

        logger.info(f"[RetrieverAgent] Query: '{query[:80]}' (attempt {attempts + 1})")

        # ── Phase 1: Extract metadata filter ──────────────────────────────
        metadata_filter = _extract_metadata_filter(
            query,
            self._vs.available_sources(),
            self._vs.available_sections(),
        )
        if metadata_filter:
            logger.info(f"[RetrieverAgent] Metadata filter applied: {metadata_filter}")
        else:
            logger.info("[RetrieverAgent] No metadata filter — searching full corpus")

        # ── Phase 2: Hybrid search ─────────────────────────────────────────
        raw_results = self._vs.search(
            query,
            top_k=settings.top_k_retrieval,
            metadata_filter=metadata_filter,
        )

        # Log score breakdown for each result
        for r in raw_results:
            logger.debug(
                f"  [{r['source']} p{r['page']}] "
                f"rrf={r.get('rrf_score', 0):.4f} "
                f"dense={r.get('dense_score', 0):.4f} "
                f"bm25={r.get('bm25_score', 0):.4f} "
                f"| {r['content'][:60]}…"
            )

        # ── Convert to DocumentChunk typed dicts ──────────────────────────
        chunks: list[DocumentChunk] = [
            DocumentChunk(
                content=r["content"],
                source=r["source"],
                page=r["page"],
                section=r.get("section", "Unknown"),
                chunk_id=r["chunk_id"],
                relevance_score=0.0,  # Set by GraderAgent
            )
            for r in raw_results
        ]

        logger.info(
            f"[RetrieverAgent] Retrieved {len(chunks)} chunks "
            f"(dense+BM25 hybrid, filter={'yes' if metadata_filter else 'no'})"
        )
        return {
            "retrieved_chunks": chunks,
            "retrieval_attempts": attempts + 1,
        }

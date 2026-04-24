"""
agents/grader_agent.py — Scores each retrieved chunk for relevance to the query.

Uses llama3-8b (fast, free on Groq) to score chunks 0.0–1.0.
Chunks below `relevance_threshold` are filtered out.

If too few chunks pass grading, signals the graph to rewrite the query.
"""

import json
import logging
from typing import Any

from groq import AsyncGroq

from graph.state import AgentState, DocumentChunk
from utils.rate_limiter import with_retry
from config import settings

logger = logging.getLogger(__name__)

GRADER_SYSTEM_PROMPT = """You are a relevance grader for a RAG system working with research papers.
Given a user query and a document chunk, output ONLY a JSON object with:
  - "score": float between 0.0 (completely irrelevant) and 1.0 (highly relevant)
  - "reason": one sentence explaining the score

Respond with raw JSON only — no markdown, no explanation outside the JSON."""

GRADER_USER_TEMPLATE = """Query: {query}

Document chunk:
\"\"\"
{chunk}
\"\"\"

Rate the relevance of this chunk to the query."""


class GraderAgent:
    """Grades retrieved chunks for query relevance using an LLM."""

    def __init__(self, client: AsyncGroq):
        self._client = client

    @with_retry(max_retries=5, base_delay=2.0)
    async def _grade_chunk(self, query: str, chunk_content: str) -> tuple[float, str]:
        """Returns (score, reason) for a single chunk."""
        response = await self._client.chat.completions.create(
            model=settings.grader_model,
            messages=[
                {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": GRADER_USER_TEMPLATE.format(
                        query=query, chunk=chunk_content[:800]  # Truncate to save tokens
                    ),
                },
            ],
            temperature=0.0,   # Deterministic grading
            max_tokens=80,
        )
        raw = response.choices[0].message.content.strip()
        try:
            data = json.loads(raw)
            return float(data.get("score", 0.0)), data.get("reason", "")
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Grader returned non-JSON: {raw[:100]}")
            return 0.0, "parse error"

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        """Grade all retrieved chunks and filter by threshold."""
        query = state.get("rewritten_query") or state["query"]
        retrieved = state.get("retrieved_chunks", [])

        logger.info(f"[GraderAgent] Grading {len(retrieved)} chunks")

        graded: list[DocumentChunk] = []
        for chunk in retrieved:
            score, reason = await self._grade_chunk(query, chunk["content"])
            if score >= settings.relevance_threshold:
                graded_chunk = DocumentChunk(**chunk)
                graded_chunk["relevance_score"] = score
                graded.append(graded_chunk)
                logger.debug(f"  ✓ chunk {chunk['chunk_id']} score={score:.2f}: {reason}")
            else:
                logger.debug(f"  ✗ chunk {chunk['chunk_id']} score={score:.2f} (filtered)")

        # Sort by relevance descending, keep top_k_after_grading
        graded.sort(key=lambda c: c["relevance_score"], reverse=True)
        graded = graded[: settings.top_k_after_grading]

        logger.info(f"[GraderAgent] {len(graded)} chunks passed grading")
        return {"graded_chunks": graded}

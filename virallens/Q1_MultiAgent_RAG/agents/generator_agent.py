"""
agents/generator_agent.py — Synthesizes a grounded answer from graded chunks.

Uses llama3-70b (best quality on Groq free tier) for answer generation.
Includes inline citations referencing source filenames and page numbers.
Falls back to query rewrite suggestion if context is too sparse.
"""

import logging
from typing import Any

from groq import AsyncGroq

from graph.state import AgentState
from utils.rate_limiter import with_retry
from config import settings

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = """You are a research assistant answering questions based strictly on provided context.

Rules:
1. Answer ONLY from the provided context chunks. Do NOT use external knowledge.
2. After each factual claim, add an inline citation like [Source: filename, p.N].
3. If the context is insufficient to answer, say "The provided documents do not contain enough information to answer this question." — do not hallucinate.
4. Be concise but complete. Prefer bullet points for multi-part answers.
5. Structure: Answer → Key Points (if applicable) → Limitations of provided context."""

GENERATOR_USER_TEMPLATE = """Question: {query}

Context (ranked by relevance):
{context}

Please provide a well-structured, cited answer."""


def _format_context(graded_chunks: list) -> str:
    """Format graded chunks into a numbered context block for the LLM."""
    lines = []
    for i, chunk in enumerate(graded_chunks, 1):
        lines.append(
            f"[{i}] Source: {chunk['source']}, Page {chunk['page']}, "
            f"Section: {chunk.get('section', 'N/A')} (relevance: {chunk['relevance_score']:.2f})\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(lines)


class GeneratorAgent:
    """Generates a grounded, cited answer from graded context chunks."""

    def __init__(self, client: AsyncGroq):
        self._client = client

    @with_retry(max_retries=5, base_delay=2.0)
    async def _generate(self, query: str, context: str) -> str:
        response = await self._client.chat.completions.create(
            model=settings.generator_model,
            messages=[
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": GENERATOR_USER_TEMPLATE.format(
                        query=query, context=context
                    ),
                },
            ],
            temperature=0.2,  # Low temp for factual grounding
            max_tokens=900,
        )
        return response.choices[0].message.content.strip()

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        graded_chunks = state.get("graded_chunks", [])
        gen_attempts = state.get("generation_attempts", 0)

        logger.info(
            f"[GeneratorAgent] Generating answer from {len(graded_chunks)} chunks "
            f"(attempt {gen_attempts + 1})"
        )

        context = _format_context(graded_chunks)
        answer = await self._generate(query, context)

        # Extract citation references from chunks
        citations = list(
            {f"{c['source']} (p.{c['page']})" for c in graded_chunks}
        )

        logger.info("[GeneratorAgent] Answer generated")
        return {
            "answer": answer,
            "citations": citations,
            "generation_attempts": gen_attempts + 1,
        }

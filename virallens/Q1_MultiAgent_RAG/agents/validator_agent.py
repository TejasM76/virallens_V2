"""
agents/validator_agent.py — Validates the generated answer for factual grounding.

Checks that every claim in the answer can be traced to the provided context chunks.
Uses NLI-style prompting to detect hallucinations.

Returns:
  - is_grounded: bool
  - validation_feedback: string explaining any issues found
"""

import json
import logging
from typing import Any

from groq import AsyncGroq

from graph.state import AgentState
from utils.rate_limiter import with_retry
from config import settings

logger = logging.getLogger(__name__)

VALIDATOR_SYSTEM_PROMPT = """You are a factual consistency checker for a RAG system.

Given an answer and the source context it was generated from, determine whether every factual claim in the answer is supported by the context.

Output ONLY a JSON object:
{
  "is_grounded": true/false,
  "unsupported_claims": ["claim 1", "claim 2"],
  "feedback": "One sentence summary of issues, or 'Answer is fully grounded.'"
}

Be strict: if any claim goes beyond what the context says, mark is_grounded as false."""

VALIDATOR_USER_TEMPLATE = """Answer to validate:
\"\"\"
{answer}
\"\"\"

Source context:
\"\"\"
{context}
\"\"\""""


def _format_context_short(graded_chunks: list) -> str:
    """Compact context for validator (saves tokens)."""
    return "\n\n".join(
        f"[{c['source']}, p.{c['page']}]: {c['content'][:400]}"
        for c in graded_chunks
    )


class ValidatorAgent:
    """Validates factual grounding of the generated answer."""

    def __init__(self, client: AsyncGroq):
        self._client = client

    @with_retry(max_retries=5, base_delay=2.0)
    async def _validate(self, answer: str, context: str) -> tuple[bool, str]:
        response = await self._client.chat.completions.create(
            model=settings.validator_model,
            messages=[
                {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": VALIDATOR_USER_TEMPLATE.format(
                        answer=answer, context=context
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        try:
            data = json.loads(raw)
            return bool(data.get("is_grounded", False)), data.get(
                "feedback", "No feedback"
            )
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Validator returned non-JSON: {raw[:100]}")
            # Be conservative: assume not grounded on parse failure
            return False, f"Could not parse validator response: {raw[:80]}"

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        answer = state.get("answer", "")
        graded_chunks = state.get("graded_chunks", [])

        logger.info("[ValidatorAgent] Validating answer grounding")

        context = _format_context_short(graded_chunks)
        is_grounded, feedback = await self._validate(answer, context)

        if is_grounded:
            logger.info("[ValidatorAgent] ✓ Answer is grounded")
        else:
            logger.warning(f"[ValidatorAgent] ✗ Hallucination detected: {feedback}")

        return {
            "is_grounded": is_grounded,
            "validation_feedback": feedback,
        }

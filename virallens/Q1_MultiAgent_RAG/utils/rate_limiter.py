"""
utils/rate_limiter.py — Token-bucket rate limiter for Groq free-tier API.

Groq free tier (as of 2025):
  - 30 requests / minute
  - 14,400 tokens / minute

Strategy:
  1. Token-bucket to track request + token budgets
  2. Exponential backoff with full jitter on 429 responses
  3. Thread-safe via asyncio.Lock for async callers
"""

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Callable, Any

from groq import RateLimitError

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Dual-rate token bucket: one bucket for request count,
    one for token count. Both must have capacity for a call to proceed.
    """

    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute

        self._req_tokens = float(requests_per_minute)
        self._tok_tokens = float(tokens_per_minute)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self):
        """Add tokens proportional to elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._req_tokens = min(
            self.rpm, self._req_tokens + elapsed * (self.rpm / 60.0)
        )
        self._tok_tokens = min(
            self.tpm, self._tok_tokens + elapsed * (self.tpm / 60.0)
        )
        self._last_refill = now

    async def acquire(self, estimated_tokens: int = 500):
        """
        Block until there is capacity for this request.
        estimated_tokens: rough token budget for the call (prompt + completion).
        """
        async with self._lock:
            while True:
                self._refill()
                if self._req_tokens >= 1.0 and self._tok_tokens >= estimated_tokens:
                    self._req_tokens -= 1.0
                    self._tok_tokens -= estimated_tokens
                    return
                # Calculate wait: whichever bucket is more starved
                req_wait = (1.0 - self._req_tokens) / (self.rpm / 60.0)
                tok_wait = max(0, (estimated_tokens - self._tok_tokens)) / (self.tpm / 60.0)
                wait = max(req_wait, tok_wait) + 0.05  # small buffer
                logger.debug(f"Rate limit: sleeping {wait:.2f}s")
                await asyncio.sleep(wait)


# Module-level singleton — shared across all agents
_bucket: TokenBucket | None = None


def init_rate_limiter(requests_per_minute: int, tokens_per_minute: int):
    global _bucket
    _bucket = TokenBucket(requests_per_minute, tokens_per_minute)


def get_bucket() -> TokenBucket:
    if _bucket is None:
        raise RuntimeError("Call init_rate_limiter() before using the rate limiter")
    return _bucket


def with_retry(
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
):
    """
    Decorator that retries on Groq RateLimitError with exponential backoff + jitter.

    Usage:
        @with_retry(max_retries=5)
        async def call_llm(...):
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries + 1):
                try:
                    await get_bucket().acquire()
                    return await fn(*args, **kwargs)
                except RateLimitError as e:
                    if attempt == max_retries:
                        logger.error(f"Max retries reached for {fn.__name__}: {e}")
                        raise
                    # Full jitter: sleep random in [0, min(cap, base * 2^attempt)]
                    sleep = random.uniform(0, min(max_delay, base_delay * (2**attempt)))
                    logger.warning(
                        f"RateLimitError on attempt {attempt + 1}/{max_retries} "
                        f"for {fn.__name__}. Retrying in {sleep:.1f}s…"
                    )
                    await asyncio.sleep(sleep)
                except Exception as e:
                    logger.error(f"Unexpected error in {fn.__name__}: {e}")
                    raise

        return wrapper

    return decorator

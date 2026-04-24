"""
config.py — Centralized configuration for the Multi-Agent RAG pipeline.
All tuneable parameters live here. No magic numbers scattered across files.

Data directory is at the PROJECT ROOT (../data from q1_multiagent_rag/).
Both Q1 and Q2 share the same PDF folder.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

# Root of the project = parent of q1_multiagent_rag/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── API Keys ────────────────────────────────────────────────────────────
    groq_api_key:        str = Field(default="", env="GROQ_API_KEY")
    openrouter_api_key:  str = Field(default="", env="OPENROUTER_API_KEY")
    gemini_api_key:      str = Field(default="", env="GEMINI_API_KEY")

    # ── Models ──────────────────────────────────────────────────────────────
    generator_model: str = "llama3-70b-8192"
    grader_model:    str = "llama3-8b-8192"
    validator_model: str = "llama3-8b-8192"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Chunking ────────────────────────────────────────────────────────────
    chunk_size:               int   = 512
    chunk_overlap:            int   = 64
    semantic_merge_threshold: float = 0.85
    min_chunk_length:         int   = 50

    # ── Retrieval ───────────────────────────────────────────────────────────
    top_k_retrieval:    int   = 6
    top_k_after_grading:int   = 4
    relevance_threshold:float = 0.6

    # ── Hybrid Search (ChromaDB + BM25 + RRF) ───────────────────────────────
    rrf_k:              int   = 60    # RRF dampening constant
    rrf_fetch_multiplier:int  = 3     # Over-fetch per leg before fusion
    dense_weight:       float = 0.6   # Logged only (RRF is rank-based)
    sparse_weight:      float = 0.4

    # ── Graph Control ───────────────────────────────────────────────────────
    max_retrieval_retries:  int = 3
    max_generation_retries: int = 2

    # ── Rate Limiting (Groq free tier) ──────────────────────────────────────
    requests_per_minute: int   = 28
    tokens_per_minute:   int   = 13000
    retry_base_delay:    float = 2.0
    retry_max_delay:     float = 60.0

    # ── Paths (relative to PROJECT ROOT, not q1 subfolder) ──────────────────
    data_dir:         Path = _PROJECT_ROOT / "data"
    vector_store_dir: Path = _PROJECT_ROOT / ".vectorstore"
    log_level:        str  = "INFO"

    class Config:
        # Look for .env in the q1 folder first, then project root
        env_file = [
            str(Path(__file__).parent / ".env"),
            str(_PROJECT_ROOT / ".env"),
        ]
        env_file_encoding = "utf-8"


settings = Settings()

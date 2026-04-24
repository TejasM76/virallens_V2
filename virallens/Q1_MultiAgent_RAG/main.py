"""
main.py — Q1 Multi-Agent RAG pipeline entry point.

Can be run directly:
    cd q1_multiagent_rag
    python main.py --query "What are the main findings?"

Or via root launcher (recommended):
    cd ..   (project root)
    python run.py q1 --query "What are the main findings?"
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from groq import AsyncGroq

from config import settings
from graph.workflow import build_workflow, get_initial_state
from utils.chunker import HybridChunker
from utils.rate_limiter import init_rate_limiter
from utils.vector_store import ChromaVectorStore

# ── Logging ─────────────────────────────────────────────────────────────────
# Use UTF-8 stream handler so Unicode chars work on Windows (CP1252 would crash)
_stream_handler = logging.StreamHandler(stream=open(
    sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, closefd=False
) if hasattr(sys.stdout, 'fileno') else sys.stdout)
_stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        _stream_handler,
        logging.FileHandler(Path(__file__).parent / "pipeline.log", encoding="utf-8"),
    ],
)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def setup_pipeline(rebuild_index: bool = False):
    """
    One-time setup:
      1. Chunk PDFs into ChromaDB  (or load cached index)
      2. Initialise Groq client + rate limiter
      3. Compile LangGraph workflow
    """
    init_rate_limiter(settings.requests_per_minute, settings.tokens_per_minute)

    vs = ChromaVectorStore()
    persist_dir = settings.vector_store_dir

    if not rebuild_index and vs.is_persisted(persist_dir):
        logger.info("Loading existing ChromaDB index…")
        vs.load(persist_dir)
    else:
        logger.info("Building ChromaDB index from PDFs…")
        chunker = HybridChunker()
        chunks  = chunker.chunk_directory(settings.data_dir)
        vs.build(chunks, persist_dir=persist_dir)
        vs.save(persist_dir)

    groq_client = AsyncGroq(api_key=settings.groq_api_key)
    graph       = build_workflow(vs, groq_client)
    return graph, groq_client


async def run_query(graph, query: str) -> dict:
    logger.info(f"\n{'='*60}\nQuery: {query}\n{'='*60}")
    try:
        final_state = await graph.ainvoke(get_initial_state(query))
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"error": str(e), "answer": "An error occurred.", "citations": []}

    result = {
        "query":               query,
        "answer":              final_state.get("answer", "No answer generated"),
        "citations":           final_state.get("citations", []),
        "is_grounded":         final_state.get("is_grounded", False),
        "retrieval_attempts":  final_state.get("retrieval_attempts", 0),
        "generation_attempts": final_state.get("generation_attempts", 0),
        "validation_feedback": final_state.get("validation_feedback"),
    }

    print(f"\n{'─'*60}")
    print(f"📋 ANSWER\n{result['answer']}")
    print(f"\n📚 SOURCES\n" + "\n".join(f"  • {c}" for c in result["citations"]))
    print(f"\n✅ Grounded: {result['is_grounded']}")
    if not result["is_grounded"]:
        print(f"⚠️  Feedback: {result['validation_feedback']}")
    print(f"{'─'*60}\n")
    return result


async def interactive_mode(graph):
    print("\n🤖 Multi-Agent RAG — Interactive Mode  (type 'exit' to quit)\n")
    while True:
        query = input("Query: ").strip()
        if query.lower() in ("exit", "quit", ""):
            break
        await run_query(graph, query)


async def main():
    parser = argparse.ArgumentParser(description="Virallens Q1 — Multi-Agent RAG")
    parser.add_argument("--query",         type=str)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--interactive",   action="store_true")
    args = parser.parse_args()

    # API key check
    if not settings.groq_api_key:
        print(
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  ERROR: GROQ_API_KEY is not set.\n\n"
            "  Steps to fix:\n"
            "    1. Get a free key → https://console.groq.com/keys\n"
            "    2. Create  .env  in this folder (or project root) with:\n"
            "         GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx\n"
            "    3. Re-run:  python main.py\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        sys.exit(1)

    # Data check
    pdfs = list(settings.data_dir.glob("*.pdf"))
    if not pdfs:
        logger.error(
            f"No PDFs found in '{settings.data_dir}'.\n"
            "Place your research papers there and re-run."
        )
        sys.exit(1)

    logger.info(f"Data dir: {settings.data_dir}  ({len(pdfs)} PDFs)")
    graph, _ = await setup_pipeline(rebuild_index=args.rebuild_index)

    if args.interactive:
        await interactive_mode(graph)
    elif args.query:
        await run_query(graph, args.query)
    else:
        for q in [
            "What are the main research contributions?",
            "What datasets or benchmarks are used?",
            "What are the key limitations?",
        ]:
            await run_query(graph, q)


if __name__ == "__main__":
    asyncio.run(main())

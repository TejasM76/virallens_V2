"""
run.py  —  Unified launcher for the Virallens AI Engineer Assignment
           Run this from the PROJECT ROOT (where data/ lives).

Usage
─────
  python run.py q1                          # RAG pipeline, demo queries
  python run.py q1 --query "What methods?"  # Single query
  python run.py q1 --interactive            # REPL mode
  python run.py q1 --rebuild-index          # Force re-index PDFs

  python run.py q2                          # Train embedder + evaluate
  python run.py q2 --skip-training          # Evaluate only (use saved model)
  python run.py q2 --epochs 5               # Custom training epochs

  python run.py both                        # Run Q2 first, then Q1 demo

Project layout expected
───────────────────────
  virallens/
  ├── run.py                ← YOU ARE HERE
  ├── .env                  ← add GROQ_API_KEY here
  ├── data/                 ← put your 5 PDF papers here
  │   ├── paper1.pdf
  │   └── ...
  ├── q1_multiagent_rag/
  └── q2_custom_embedder/
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Silence noisy third-party loggers immediately — before any imports trigger them
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# ── Resolve project root ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def _banner(title: str) -> None:
    w = 62
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)


def _check_env() -> bool:
    """Check .env exists and GROQ_API_KEY is set. Returns True if OK."""
    # Try loading .env from root
    env_file = ROOT / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print(
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  ERROR: GROQ_API_KEY is not set.\n\n"
            "  Steps to fix:\n"
            "    1. Get a free key → https://console.groq.com/keys\n"
            "    2. Create  virallens/.env  with:\n"
            "         GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx\n"
            "    3. Re-run:  python run.py q1\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        return False
    return True


def _check_data() -> bool:
    """Check that at least one PDF exists in data/."""
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  ERROR: No PDF files found in  data/\n\n"
            "  Place your 5 research paper PDFs in:\n"
            f"    {DATA_DIR}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        return False
    print(f"  Found {len(pdfs)} PDF(s) in data/:")
    for p in sorted(pdfs):
        print(f"    • {p.name}")
    return True


# ── Q1 runner ───────────────────────────────────────────────────────────────

def run_q1(args: argparse.Namespace) -> None:
    _banner("Q1  —  Multi-Agent RAG  (LangChain + LangGraph + ChromaDB)")

    if not _check_env():
        sys.exit(1)
    if not _check_data():
        sys.exit(1)

    # Add q1 to path so its imports work
    q1_dir = ROOT / "q1_multiagent_rag"
    if str(q1_dir) not in sys.path:
        sys.path.insert(0, str(q1_dir))

    import asyncio

    # Inline async runner so we don't import main at module level
    async def _run():
        from main import setup_pipeline, run_query, interactive_mode

        graph, _ = await setup_pipeline(rebuild_index=args.rebuild_index)

        if args.interactive:
            await interactive_mode(graph)
        elif args.query:
            await run_query(graph, args.query)
        else:
            demo_queries = [
                "What are the main research contributions described in the papers?",
                "What datasets or benchmarks are used in the experiments?",
                "What are the key limitations acknowledged by the authors?",
            ]
            for q in demo_queries:
                await run_query(graph, q)

    asyncio.run(_run())


# ── Q2 runner ───────────────────────────────────────────────────────────────

def run_q2(args: argparse.Namespace) -> None:
    _banner("Q2  —  Custom Domain Embedder  (Triplet fine-tuning + Clustering)")

    if not _check_data():
        sys.exit(1)

    q2_dir = ROOT / "q2_custom_embedder"
    if str(q2_dir) not in sys.path:
        sys.path.insert(0, str(q2_dir))

    from main import main as q2_main
    import argparse as _ap

    # Rebuild args for q2 main, pointing data_dir to root data/
    q2_args = _ap.Namespace(
        data_dir=DATA_DIR,
        skip_training=args.skip_training,
        epochs=args.epochs,
        triplets_per_doc=200,
    )
    q2_main.__globals__["__name__"] = "__not_main__"   # prevent double-execution guard

    # Call q2 directly
    import logging, sys as _sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(_sys.stdout)],
    )

    from utils.preprocessor import load_corpus
    from utils.triplet_generator import generate_triplets, triplets_to_sentence_transformer_format
    from models.custom_embedder import DomainEmbedder
    from evaluate_embedder import evaluate
    from cluster_analysis import run as run_clusters
    from pathlib import Path as _Path

    MODEL_OUTPUT  = str(q2_dir / "outputs" / "domain_embedder")
    EVAL_OUTPUT   = q2_dir / "outputs" / "eval_plots"
    CLUSTER_OUTPUT= q2_dir / "outputs" / "clusters"

    corpus = load_corpus(DATA_DIR)

    if not args.skip_training:
        _banner("Q2 Phase 1 — Training")
        triplets      = generate_triplets(corpus, n_triplets_per_doc=200)
        triplet_tuples= triplets_to_sentence_transformer_format(triplets)
        embedder      = DomainEmbedder()
        embedder.train(triplet_tuples, output_path=MODEL_OUTPUT, epochs=args.epochs)

    _banner("Q2 Phase 2 — Evaluation")
    m_base, m_custom = evaluate(DATA_DIR, MODEL_OUTPUT, EVAL_OUTPUT)

    _banner("Q2 Phase 3 — Cluster Analysis")
    run_clusters(DATA_DIR, MODEL_OUTPUT, CLUSTER_OUTPUT)

    print(f"\n✅  Q2 complete.  Plots saved to: {q2_dir / 'outputs'}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="Virallens Assignment — unified launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py q1                         Run Q1 with demo queries
  python run.py q1 --query "What methods?" Single query
  python run.py q1 --interactive           Interactive REPL
  python run.py q1 --rebuild-index         Force re-index PDFs
  python run.py q2                         Train embedder + full eval
  python run.py q2 --skip-training         Eval only (skip training)
  python run.py both                       Q2 then Q1
        """,
    )

    parser.add_argument(
        "task",
        choices=["q1", "q2", "both"],
        help="Which task to run",
    )

    # Q1 options
    q1_group = parser.add_argument_group("Q1 options")
    q1_group.add_argument("--query",         type=str,  help="Single query for Q1")
    q1_group.add_argument("--interactive",   action="store_true", help="Interactive REPL for Q1")
    q1_group.add_argument("--rebuild-index", action="store_true", help="Force rebuild ChromaDB index")

    # Q2 options
    q2_group = parser.add_argument_group("Q2 options")
    q2_group.add_argument("--skip-training", action="store_true", help="Skip embedder training (Q2)")
    q2_group.add_argument("--epochs",        type=int, default=3,  help="Training epochs for Q2 (default: 3)")

    args = parser.parse_args()

    print(f"\n{'─'*62}")
    print(f"  Virallens AI Engineer Assignment")
    print(f"  Data dir : {DATA_DIR}")
    print(f"  Task     : {args.task.upper()}")
    print(f"{'─'*62}")

    if args.task == "q1":
        run_q1(args)
    elif args.task == "q2":
        run_q2(args)
    elif args.task == "both":
        run_q2(args)
        run_q1(args)


if __name__ == "__main__":
    main()

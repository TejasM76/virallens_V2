"""
train_embedder.py — Train the domain-specific embedder on research paper corpus.

Usage:
    python train_embedder.py --data-dir data/ --epochs 3 --output outputs/domain_embedder
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train domain-specific embedder")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output", type=str, default="outputs/domain_embedder")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--triplets-per-doc", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from utils.preprocessor import load_corpus
    from utils.triplet_generator import generate_triplets, triplets_to_sentence_transformer_format
    from models.custom_embedder import DomainEmbedder

    # ── Step 1: Load corpus ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preprocessing corpus")
    logger.info("=" * 60)
    corpus = load_corpus(args.data_dir)

    # ── Step 2: Generate triplets ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Mining training triplets (BM25 hard negatives)")
    logger.info("=" * 60)
    triplets = generate_triplets(
        corpus,
        n_triplets_per_doc=args.triplets_per_doc,
        random_seed=args.seed,
    )
    triplet_tuples = triplets_to_sentence_transformer_format(triplets)
    logger.info(f"Total triplets: {len(triplet_tuples)}")

    # ── Step 3: Train ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Fine-tuning SentenceTransformer")
    logger.info("=" * 60)
    embedder = DomainEmbedder()
    embedder.train(
        triplets=triplet_tuples,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    logger.info(f"\n✅ Training complete. Model saved to: {args.output}")
    logger.info("Next step: python evaluate_embedder.py")


if __name__ == "__main__":
    main()

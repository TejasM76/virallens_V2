"""
models/custom_embedder.py — Domain-specific embedder fine-tuned via triplet loss.

Base model: all-MiniLM-L6-v2 (22M params, 384-dim)
  - Fast inference
  - Strong baseline for semantic similarity
  - Small enough to fine-tune on a laptop/Colab free tier

Training objective: TripletLoss (margin-based)
  - Pushes anchor closer to positive than negative by margin δ
  - Hard negatives from BM25 mining make this meaningful
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIM = 384  # all-MiniLM-L6-v2 native output — no projection needed
                  # (Note: README says "no need for long embedding lengths" — 384 is fine)


class DomainEmbedder:
    """
    Wrapper around a fine-tuned SentenceTransformer model.

    Supports:
      - Training from triplets
      - Encoding sentences to embeddings
      - Saving / loading
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path and Path(model_path).exists():
            logger.info(f"Loading fine-tuned model from {model_path}")
            self.model = SentenceTransformer(model_path)
        else:
            logger.info(f"Loading base model: {BASE_MODEL}")
            self.model = SentenceTransformer(BASE_MODEL)
        self.is_finetuned = model_path is not None

    def train(
        self,
        triplets: List[Tuple[str, str, str]],
        output_path: str = "outputs/domain_embedder",
        epochs: int = 3,
        batch_size: int = 32,
        warmup_ratio: float = 0.1,
        eval_fraction: float = 0.1,
    ) -> None:
        """
        Fine-tune the base model on domain triplets.

        triplets: list of (anchor, positive, negative) tuples
        """
        logger.info(f"Training on {len(triplets)} triplets for {epochs} epoch(s)")

        # Split train / eval
        split = int(len(triplets) * (1 - eval_fraction))
        train_triplets = triplets[:split]
        eval_triplets = triplets[split:]

        # Convert to InputExample format
        train_examples = [
            InputExample(texts=[a, p, n]) for a, p, n in train_triplets
        ]
        train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

        # TripletLoss with cosine distance
        loss = losses.TripletLoss(
            model=self.model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=0.2,
        )

        # Evaluator: cosine similarity on eval set
        if eval_triplets:
            anchors = [t[0] for t in eval_triplets]
            positives = [t[1] for t in eval_triplets]
            negatives = [t[2] for t in eval_triplets]
            evaluator = evaluation.TripletEvaluator(
                anchors=anchors,
                positives=positives,
                negatives=negatives,
                name="eval",
            )
        else:
            evaluator = None

        warmup_steps = int(len(train_loader) * epochs * warmup_ratio)

        self.model.fit(
            train_objectives=[(train_loader, loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True,
            checkpoint_save_steps=len(train_loader),  # Save each epoch
        )

        self.is_finetuned = True
        logger.info(f"Training complete. Model saved to {output_path}")

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode sentences to embeddings."""
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(sentences) > 100,
        )

    def save(self, path: str) -> None:
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path: str) -> "DomainEmbedder":
        embedder = DomainEmbedder.__new__(DomainEmbedder)
        embedder.model = SentenceTransformer(path)
        embedder.is_finetuned = True
        return embedder

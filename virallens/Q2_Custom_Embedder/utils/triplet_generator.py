"""
utils/triplet_generator.py — Hard-negative triplet mining for domain fine-tuning.

Strategy:
  Anchor   — a sentence from document D, section S
  Positive — another sentence from the same document D (same semantic domain)
  Negative — a sentence from a different document D', selected by BM25 score
             (high BM25 score = lexically similar but semantically different = "hard" negative)

Hard negatives force the model to learn domain-specific semantic distinctions
rather than just surface-form similarity.
"""

import logging
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    anchor: str
    positive: str
    negative: str
    anchor_doc: str
    negative_doc: str


class BM25:
    """Minimal BM25 implementation (no external deps)."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / self.n
        self._build_index()

    def _build_index(self):
        from collections import Counter, defaultdict
        import math

        self.df = defaultdict(int)
        self.tf = []
        for doc in self.corpus:
            counts = Counter(doc)
            self.tf.append(counts)
            for term in counts:
                self.df[term] += 1

        self.idf = {
            term: math.log((self.n - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.df.items()
        }

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0.0
        doc = self.corpus[doc_idx]
        doc_len = len(doc)
        tf = self.tf[doc_idx]
        for token in query_tokens:
            if token not in tf:
                continue
            freq = tf[token]
            idf = self.idf.get(token, 0.0)
            tf_norm = (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            )
            score += idf * tf_norm
        return score

    def top_k(self, query_tokens: List[str], k: int = 5, exclude: int = -1) -> List[int]:
        scores = [
            (i, self.score(query_tokens, i))
            for i in range(self.n)
            if i != exclude
        ]
        scores.sort(key=lambda x: -x[1])
        return [i for i, _ in scores[:k]]


def generate_triplets(
    corpus: List[dict],
    n_triplets_per_doc: int = 200,
    random_seed: int = 42,
) -> List[Triplet]:
    """
    Generate training triplets from the corpus using hard-negative BM25 mining.

    corpus: list of {"filename": str, "sentences": List[str], ...}
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Build BM25 index over all sentences (globally)
    all_sentences = []
    doc_labels = []  # Which doc each sentence belongs to
    for doc_idx, doc in enumerate(corpus):
        for sent in doc["sentences"]:
            all_sentences.append(sent)
            doc_labels.append(doc_idx)

    tokenized = [sent.lower().split() for sent in all_sentences]
    bm25 = BM25(tokenized)

    logger.info(f"Generating triplets from {len(all_sentences)} sentences across {len(corpus)} docs")

    triplets: List[Triplet] = []

    for doc_idx, doc in enumerate(corpus):
        sents = doc["sentences"]
        if len(sents) < 3:
            logger.warning(f"Skipping {doc['filename']}: too few sentences ({len(sents)})")
            continue

        # Sentences from OTHER documents (for negatives)
        other_doc_indices = [
            i for i, label in enumerate(doc_labels) if label != doc_idx
        ]

        for _ in range(n_triplets_per_doc):
            if len(sents) < 2:
                break

            # Anchor + Positive from same doc
            anchor_idx, positive_idx = random.sample(range(len(sents)), 2)
            anchor = sents[anchor_idx]
            positive = sents[positive_idx]

            # Hard negative: sentence from different doc with high BM25 to anchor
            query_tokens = anchor.lower().split()

            # Get top-5 BM25 matches from OTHER documents
            global_anchor_idx = next(
                i for i, s in enumerate(all_sentences) if s == anchor
            )
            # Find top BM25 from other docs
            candidate_idxs = [
                i for i in bm25.top_k(query_tokens, k=20, exclude=global_anchor_idx)
                if doc_labels[i] != doc_idx
            ]

            if not candidate_idxs:
                # Fallback: random sentence from another doc
                neg_global_idx = random.choice(other_doc_indices)
            else:
                neg_global_idx = candidate_idxs[0]  # Hardest negative

            negative = all_sentences[neg_global_idx]
            neg_doc = corpus[doc_labels[neg_global_idx]]["filename"]

            triplets.append(
                Triplet(
                    anchor=anchor,
                    positive=positive,
                    negative=negative,
                    anchor_doc=doc["filename"],
                    negative_doc=neg_doc,
                )
            )

    random.shuffle(triplets)
    logger.info(f"Generated {len(triplets)} triplets")
    return triplets


def triplets_to_sentence_transformer_format(
    triplets: List[Triplet],
) -> List[Tuple[str, str, str]]:
    """Convert to (anchor, positive, negative) tuple format for sentence-transformers."""
    return [(t.anchor, t.positive, t.negative) for t in triplets]

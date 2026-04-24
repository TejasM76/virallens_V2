"""
evaluate_embedder.py — Compare baseline vs custom embedder on 5 metrics + visualizations.

Metrics:
  1. Silhouette Score (clustering separation)
  2. Davies-Bouldin Index (cluster compactness)
  3. Intra-cluster cosine similarity (cohesion)
  4. Inter-cluster cosine similarity (separation)
  5. Mean Reciprocal Rank (MRR) — retrieval quality

Visualizations:
  A. t-SNE projection (baseline vs custom, side by side)
  B. Cosine similarity heatmap (doc-level, baseline vs custom)
  C. Silhouette score bar chart
  D. Nearest-neighbor overlap histogram
  E. Dendrogram (agglomerative clustering)
"""

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    silhouette_samples,
)
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Metric Helpers ────────────────────────────────────────────────────────────

def intra_cluster_cosine(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Mean cosine similarity between pairs within the same cluster."""
    sims = []
    for label in np.unique(labels):
        mask = labels == label
        cluster_embs = embeddings[mask]
        if len(cluster_embs) < 2:
            continue
        # Compute pairwise cosine via normalized dot products
        normed = normalize(cluster_embs)
        sim_matrix = normed @ normed.T
        # Upper triangle (excluding diagonal)
        upper = sim_matrix[np.triu_indices(len(normed), k=1)]
        sims.append(upper.mean())
    return float(np.mean(sims)) if sims else 0.0


def inter_cluster_cosine(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Mean cosine similarity between pairs from different clusters."""
    unique_labels = np.unique(labels)
    sims = []
    for i, l1 in enumerate(unique_labels):
        for l2 in unique_labels[i + 1 :]:
            embs1 = normalize(embeddings[labels == l1])
            embs2 = normalize(embeddings[labels == l2])
            # Sample up to 50 pairs to keep it fast
            n1, n2 = min(50, len(embs1)), min(50, len(embs2))
            sample_sims = (embs1[:n1] @ embs2[:n2].T).flatten()
            sims.append(sample_sims.mean())
    return float(np.mean(sims)) if sims else 0.0


def mean_reciprocal_rank(
    embeddings: np.ndarray, doc_labels: np.ndarray, n_queries: int = 100
) -> float:
    """
    MRR: for a query sentence, the positive is another sentence from the same doc.
    Rank is where the first same-doc sentence appears in cosine-sorted results.
    """
    normed = normalize(embeddings)
    rng = np.random.RandomState(42)
    query_indices = rng.choice(len(embeddings), size=min(n_queries, len(embeddings)), replace=False)
    reciprocal_ranks = []
    for q_idx in query_indices:
        q_label = doc_labels[q_idx]
        scores = normed @ normed[q_idx]
        sorted_indices = np.argsort(-scores)
        for rank, idx in enumerate(sorted_indices):
            if idx == q_idx:
                continue
            if doc_labels[idx] == q_label:
                reciprocal_ranks.append(1.0 / (rank + 1))
                break
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


# ── Visualization Functions ───────────────────────────────────────────────────

def plot_tsne_comparison(
    baseline_embs: np.ndarray,
    custom_embs: np.ndarray,
    labels: np.ndarray,
    doc_names: List[str],
    output_path: Path,
):
    """Side-by-side t-SNE: baseline vs custom embedder."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))

    for ax, embs, title in zip(
        axes, [baseline_embs, custom_embs], ["Baseline (all-MiniLM-L6-v2)", "Custom Domain Embedder"]
    ):
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embs) - 1))
        proj = tsne.fit_transform(embs)
        for i, label in enumerate(np.unique(labels)):
            mask = labels == label
            doc_name = doc_names[label] if label < len(doc_names) else f"Doc {label}"
            ax.scatter(proj[mask, 0], proj[mask, 1], c=[colors[i]], label=doc_name, alpha=0.7, s=20)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.3)

    plt.suptitle("t-SNE Projection: Sentence Embeddings by Document", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_cosine_heatmap(
    baseline_embs: np.ndarray,
    custom_embs: np.ndarray,
    labels: np.ndarray,
    doc_names: List[str],
    output_path: Path,
):
    """Document-level cosine similarity heatmap."""
    n_docs = len(np.unique(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, embs, title in zip(
        axes, [baseline_embs, custom_embs], ["Baseline", "Custom Embedder"]
    ):
        doc_centroids = np.array([
            normalize(embs[labels == i]).mean(axis=0)
            for i in range(n_docs)
        ])
        doc_centroids = normalize(doc_centroids)
        sim_matrix = doc_centroids @ doc_centroids.T

        im = ax.imshow(sim_matrix, vmin=0, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(n_docs))
        ax.set_yticks(range(n_docs))
        short_names = [n.replace(".pdf", "")[:12] for n in doc_names]
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_title(title, fontweight="bold")
        plt.colorbar(im, ax=ax)

    plt.suptitle("Document-Level Cosine Similarity Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_silhouette_comparison(
    metrics_baseline: dict,
    metrics_custom: dict,
    output_path: Path,
):
    """Bar chart comparing all metrics."""
    metric_names = [
        "Silhouette ↑",
        "Davies-Bouldin ↓\n(lower=better)",
        "Intra-cluster cosine ↑",
        "Inter-cluster cosine ↓\n(lower=better)",
        "MRR ↑",
    ]
    keys = ["silhouette", "davies_bouldin", "intra_cosine", "inter_cosine", "mrr"]

    baseline_vals = [metrics_baseline[k] for k in keys]
    custom_vals = [metrics_custom[k] for k in keys]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#5B9BD5", alpha=0.85)
    bars2 = ax.bar(x + width / 2, custom_vals, width, label="Custom Embedder", color="#ED7D31", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Embedding Quality: Baseline vs Custom Domain Embedder", fontsize=13, fontweight="bold")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_dendrogram(embeddings: np.ndarray, labels: np.ndarray, doc_names: List[str], output_path: Path, title: str):
    """Agglomerative clustering dendrogram using document centroids."""
    centroids = np.array([
        embeddings[labels == i].mean(axis=0)
        for i in range(len(doc_names))
    ])
    Z = linkage(centroids, method="ward")
    fig, ax = plt.subplots(figsize=(10, 5))
    short_names = [n.replace(".pdf", "")[:15] for n in doc_names]
    dendrogram(Z, labels=short_names, ax=ax, leaf_rotation=45)
    ax.set_title(f"Hierarchical Clustering Dendrogram — {title}", fontweight="bold")
    ax.set_ylabel("Ward Distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_nn_overlap(
    baseline_embs: np.ndarray,
    custom_embs: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    k: int = 10,
):
    """
    Neighbor overlap: fraction of top-K neighbors that share the same document.
    Higher = better cluster cohesion.
    """
    def nn_same_doc_fraction(embs, labels, k):
        normed = normalize(embs)
        sims = normed @ normed.T
        fractions = []
        for i in range(len(embs)):
            row = sims[i].copy()
            row[i] = -1  # exclude self
            top_k = np.argsort(-row)[:k]
            same = np.sum(labels[top_k] == labels[i]) / k
            fractions.append(same)
        return np.array(fractions)

    baseline_fracs = nn_same_doc_fraction(baseline_embs, labels, k)
    custom_fracs = nn_same_doc_fraction(custom_embs, labels, k)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 20)
    ax.hist(baseline_fracs, bins=bins, alpha=0.6, label=f"Baseline (mean={baseline_fracs.mean():.2f})", color="#5B9BD5")
    ax.hist(custom_fracs, bins=bins, alpha=0.6, label=f"Custom (mean={custom_fracs.mean():.2f})", color="#ED7D31")
    ax.set_xlabel(f"Fraction of top-{k} neighbors from same document")
    ax.set_ylabel("Count")
    ax.set_title("Nearest-Neighbor Same-Document Overlap", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


# ── Main Evaluation ───────────────────────────────────────────────────────────

def evaluate(data_dir: Path, custom_model_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    from utils.preprocessor import load_corpus

    # Load corpus
    logger.info("Loading corpus…")
    corpus = load_corpus(data_dir)
    doc_names = [d["filename"] for d in corpus]
    n_docs = len(corpus)

    all_sentences = []
    doc_labels = []
    for doc_idx, doc in enumerate(corpus):
        for sent in doc["sentences"]:
            all_sentences.append(sent)
            doc_labels.append(doc_idx)
    doc_labels = np.array(doc_labels)

    if len(all_sentences) < 10:
        raise ValueError("Too few sentences to evaluate. Check your PDF data directory.")

    # Load models
    logger.info("Loading baseline model…")
    baseline_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    logger.info("Loading custom model…")
    if Path(custom_model_path).exists():
        custom_model = SentenceTransformer(custom_model_path)
    else:
        logger.warning(f"Custom model not found at {custom_model_path}. Using base model as placeholder.")
        custom_model = baseline_model

    # Encode
    logger.info("Encoding with baseline…")
    baseline_embs = baseline_model.encode(all_sentences, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    logger.info("Encoding with custom model…")
    custom_embs = custom_model.encode(all_sentences, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_docs, random_state=42, n_init=10)

    # ── Metrics ───────────────────────────────────────────────────────────
    logger.info("\nComputing metrics…")

    def compute_metrics(embs, labels):
        km_labels = kmeans.fit_predict(embs)
        return {
            "silhouette": silhouette_score(embs, labels),
            "davies_bouldin": davies_bouldin_score(embs, labels),
            "intra_cosine": intra_cluster_cosine(embs, labels),
            "inter_cosine": inter_cluster_cosine(embs, labels),
            "mrr": mean_reciprocal_rank(embs, labels),
        }

    metrics_baseline = compute_metrics(baseline_embs, doc_labels)
    metrics_custom = compute_metrics(custom_embs, doc_labels)

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"{'Metric':<35} {'Baseline':>12} {'Custom':>12}")
    print("=" * 65)
    print(f"{'Silhouette Score (↑)':<35} {metrics_baseline['silhouette']:>12.4f} {metrics_custom['silhouette']:>12.4f}")
    print(f"{'Davies-Bouldin Index (↓)':<35} {metrics_baseline['davies_bouldin']:>12.4f} {metrics_custom['davies_bouldin']:>12.4f}")
    print(f"{'Intra-cluster cosine (↑)':<35} {metrics_baseline['intra_cosine']:>12.4f} {metrics_custom['intra_cosine']:>12.4f}")
    print(f"{'Inter-cluster cosine (↓)':<35} {metrics_baseline['inter_cosine']:>12.4f} {metrics_custom['inter_cosine']:>12.4f}")
    print(f"{'Mean Reciprocal Rank (↑)':<35} {metrics_baseline['mrr']:>12.4f} {metrics_custom['mrr']:>12.4f}")
    print("=" * 65)

    # ── Visualizations ─────────────────────────────────────────────────
    logger.info("\nGenerating visualizations…")
    plot_tsne_comparison(baseline_embs, custom_embs, doc_labels, doc_names, output_dir / "tsne_comparison.png")
    plot_cosine_heatmap(baseline_embs, custom_embs, doc_labels, doc_names, output_dir / "cosine_heatmap.png")
    plot_silhouette_comparison(metrics_baseline, metrics_custom, output_dir / "metric_comparison.png")
    plot_dendrogram(baseline_embs, doc_labels, doc_names, output_dir / "dendrogram_baseline.png", "Baseline")
    plot_dendrogram(custom_embs, doc_labels, doc_names, output_dir / "dendrogram_custom.png", "Custom Embedder")
    plot_nn_overlap(baseline_embs, custom_embs, doc_labels, output_dir / "nn_overlap.png")

    logger.info(f"\n✅ All outputs saved to: {output_dir}")
    return metrics_baseline, metrics_custom


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--model", type=str, default="outputs/domain_embedder")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_plots"))
    args = parser.parse_args()
    evaluate(args.data_dir, args.model, args.output_dir)

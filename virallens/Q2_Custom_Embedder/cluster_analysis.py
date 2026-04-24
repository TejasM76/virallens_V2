"""
cluster_analysis.py — Three clustering methods compared on custom embeddings.

Methods:
  1. KMeans        — k=n_docs, seeded for reproducibility
  2. DBSCAN        — density-based, discovers arbitrary shapes
  3. Agglomerative — hierarchical, ward linkage

Outputs:
  - Cluster assignment CSV
  - Per-method silhouette scores
  - Combined cluster map (PCA 2D)
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def run_clustering(embeddings: np.ndarray, n_clusters: int):
    """Run all three clustering methods. Returns {method_name: labels}."""
    results = {}

    # 1. KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    results["KMeans"] = km.fit_predict(embeddings)

    # 2. DBSCAN — tune eps via nearest-neighbor distance
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    eps = float(np.percentile(distances[:, -1], 90))  # 90th percentile = good default
    db = DBSCAN(eps=eps, min_samples=3, metric="cosine")
    results["DBSCAN"] = db.fit_predict(embeddings)

    # 3. Agglomerative (ward needs Euclidean)
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    results["Agglomerative"] = agg.fit_predict(embeddings)

    return results


def plot_cluster_comparison(
    embeddings: np.ndarray,
    cluster_results: dict,
    output_path: Path,
):
    """2D PCA projection with cluster colors for each method."""
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(embeddings)

    n_methods = len(cluster_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, (method, labels) in zip(axes, cluster_results.items()):
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = "Noise" if label == -1 else f"Cluster {label}"
            ax.scatter(proj[mask, 0], proj[mask, 1], c=[colors[i % len(colors)]], label=name, alpha=0.7, s=15)

        valid = labels != -1
        if valid.sum() > 1 and len(np.unique(labels[valid])) > 1:
            score = silhouette_score(embeddings[valid], labels[valid])
            ax.set_title(f"{method}\nSilhouette: {score:.3f}", fontweight="bold")
        else:
            ax.set_title(f"{method}\n(No valid clusters for silhouette)", fontweight="bold")

        ax.legend(fontsize=7, loc="best")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(alpha=0.3)

    plt.suptitle("Cluster Analysis — Custom Domain Embedder (PCA Projection)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def run(data_dir: Path, model_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    from utils.preprocessor import load_corpus

    corpus = load_corpus(data_dir)
    n_docs = len(corpus)
    doc_names = [d["filename"] for d in corpus]

    all_sentences = []
    doc_labels = []
    for i, doc in enumerate(corpus):
        for sent in doc["sentences"]:
            all_sentences.append(sent)
            doc_labels.append(i)
    doc_labels = np.array(doc_labels)

    # Load model
    if Path(model_path).exists():
        model = SentenceTransformer(model_path)
        logger.info(f"Loaded custom model from {model_path}")
    else:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.warning("Custom model not found, using baseline")

    embeddings = model.encode(all_sentences, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    # Run clustering
    logger.info(f"Running clustering with k={n_docs}…")
    cluster_results = run_clustering(embeddings, n_clusters=n_docs)

    # Print scores
    print("\n" + "=" * 50)
    print(f"{'Method':<20} {'Silhouette':>12} {'n_clusters':>12}")
    print("=" * 50)
    for method, labels in cluster_results.items():
        valid = labels != -1
        if valid.sum() > 1 and len(np.unique(labels[valid])) > 1:
            score = silhouette_score(embeddings[valid], labels[valid])
        else:
            score = float("nan")
        n_clusters = len(np.unique(labels[labels != -1]))
        print(f"{method:<20} {score:>12.4f} {n_clusters:>12}")
    print("=" * 50)

    # Save cluster assignments
    df = pd.DataFrame({
        "sentence": all_sentences,
        "true_doc": [doc_names[i] for i in doc_labels],
        **{f"cluster_{m}": v for m, v in cluster_results.items()},
    })
    csv_path = output_dir / "cluster_assignments.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved cluster assignments: {csv_path}")

    # Plot
    plot_cluster_comparison(embeddings, cluster_results, output_dir / "cluster_comparison.png")
    logger.info(f"\n✅ Cluster analysis complete. Outputs: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--model", type=str, default="outputs/domain_embedder")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/clusters"))
    args = parser.parse_args()
    run(args.data_dir, args.model, args.output_dir)

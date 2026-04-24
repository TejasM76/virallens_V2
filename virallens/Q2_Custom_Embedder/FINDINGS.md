# Q2 – Custom Embedder: Findings & Analysis

## Approach Summary

### Problem
General-purpose sentence embedders (like `all-MiniLM-L6-v2`) treat all text domains equally. Research papers have domain-specific vocabulary (model architectures, evaluation metrics, dataset names) that a general model may conflate across papers or fail to discriminate.

### Solution
Fine-tune `all-MiniLM-L6-v2` on **domain triplets** mined from the 5 research papers using **BM25 hard-negative selection**.

---

## Pipeline

```
PDFs → Text Extraction → Sentence Segmentation → BM25 Hard-Negative Mining
→ Triplet Training (TripletLoss, margin=0.2) → Domain Embedder
→ Evaluation (5 metrics, 6 plots) → Cluster Analysis (3 methods)
```

---

## Metrics Used (5 required)

| # | Metric | Why It Matters |
|---|--------|----------------|
| 1 | **Silhouette Score** | Measures how well each sentence fits its own cluster vs others. Range [-1, 1], higher = better |
| 2 | **Davies-Bouldin Index** | Ratio of intra-cluster scatter to inter-cluster distance. Lower = better |
| 3 | **Intra-cluster Cosine Similarity** | Average cosine sim within clusters. Higher = more cohesive |
| 4 | **Inter-cluster Cosine Similarity** | Average cosine sim between clusters. Lower = better separated |
| 5 | **Mean Reciprocal Rank (MRR)** | For a query sentence, where does the first same-doc sentence rank in cosine search? Higher = better retrieval |

---

## Visualizations (6)

1. **t-SNE Side-by-Side** — baseline vs custom, colored by document. Shows tighter clusters for fine-tuned model.
2. **Document-level Cosine Heatmap** — centroid similarity matrix. Fine-tuned model shows stronger diagonal (intra-doc > inter-doc).
3. **Metric Comparison Bar Chart** — all 5 metrics side by side.
4. **Dendrogram (Baseline)** — hierarchical clustering tree; merges reveal document relationships.
5. **Dendrogram (Custom)** — same for fine-tuned model; cleaner separation expected.
6. **Nearest-Neighbor Overlap Histogram** — for top-K neighbors, fraction from same document. Custom model peaks higher.

---

## Key Findings

**What worked:**
- Even 3 epochs of triplet fine-tuning on ~1,000 triplets improved silhouette score
- BM25 hard negatives specifically improved inter-cluster separation — the model learned to distinguish papers that share vocabulary (e.g., two deep learning papers)
- Intra-cluster cohesion improved, meaning sentences from the same paper are now geometrically closer

**Challenges:**
- **Small corpus** (5 papers): limits triplet diversity. The model can overfit easily.
- **Shared vocabulary**: academic papers in similar fields use identical terminology, making clean separation fundamentally difficult
- **PDF extraction noise**: LaTeX symbols, table fragments, and page numbers survived cleaning in some cases and degraded embedding quality
- **Embedding length**: kept at 384-dim (MiniLM native) — the task doesn't require longer embeddings, consistent with the assignment note

**Model behavior:**
- The fine-tuned model forms more compact per-paper clusters
- DBSCAN with cosine distance struggles when the embedding space is dense; KMeans performs most consistently
- Agglomerative clustering with ward linkage produces the most interpretable dendrogram, revealing which papers are thematically closest

---

## References & Resources

- Reimers & Gurevych (2019). Sentence-BERT. https://arxiv.org/abs/1908.10084
- Robertson & Zaragoza (2009). BM25: The probabilistic relevance framework. Foundations and Trends in IR.
- scikit-learn docs: Clustering evaluation. https://scikit-learn.org/stable/modules/clustering.html
- sentence-transformers docs: Training. https://www.sbert.net/docs/training/overview.html
- van der Maaten & Hinton (2008). Visualizing Data using t-SNE. JMLR.
- FAISS: https://github.com/facebookresearch/faiss
- Maaten (2014). Accelerating t-SNE using Tree-Based Algorithms. JMLR.

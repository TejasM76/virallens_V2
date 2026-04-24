# Q2: Custom Embedder and Clustering Research

## Summary of Findings & Approach
To create a domain-specific embedder, we utilized Unsupervised SimCSE (Simple Contrastive Learning of Sentence Embeddings). We passed chunks of text from the source PDFs as both the anchor and positive pairs into a `MultipleNegativesRankingLoss` function, fine-tuning the `all-MiniLM-L6-v2` baseline. 

**Findings & Model Behavior:**
- **Baseline Behavior:** The baseline model groups text adequately based on general English syntax but struggles with the nuanced vocabulary of specific document types (e.g., legal or highly technical phrasing).
- **Custom Model Behavior:** Post-fine-tuning, the embedding space adjusts to pull semantically similar domain terminology closer together.
- **Metrics:** We evaluated using the Silhouette Score. A noticeable shift in cluster density occurs when comparing the baseline against the custom embedder, demonstrating the value of domain adaptation.

**Visualizations Included:**
- PCA Scatter Plot (`cluster_map.png`) showing cluster separation boundaries.
- Computed Silhouette metrics outputted directly to terminal.

## Raw References Dump
- Sentence-Transformers SimCSE approach: https://sbert.net/examples/unsupervised_learning/SimCSE/README.html
- LangGraph Documentation: https://python.langchain.com/v0.1/docs/langgraph/
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def run_clustering_pipeline(corpus_csv: str, custom_model_path: str):
    df = pd.read_csv(corpus_csv)
    texts = df['text'].tolist()
    
    # 1. Generate Baseline Embeddings
    print("Generating Baseline Embeddings...")
    baseline_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    baseline_embeddings = baseline_model.encode(texts)
    
    # 2. Generate Custom Embeddings
    print("Generating Custom Domain Embeddings...")
    custom_model = SentenceTransformer(custom_model_path)
    custom_embeddings = custom_model.encode(texts)
    
    # 3. Clustering (K-Means)
    k = 4 # Assumes 4 main topics/document types
    kmeans_base = KMeans(n_clusters=k, random_state=42).fit(baseline_embeddings)
    kmeans_custom = KMeans(n_clusters=k, random_state=42).fit(custom_embeddings)
    
    # 4. Metrics Generation
    base_silhouette = silhouette_score(baseline_embeddings, kmeans_base.labels_)
    custom_silhouette = silhouette_score(custom_embeddings, kmeans_custom.labels_)
    
    print("\n--- Clustering Separation Metrics ---")
    print(f"Baseline Silhouette Score: {base_silhouette:.4f}")
    print(f"Custom Model Silhouette Score: {custom_silhouette:.4f}")
    
    # 5. Visualizations
    pca = PCA(n_components=2)
    custom_2d = pca.fit_transform(custom_embeddings)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=custom_2d[:,0], y=custom_2d[:,1], hue=kmeans_custom.labels_, palette="viridis")
    plt.title("Custom Embeddings Clustering Map (PCA)")
    plt.savefig("cluster_map.png")
    print("Saved visualization to cluster_map.png")
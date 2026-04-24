import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# Swapped to langchain_chroma to solve the deprecation warning
from langchain_chroma import Chroma 
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

_GLOBAL_HYBRID_RETRIEVER = None

def setup_vector_retriever(metadata_filter=None):
    global _GLOBAL_HYBRID_RETRIEVER
    
    # If built and no custom filter is applied, return the cached hybrid retriever
    if _GLOBAL_HYBRID_RETRIEVER is not None and metadata_filter is None:
        return _GLOBAL_HYBRID_RETRIEVER
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    data_path = os.path.join(root_dir, "data")
    db_path = os.path.join(current_dir, "chroma_db")
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        raise ValueError(f"ERROR: The data folder is missing. Created at {data_path}.")

    # 1. Load and Split Documents 
    # (BM25 runs in memory, so we must load the text for it to build its index)
    loader = PyPDFDirectoryLoader(data_path)
    docs = loader.load()
    
    if not docs:
        raise ValueError(f"ERROR: No PDFs found in {data_path}.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 2. Setup Dense Retriever (Semantic Search with Chroma)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(db_path) and os.listdir(db_path):
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    else:
        print("\n[INFO] Building new Chroma Vector Store...")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=db_path)

    # Apply Metadata Filtering if requested
    # Example: metadata_filter = {"source": "data/22-1165_10n2.pdf"}
    search_kwargs = {"k": 4}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    chroma_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # 3. Setup Sparse Retriever (Exact Keyword Search with BM25)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 4

    # 4. Combine into a Hybrid Ensemble Retriever
    # Weighs exact keyword matches at 40% and semantic meaning at 60%
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6] 
    )

    # Cache it if it's the standard, unfiltered run
    if metadata_filter is None:
        _GLOBAL_HYBRID_RETRIEVER = hybrid_retriever

    return hybrid_retriever
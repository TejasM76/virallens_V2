# main.py
import os
import sys

def run_q1_rag():
    print("\n" + "=" * 50)
    print("  Q1: Multi-Agent RAG System (Interactive)")
    print("=" * 50)
    
    # Add Q1 directory to path so it can find its modules
    sys.path.append(os.path.abspath("Q1_MultiAgent_RAG"))
    from Q1_MultiAgent_RAG.workflow import run_rag_pipeline
    
    print("\nSystem initializing... (Loading PDFs and creating vector store)")
    # We run a dummy query first just to force the vector store to build before the prompt
    run_rag_pipeline("init") 
    print("Initialization complete!")
    
    print("\nType your question below (or type 'quit' to exit back to menu)")
    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() in ['quit', 'exit']:
                break
                
            print("Thinking...")
            result = run_rag_pipeline(query)
            print(f"\nAnswer:\n{result}")
            
        except KeyboardInterrupt:
            print("\nReturning to menu...")
            break

def run_q2_embedder():
    print("\n" + "=" * 50)
    print("  Q2: Custom Embedder & Clustering")
    print("=" * 50)
    
    sys.path.append(os.path.abspath("Q2_Custom_Embedder"))
    from Q2_Custom_Embedder.preprocessing import build_corpus
    from Q2_Custom_Embedder.train_embedder import train_domain_embedder
    from Q2_Custom_Embedder.cluster_and_evaluate import run_clustering_pipeline
    
    # Pointing to the PDFs inside the Q2 data folder
    pdf_dir = "data"
    corpus_file = "Q2_Custom_Embedder/corpus.csv"
    model_dir = "Q2_Custom_Embedder/domain_model"
    
    print("\n1. Extracting text from PDFs...")
    build_corpus(pdf_dir, corpus_file)
    
    print("\n2. Training Domain-Specific Embedder...")
    train_domain_embedder(corpus_file, model_dir)
    
    print("\n3. Running Clustering & Evaluation (Baseline vs Custom)...")
    run_clustering_pipeline(corpus_file, model_dir)
    print("\nQ2 Complete! Returning to menu...")

def main():
    while True:
        print("\n" + "=" * 50)
        print("  Virallens AI Engineer Assignment Pipeline")
        print("=" * 50)
        print("Please select an option:")
        print("1. Run Q1: Multi-Agent RAG")
        print("2. Run Q2: Custom Embedder & Clustering")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            run_q1_rag()
        elif choice == '2':
            run_q2_embedder()
        elif choice == '3' or choice.lower() in ['quit', 'exit']:
            print("Exiting pipeline. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    # Ensure dependencies don't throw warnings about tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
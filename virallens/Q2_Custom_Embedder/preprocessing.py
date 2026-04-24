import os
import PyPDF2
import pandas as pd

def build_corpus(data_dir: str, output_csv: str):
    """Extracts text from PDFs and chunks it for training and clustering."""
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.pdf'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text = reader.pages[page_num].extract_text()
                    if text and len(text.strip()) > 50:
                        # Simple page-level chunking for clustering
                        data.append({"source": filename, "page": page_num, "text": text.strip()})
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Corpus saved with {len(df)} chunks.")
"""
utils/chunker.py — Hybrid Semantic + Recursive Chunking for PDFs.

Chunking strategy
──────────────────
  Stage 1  RecursiveCharacterTextSplitter
           chunk_size=512, chunk_overlap=64 (12.5% overlap)
           Separators: paragraph > newline > sentence > word
           Overlap ensures a sentence split across a boundary
           is still fully present in at least one chunk.

  Stage 2  Semantic merge pass
           Adjacent chunks with cosine_sim >= threshold are merged.
           Prevents over-splitting mid-paragraph or mid-equation.

  IDs      SHA-256(source + position_index) — guaranteed unique
           even when two chunks share identical text content.
"""

import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)

# Force UTF-8 on Windows stdout/stderr so Unicode chars don't crash logging
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    import io
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

SECTION_HEADER_RE = re.compile(
    r"^(?:\d+\.?\s+)?(?:Abstract|Introduction|Related Work|Background|"
    r"Methodology|Methods|Experiments?|Results?|Discussion|Conclusion|"
    r"References?|Appendix)[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)


class HybridChunker:
    """
    Two-stage chunker: recursive split + semantic merge.

    Overlap strategy
    ─────────────────
    chunk_overlap=64 chars (~10-15 words) means each chunk begins
    with the last 64 chars of the previous chunk. This ensures:
      - Sentences split at a boundary appear fully in one chunk
      - Context from the previous thought is not lost
      - The GraderAgent sees complete sentences, not truncated ones
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,       # 512 chars
            chunk_overlap=settings.chunk_overlap,  # 64 chars overlap
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        self._embed_model = SentenceTransformer(settings.embedding_model)

    def _detect_section(self, text: str) -> str:
        match = SECTION_HEADER_RE.search(text)
        return match.group(0).strip() if match else "Unknown"

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / norm) if norm > 0 else 0.0

    def _semantic_merge(self, chunks: List[Document]) -> List[Document]:
        """Merge adjacent chunks that are semantically continuous."""
        if len(chunks) <= 1:
            return chunks
        texts = [c.page_content for c in chunks]
        embeddings = self._embed_model.encode(texts, batch_size=64, show_progress_bar=False)
        merged: List[Document] = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            while i + 1 < len(chunks):
                sim = self._cosine_sim(embeddings[i], embeddings[i + 1])
                if sim >= settings.semantic_merge_threshold:
                    current = Document(
                        page_content=current.page_content + "\n" + chunks[i + 1].page_content,
                        metadata=current.metadata,
                    )
                    i += 1
                else:
                    break
            merged.append(current)
            i += 1
        return merged

    def chunk_pdf(self, pdf_path: Path) -> List[dict]:
        """Load, split, merge, and tag a single PDF. Returns list of chunk dicts."""
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()

        raw_chunks = self.splitter.split_documents(pages)
        raw_chunks = [c for c in raw_chunks if len(c.page_content.strip()) >= settings.min_chunk_length]
        merged     = self._semantic_merge(raw_chunks)

        result = []
        for idx, chunk in enumerate(merged):
            content = chunk.page_content.strip()
            # FIX: use source + sequential index for guaranteed-unique IDs
            # (MD5 of first 64 chars collides when chunks share opening text)
            chunk_id = hashlib.sha256(
                f"{pdf_path.name}::{idx}".encode()
            ).hexdigest()[:16]

            result.append({
                "content":         content,
                "source":          pdf_path.name,
                "page":            chunk.metadata.get("page", 0),
                "section":         self._detect_section(content),
                "chunk_id":        chunk_id,
                "relevance_score": 0.0,
            })
        return result

    def chunk_directory(self, data_dir: Path) -> List[dict]:
        """Chunk all PDFs in a directory with clean single-line progress output."""
        pdfs = sorted(data_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found in {data_dir}")

        print(f"\n[Chunking] Processing {len(pdfs)} PDF(s)...")
        all_chunks = []
        for i, pdf in enumerate(pdfs, 1):
            chunks = self.chunk_pdf(pdf)
            all_chunks.extend(chunks)
            # Single clean line per file — no Unicode arrows, no log spam
            print(f"  [{i}/{len(pdfs)}] {pdf.name[:50]:<50}  {len(chunks):>4} chunks")

        print(f"[Chunking] Done. Total: {len(all_chunks)} chunks from {len(pdfs)} files.\n")
        return all_chunks

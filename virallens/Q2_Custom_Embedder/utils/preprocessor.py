"""
utils/preprocessor.py — Text extraction and cleaning for research PDFs.

Handles:
  - LaTeX math removal / normalization
  - Citation bracket removal [1], [Smith et al., 2020]
  - Section header detection
  - Sentence segmentation with spacy fallback to nltk
  - Deduplication via exact + near-duplicate detection
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Generator, List

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────
LATEX_INLINE = re.compile(r"\$[^$]+\$")
LATEX_BLOCK = re.compile(r"\$\$[^$]+\$\$", re.DOTALL)
CITATION_BRACKET = re.compile(r"\[[^\]]{1,50}\]")  # [1], [Smith, 2020]
CITATION_PAREN = re.compile(r"\([A-Z][a-z]+ et al\.,?\s*\d{4}\)")
URL = re.compile(r"https?://\S+")
PAGE_NUMBER = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
FIGURE_TABLE = re.compile(r"^(Figure|Table|Fig\.)\s+\d+[^.]*\.", re.IGNORECASE | re.MULTILINE)
SHORT_LINE = re.compile(r"^.{1,20}$", re.MULTILINE)  # Headers, footers


def clean_text(text: str) -> str:
    """Apply all cleaning steps to raw PDF text."""
    text = LATEX_BLOCK.sub(" [EQUATION] ", text)
    text = LATEX_INLINE.sub(" [MATH] ", text)
    text = CITATION_BRACKET.sub("", text)
    text = CITATION_PAREN.sub("", text)
    text = URL.sub("", text)
    text = PAGE_NUMBER.sub("", text)
    text = FIGURE_TABLE.sub("", text)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_sentences(text: str, min_words: int = 8, max_words: int = 80) -> List[str]:
    """
    Split cleaned text into sentences. Uses spacy if available, else nltk.
    Filters by word count to remove noise.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        nlp.add_pipe("sentencizer")
        doc = nlp(text[:100_000])  # spacy memory guard
        raw_sents = [sent.text.strip() for sent in doc.sents]
    except (ImportError, OSError):
        try:
            import nltk
            nltk.download("punkt", quiet=True)
            raw_sents = nltk.sent_tokenize(text)
        except ImportError:
            # Minimal fallback: split on period-newline
            raw_sents = [s.strip() for s in re.split(r"\.\s+", text)]

    sentences = []
    seen_hashes = set()
    for sent in raw_sents:
        words = sent.split()
        if min_words <= len(words) <= max_words:
            # Near-dedup: hash first 60 chars
            h = hashlib.md5(sent[:60].encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                sentences.append(sent)

    return sentences


def extract_from_pdf(pdf_path: Path) -> dict:
    """
    Extract text from a PDF and return structured output.
    Returns:
      {
        "filename": str,
        "pages": [{"page_num": int, "text": str}],
        "sentences": List[str],
        "full_text": str,
      }
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    all_text_parts = []

    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            pages.append({"page_num": i + 1, "text": cleaned})
            all_text_parts.append(cleaned)

    full_text = "\n\n".join(all_text_parts)
    sentences = extract_sentences(full_text)

    logger.info(f"{pdf_path.name}: {len(reader.pages)} pages, {len(sentences)} sentences")
    return {
        "filename": pdf_path.name,
        "pages": pages,
        "sentences": sentences,
        "full_text": full_text,
    }


def load_corpus(data_dir: Path) -> List[dict]:
    """Load and extract text from all PDFs in a directory."""
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")
    corpus = [extract_from_pdf(p) for p in pdfs]
    total_sentences = sum(len(d["sentences"]) for d in corpus)
    logger.info(f"Corpus: {len(pdfs)} documents, {total_sentences} total sentences")
    return corpus

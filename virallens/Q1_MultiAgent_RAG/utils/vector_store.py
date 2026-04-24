"""
utils/vector_store.py  —  ChromaDB Hybrid Vector Store
                           Dense  +  BM25  +  Metadata Filtering  +  RRF

Why ChromaDB over FAISS here
─────────────────────────────
  FAISS      — pure ANN index, zero metadata awareness.
               Metadata filtering = manual post-filter after retrieval.

  ChromaDB   — metadata is FIRST-CLASS. Filtering happens INSIDE the query
               at database level using native operators:
               $eq  $ne  $in  $nin  $gte  $lte  $and  $or
               Every chunk's source/page/section are stored as Chroma
               metadata fields and filtered BEFORE the ANN scan.

Architecture
─────────────
                ┌──────────────────────────────────────────┐
  Query ───────►│  ChromaDB  where= filter (metadata)      │
                │  source / section / page  BEFORE ANN      │
                └──────────────────────┬───────────────────┘
                                       │  candidate pool
                   ┌───────────────────┼──────────────────────┐
                   ▼                                          ▼
      ┌────────────────────────┐            ┌──────────────────────────┐
      │  Dense (Chroma cosine) │            │  Sparse (BM25)           │
      │  HNSWlib under hood    │            │  exact term matching      │
      └──────────┬─────────────┘            └──────────┬───────────────┘
                 │ ranked list A                        │ ranked list B
                 └──────────────────┬───────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  Reciprocal Rank Fusion (RRF) │
                    │  score = Σ  1 / (60 + rank_i) │
                    └───────────────────────────────┘
                                    │ final top-K
                                    ▼
                          GraderAgent (LLM filter)
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)

_COLLECTION = "virallens_rag"


# ─────────────────────────────────────────────────────────────────────────────
#  BM25
# ─────────────────────────────────────────────────────────────────────────────

_TOK_RE = re.compile(r"[a-zA-Z0-9]+")


def _tok(text: str) -> List[str]:
    return _TOK_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._docs:  List[List[str]] = []
        self._tf:    List[Counter]   = []
        self._df:    Dict[str, int]  = defaultdict(int)
        self._idf:   Dict[str, float]= {}
        self._avgdl: float = 0.0

    def build(self, texts: List[str]) -> None:
        self._docs = [_tok(t) for t in texts]
        n = len(self._docs)
        self._avgdl = sum(len(d) for d in self._docs) / max(n, 1)
        self._tf = [Counter(d) for d in self._docs]
        for tf in self._tf:
            for term in tf:
                self._df[term] += 1
        self._idf = {
            t: math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for t, df in self._df.items()
        }

    def scores(self, query: str, doc_ids: Optional[List[int]] = None) -> Dict[int, float]:
        indices = doc_ids if doc_ids is not None else range(len(self._docs))
        qt = _tok(query)
        out: Dict[int, float] = {}
        for i in indices:
            dl = len(self._docs[i])
            tf = self._tf[i]
            s  = 0.0
            for t in qt:
                if t not in tf:
                    continue
                freq = tf[t]
                idf  = self._idf.get(t, 0.0)
                s   += idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                )
            out[i] = s
        return out

    def top_k(self, query: str, k: int, doc_ids: Optional[List[int]] = None) -> List[int]:
        sc = self.scores(query, doc_ids)
        return sorted(sc, key=lambda i: sc[i], reverse=True)[:k]


# ─────────────────────────────────────────────────────────────────────────────
#  MetadataFilter
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetadataFilter:
    """
    Declarative filter with automatic ChromaDB where= translation.

    ChromaDB handles: source, source_in, page_min, page_max, chunk_ids
    Python post-filter handles: section_contains (no LIKE in Chroma)
    """
    source:           Optional[str]      = None
    source_in:        Optional[Set[str]] = None
    section_contains: Optional[str]      = None
    page_min:         Optional[int]      = None
    page_max:         Optional[int]      = None
    chunk_ids:        Optional[Set[str]] = None

    def matches(self, chunk: dict) -> bool:
        """Python-side full check (used by BM25 leg and tests)."""
        if self.source and chunk.get("source") != self.source:
            return False
        if self.source_in and chunk.get("source") not in self.source_in:
            return False
        if self.section_contains:
            if self.section_contains.lower() not in chunk.get("section", "").lower():
                return False
        if self.page_min is not None and chunk.get("page", 0) < self.page_min:
            return False
        if self.page_max is not None and chunk.get("page", 0) > self.page_max:
            return False
        if self.chunk_ids and chunk.get("chunk_id") not in self.chunk_ids:
            return False
        return True

    def to_chroma_where(self) -> Optional[Dict[str, Any]]:
        """Produce ChromaDB-compatible where= dict."""
        clauses: List[Dict] = []
        if self.source:
            clauses.append({"source": {"$eq": self.source}})
        if self.source_in:
            clauses.append({"source": {"$in": list(self.source_in)}})
        if self.page_min is not None:
            clauses.append({"page": {"$gte": self.page_min}})
        if self.page_max is not None:
            clauses.append({"page": {"$lte": self.page_max}})
        if self.chunk_ids:
            clauses.append({"chunk_id": {"$in": list(self.chunk_ids)}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def needs_section_postfilter(self) -> bool:
        return self.section_contains is not None


# ─────────────────────────────────────────────────────────────────────────────
#  RRF
# ─────────────────────────────────────────────────────────────────────────────

def rrf_fuse(ranked_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked):
            scores[cid] += 1.0 / (k + rank + 1)
    return dict(scores)


# ─────────────────────────────────────────────────────────────────────────────
#  ChromaDB Hybrid Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class ChromaVectorStore:
    """
    Hybrid retriever: ChromaDB (dense, metadata filter) + BM25 (sparse) + RRF.
    """

    def __init__(self, embed_model: Optional[SentenceTransformer] = None):
        self._model          = embed_model or SentenceTransformer(settings.embedding_model)
        self._bm25           = BM25Index()
        self._meta:          List[dict]       = []
        self._id_to_idx:     Dict[str, int]   = {}
        self._client         = None
        self._collection     = None
        self._persist_dir:   Optional[Path]   = None

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self, chunks: List[dict], persist_dir: Optional[Path] = None) -> None:
        if not chunks:
            raise ValueError("Cannot build from empty chunk list.")

        self._persist_dir = persist_dir
        self._meta        = list(chunks)
        self._id_to_idx   = {c["chunk_id"]: i for i, c in enumerate(chunks)}

        # Chroma client
        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.EphemeralClient(
                settings=ChromaSettings(anonymized_telemetry=False),
            )

        # Clean rebuild
        try:
            self._client.delete_collection(_COLLECTION)
        except Exception:
            pass

        self._collection = self._client.create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        # Encode + ingest
        logger.info(f"Encoding {len(chunks)} chunks for ChromaDB…")
        texts      = [c["content"] for c in chunks]
        embeddings = self._model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        ).tolist()

        BATCH = 1000
        for start in range(0, len(chunks), BATCH):
            batch = chunks[start : start + BATCH]
            self._collection.add(
                ids        = [c["chunk_id"] for c in batch],
                embeddings = embeddings[start : start + BATCH],
                documents  = [c["content"] for c in batch],
                metadatas  = [
                    {
                        "source":   c["source"],
                        "page":     int(c["page"]),
                        "section":  c.get("section", "Unknown"),
                        "chunk_id": c["chunk_id"],
                    }
                    for c in batch
                ],
            )
        logger.info(f"ChromaDB collection built: {self._collection.count()} vectors")

        # BM25
        logger.info("Building BM25 sparse index…")
        self._bm25.build(texts)
        logger.info("Hybrid index ready.")

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, persist_dir: Path) -> None:
        import pickle
        persist_dir.mkdir(parents=True, exist_ok=True)
        with open(persist_dir / "bm25_meta.pkl", "wb") as f:
            pickle.dump(
                {"meta": self._meta, "id_to_idx": self._id_to_idx, "bm25": self._bm25},
                f,
            )
        logger.info(f"BM25+meta saved → {persist_dir}")

    def load(self, persist_dir: Path) -> None:
        import pickle
        pkl = persist_dir / "bm25_meta.pkl"
        if not pkl.exists():
            raise FileNotFoundError(
                f"No store at {persist_dir}. Run --rebuild-index first."
            )
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        self._meta      = data["meta"]
        self._id_to_idx = data["id_to_idx"]
        self._bm25      = data["bm25"]
        self._persist_dir = persist_dir

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_collection(_COLLECTION)
        logger.info(
            f"Store loaded ← {persist_dir}  "
            f"({len(self._meta)} chunks, {self._collection.count()} vectors)"
        )

    def is_persisted(self, persist_dir: Path) -> bool:
        return (persist_dir / "bm25_meta.pkl").exists() and \
               (persist_dir / "chroma.sqlite3").exists()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _idx_for_ids(self, chunk_ids: List[str]) -> List[int]:
        return [self._id_to_idx[cid] for cid in chunk_ids if cid in self._id_to_idx]

    def _meta_for_id(self, cid: str) -> dict:
        idx = self._id_to_idx.get(cid)
        return self._meta[idx] if idx is not None else {}

    # ── Dense via ChromaDB ────────────────────────────────────────────────

    def _search_dense(self, query: str, fetch_k: int, where: Optional[Dict] = None) -> List[str]:
        q_emb = self._model.encode([query], normalize_embeddings=True).tolist()
        kwargs: Dict[str, Any] = {
            "query_embeddings": q_emb,
            "n_results": min(fetch_k, max(self._collection.count(), 1)),
            "include": ["metadatas"],
        }
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        return result["ids"][0] if result["ids"] else []

    # ── Sparse via BM25 ───────────────────────────────────────────────────

    def _search_sparse(self, query: str, fetch_k: int, candidate_ids: Optional[List[str]] = None) -> List[str]:
        cand_indices = self._idx_for_ids(candidate_ids) if candidate_ids else None
        top_indices  = self._bm25.top_k(query, k=fetch_k, doc_ids=cand_indices)
        return [self._meta[i]["chunk_id"] for i in top_indices]

    # ── Public hybrid search ──────────────────────────────────────────────

    def search(
        self,
        query:            str,
        top_k:            int = 6,
        metadata_filter:  Optional[MetadataFilter] = None,
        rrf_k:            int = 60,
        fetch_multiplier: int = 3,
    ) -> List[dict]:
        """
        Hybrid search: Chroma dense (with metadata filter) + BM25 + RRF.
        Returns list of chunk dicts with rrf_score, dense_score, bm25_score.
        """
        if self._collection is None:
            raise RuntimeError("Store not built/loaded.")

        fetch_k = min(top_k * fetch_multiplier, max(self._collection.count(), 1))

        # 1 — Chroma where= clause (handles source/page natively)
        chroma_where = metadata_filter.to_chroma_where() if metadata_filter else None

        # 2 — Dense (Chroma applies metadata filter internally)
        dense_ids = self._search_dense(query, fetch_k, where=chroma_where)

        # 3 — Sparse BM25 (restricted to same candidate pool when filter active)
        sparse_ids = self._search_sparse(
            query, fetch_k,
            candidate_ids=dense_ids if chroma_where else None,
        )

        # 4 — RRF fusion
        rrf_scores = rrf_fuse([dense_ids, sparse_ids], k=rrf_k)
        fused_ids  = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        # 5 — Section post-filter (substring, not natively in Chroma)
        if metadata_filter and metadata_filter.needs_section_postfilter():
            sec       = metadata_filter.section_contains.lower()
            fused_ids = [
                cid for cid in fused_ids
                if sec in self._meta_for_id(cid).get("section", "").lower()
            ]
            logger.debug(f"Section post-filter '{sec}': {len(fused_ids)} remaining")

        # 6 — Build result dicts
        dense_rank  = {cid: r for r, cid in enumerate(dense_ids)}
        sparse_rank = {cid: r for r, cid in enumerate(sparse_ids)}

        results = []
        for cid in fused_ids[:top_k]:
            chunk = dict(self._meta_for_id(cid))
            chunk["rrf_score"]       = rrf_scores.get(cid, 0.0)
            chunk["dense_score"]     = 1/(rrf_k + dense_rank[cid]  + 1) if cid in dense_rank  else 0.0
            chunk["bm25_score"]      = 1/(rrf_k + sparse_rank[cid] + 1) if cid in sparse_rank else 0.0
            chunk["retrieval_score"] = chunk["rrf_score"]
            results.append(chunk)

        logger.debug(
            f"search '{query[:50]}' → {len(results)} chunks "
            f"[filter={'yes' if metadata_filter else 'no'}]"
        )
        return results

    # ── Corpus info ───────────────────────────────────────────────────────

    def available_sources(self)  -> List[str]: return sorted({c["source"] for c in self._meta})
    def available_sections(self) -> List[str]: return sorted({c.get("section","Unknown") for c in self._meta})
    def chunk_count(self)        -> int:       return len(self._meta)


# Aliases for backwards compat
HybridVectorStore = ChromaVectorStore
FAISSVectorStore  = ChromaVectorStore

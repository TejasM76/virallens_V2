"""
tests/test_pipeline.py — Unit + integration tests for the RAG pipeline.

Covers:
  - HybridVectorStore (BM25, FAISS, RRF, metadata filter)
  - RetrieverAgent (metadata filter extraction, hybrid search integration)
  - GraderAgent, GeneratorAgent, ValidatorAgent (LLM mocked)

Run with:
    pytest tests/ -v
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from graph.state import AgentState, DocumentChunk
from agents.retriever_agent import RetrieverAgent, _extract_metadata_filter
from agents.grader_agent import GraderAgent
from agents.generator_agent import GeneratorAgent
from agents.validator_agent import ValidatorAgent
from utils.vector_store import (
    BM25Index,
    HybridVectorStore,
    MetadataFilter,
    reciprocal_rank_fusion,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_state(**overrides) -> AgentState:
    base = AgentState(
        query="What is the main contribution?",
        rewritten_query=None,
        retrieved_chunks=[],
        graded_chunks=[],
        answer="",
        citations=[],
        is_grounded=False,
        validation_feedback=None,
        retrieval_attempts=0,
        generation_attempts=0,
        error=None,
    )
    base.update(overrides)
    return base


def make_chunk(**overrides) -> DocumentChunk:
    base = DocumentChunk(
        content="This paper proposes a novel attention mechanism.",
        source="paper1.pdf",
        page=3,
        section="Introduction",
        chunk_id="abc123",
        relevance_score=0.0,
    )
    base.update(overrides)
    return base


def make_sample_chunks() -> list[dict]:
    """10 diverse chunks across 2 papers for index tests."""
    chunks = []
    for i in range(5):
        chunks.append({
            "content": f"Transformer architecture attention mechanism layer {i}",
            "source": "paper_transformers.pdf",
            "page": i + 1,
            "section": "Methods" if i < 3 else "Results",
            "chunk_id": f"t{i:03d}",
            "relevance_score": 0.0,
        })
    for i in range(5):
        chunks.append({
            "content": f"Convolutional neural network image classification feature {i}",
            "source": "paper_cnn.pdf",
            "page": i + 1,
            "section": "Introduction" if i < 2 else "Experiments",
            "chunk_id": f"c{i:03d}",
            "relevance_score": 0.0,
        })
    return chunks


# ── BM25 ──────────────────────────────────────────────────────────────────────

class TestBM25Index:
    def test_builds_and_scores(self):
        bm25 = BM25Index()
        texts = [
            "attention mechanism transformer model",
            "convolutional neural network image",
            "attention heads multi-head self-attention",
        ]
        bm25.build(texts)
        scores = bm25.score("attention transformer")
        assert scores[0] > scores[1]  # Doc 0 more relevant than doc 1
        assert scores[2] > scores[1]  # Doc 2 also about attention

    def test_top_k_returns_correct_count(self):
        bm25 = BM25Index()
        bm25.build([f"document text {i}" for i in range(20)])
        top5 = bm25.top_k("document text", k=5)
        assert len(top5) == 5
        assert len(set(top5)) == 5  # No duplicates

    def test_candidate_ids_filter(self):
        bm25 = BM25Index()
        texts = ["attention transformer", "image cnn", "attention network"]
        bm25.build(texts)
        # Only search indices 0 and 1
        result = bm25.top_k("attention", k=3, doc_indices=[0, 1])
        assert all(r in [0, 1] for r in result)
        assert 2 not in result  # Index 2 excluded

    def test_zero_score_for_no_match(self):
        bm25 = BM25Index()
        bm25.build(["hello world", "foo bar"])
        scores = bm25.score("quantum physics")
        assert all(s == 0.0 for s in scores.values())


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

class TestRRF:
    def test_combines_two_ranked_lists(self):
        # Doc 0 is top in both lists — should have highest RRF score
        list1 = [0, 1, 2, 3]
        list2 = [0, 2, 1, 3]
        scores = reciprocal_rank_fusion([list1, list2])
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_promotes_consensus_doc(self):
        # Doc 5 appears in both lists; doc 0 only in list1
        list1 = [0, 1, 5]
        list2 = [5, 2, 3]
        scores = reciprocal_rank_fusion([list1, list2])
        # Doc 5 appears in both → should beat doc 0 (only in one)
        assert scores[5] > scores[0]

    def test_k_parameter_dampening(self):
        # Higher k = less weight on top rank = more balanced scores
        scores_k10 = reciprocal_rank_fusion([[0, 1, 2]], k=10)
        scores_k100 = reciprocal_rank_fusion([[0, 1, 2]], k=100)
        diff_k10 = scores_k10[0] - scores_k10[2]
        diff_k100 = scores_k100[0] - scores_k100[2]
        assert diff_k10 > diff_k100  # Less dampening → bigger spread


# ── MetadataFilter ────────────────────────────────────────────────────────────

class TestMetadataFilter:
    def test_source_filter(self):
        f = MetadataFilter(source="paper1.pdf")
        assert f.matches({"source": "paper1.pdf", "page": 1, "section": "Intro", "chunk_id": "x"})
        assert not f.matches({"source": "paper2.pdf", "page": 1, "section": "Intro", "chunk_id": "y"})

    def test_source_in_filter(self):
        f = MetadataFilter(source_in={"paper1.pdf", "paper2.pdf"})
        assert f.matches({"source": "paper1.pdf", "page": 1, "section": "Intro", "chunk_id": "a"})
        assert f.matches({"source": "paper2.pdf", "page": 1, "section": "Intro", "chunk_id": "b"})
        assert not f.matches({"source": "paper3.pdf", "page": 1, "section": "Intro", "chunk_id": "c"})

    def test_section_contains_case_insensitive(self):
        f = MetadataFilter(section_contains="method")
        assert f.matches({"source": "p.pdf", "page": 1, "section": "Methodology", "chunk_id": "x"})
        assert f.matches({"source": "p.pdf", "page": 1, "section": "Methods", "chunk_id": "y"})
        assert not f.matches({"source": "p.pdf", "page": 1, "section": "Introduction", "chunk_id": "z"})

    def test_page_range_filter(self):
        f = MetadataFilter(page_min=3, page_max=7)
        assert f.matches({"source": "p.pdf", "page": 5, "section": "X", "chunk_id": "a"})
        assert f.matches({"source": "p.pdf", "page": 3, "section": "X", "chunk_id": "b"})
        assert not f.matches({"source": "p.pdf", "page": 2, "section": "X", "chunk_id": "c"})
        assert not f.matches({"source": "p.pdf", "page": 8, "section": "X", "chunk_id": "d"})

    def test_combined_filters_are_anded(self):
        f = MetadataFilter(source="paper1.pdf", section_contains="Method", page_min=2, page_max=6)
        assert f.matches({"source": "paper1.pdf", "page": 4, "section": "Methods", "chunk_id": "a"})
        assert not f.matches({"source": "paper2.pdf", "page": 4, "section": "Methods", "chunk_id": "b"})  # wrong source
        assert not f.matches({"source": "paper1.pdf", "page": 1, "section": "Methods", "chunk_id": "c"})  # out of page range


# ── HybridVectorStore ─────────────────────────────────────────────────────────

class TestHybridVectorStore:
    def test_build_and_search_returns_top_k(self):
        vs = HybridVectorStore()
        vs.build(make_sample_chunks())
        results = vs.search("attention transformer architecture", top_k=3)
        assert len(results) == 3
        assert all("rrf_score" in r for r in results)
        assert all("dense_score" in r for r in results)
        assert all("bm25_score" in r for r in results)

    def test_dense_prefers_semantic_match(self):
        vs = HybridVectorStore()
        vs.build(make_sample_chunks())
        results = vs.search_dense("self-attention multi-head", top_k=5)
        # Transformer paper chunks should dominate
        sources = [make_sample_chunks()[i]["source"] for i in results]
        transformer_count = sources.count("paper_transformers.pdf")
        assert transformer_count >= 3

    def test_sparse_prefers_exact_term(self):
        vs = HybridVectorStore()
        vs.build(make_sample_chunks())
        # "convolutional" is a very specific term only in CNN paper
        results = vs.search_sparse("convolutional feature map", top_k=5)
        sources = [make_sample_chunks()[i]["source"] for i in results]
        cnn_count = sources.count("paper_cnn.pdf")
        assert cnn_count >= 3

    def test_metadata_filter_restricts_results(self):
        vs = HybridVectorStore()
        vs.build(make_sample_chunks())
        f = MetadataFilter(source="paper_cnn.pdf")
        results = vs.search("neural network", top_k=6, metadata_filter=f)
        assert all(r["source"] == "paper_cnn.pdf" for r in results)

    def test_metadata_filter_section(self):
        vs = HybridVectorStore()
        vs.build(make_sample_chunks())
        f = MetadataFilter(section_contains="Method")
        results = vs.search("attention", top_k=10, metadata_filter=f)
        assert all("method" in r.get("section", "").lower() for r in results)

    def test_persist_and_reload(self, tmp_path):
        vs = HybridVectorStore()
        vs.build(make_sample_chunks())
        vs.save(tmp_path)

        vs2 = HybridVectorStore()
        vs2.load(tmp_path)
        results = vs2.search("attention mechanism", top_k=3)
        assert len(results) == 3

    def test_available_sources(self):
        vs = HybridVectorStore()
        vs.build(make_sample_chunks())
        sources = vs.available_sources()
        assert "paper_transformers.pdf" in sources
        assert "paper_cnn.pdf" in sources

    def test_rrf_scores_are_ordered(self):
        vs = HybridVectorStore()
        vs.build(make_sample_chunks())
        results = vs.search("attention transformer", top_k=5)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)  # Must be descending


# ── RetrieverAgent filter extraction ─────────────────────────────────────────

class TestFilterExtraction:
    def test_section_filter_detected(self):
        f = _extract_metadata_filter(
            "what methods were used",
            available_sources=["paper1.pdf"],
            available_sections=["Methods", "Introduction", "Results"],
        )
        assert f is not None
        assert f.section_contains == "Method"

    def test_no_filter_for_generic_query(self):
        f = _extract_metadata_filter(
            "what are the main findings",
            available_sources=["paper1.pdf", "paper2.pdf"],
            available_sections=["Introduction", "Results"],
        )
        assert f is None

    def test_page_filter_detected(self):
        f = _extract_metadata_filter(
            "what does page 5 say about training",
            available_sources=["paper1.pdf"],
            available_sections=["Methods"],
        )
        assert f is not None
        assert f.page_min == 5
        assert f.page_max == 5


# ── RetrieverAgent integration ────────────────────────────────────────────────

class TestRetrieverAgent:
    def test_retrieves_and_appends_chunks(self):
        mock_vs = MagicMock(spec=HybridVectorStore)
        mock_vs.search.return_value = [
            {
                "content": "Test content",
                "source": "paper.pdf",
                "page": 1,
                "section": "Abstract",
                "chunk_id": "id001",
                "rrf_score": 0.9,
                "dense_score": 0.85,
                "bm25_score": 0.7,
            }
        ]
        mock_vs.available_sources.return_value = ["paper.pdf"]
        mock_vs.available_sections.return_value = ["Abstract"]

        agent = RetrieverAgent(mock_vs)
        result = agent(make_state())
        assert len(result["retrieved_chunks"]) == 1
        assert result["retrieval_attempts"] == 1

    def test_uses_rewritten_query(self):
        mock_vs = MagicMock(spec=HybridVectorStore)
        mock_vs.search.return_value = []
        mock_vs.available_sources.return_value = []
        mock_vs.available_sections.return_value = []
        agent = RetrieverAgent(mock_vs)
        agent(make_state(rewritten_query="improved specific query"))
        call_args = mock_vs.search.call_args
        assert call_args[0][0] == "improved specific query"


# ── GraderAgent ───────────────────────────────────────────────────────────────

class TestGraderAgent:
    @pytest.mark.asyncio
    async def test_filters_low_relevance(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"score": 0.2, "reason": "not relevant"}'))]
        )
        agent = GraderAgent(mock_client)
        result = await agent(make_state(retrieved_chunks=[make_chunk()]))
        assert result["graded_chunks"] == []

    @pytest.mark.asyncio
    async def test_keeps_high_relevance(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"score": 0.92, "reason": "highly relevant"}'))]
        )
        agent = GraderAgent(mock_client)
        result = await agent(make_state(retrieved_chunks=[make_chunk()]))
        assert len(result["graded_chunks"]) == 1
        assert result["graded_chunks"][0]["relevance_score"] == 0.92


# ── GeneratorAgent ────────────────────────────────────────────────────────────

class TestGeneratorAgent:
    @pytest.mark.asyncio
    async def test_generates_answer(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="The paper proposes X [Source: paper1.pdf, p.3]."))]
        )
        agent = GeneratorAgent(mock_client)
        result = await agent(make_state(graded_chunks=[make_chunk(relevance_score=0.9)]))
        assert "answer" in result
        assert result["generation_attempts"] == 1


# ── ValidatorAgent ────────────────────────────────────────────────────────────

class TestValidatorAgent:
    @pytest.mark.asyncio
    async def test_marks_grounded(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"is_grounded": true, "unsupported_claims": [], "feedback": "Answer is fully grounded."}'
            ))]
        )
        agent = ValidatorAgent(mock_client)
        result = await agent(make_state(
            answer="The paper uses transformers.",
            graded_chunks=[make_chunk(content="The paper uses transformers for NLP.")],
        ))
        assert result["is_grounded"] is True

    @pytest.mark.asyncio
    async def test_flags_hallucination(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"is_grounded": false, "unsupported_claims": ["99% accuracy"], "feedback": "Not in context."}'
            ))]
        )
        agent = ValidatorAgent(mock_client)
        result = await agent(make_state(
            answer="The model achieves 99% accuracy.",
            graded_chunks=[make_chunk(content="Results show improvement.")],
        ))
        assert result["is_grounded"] is False

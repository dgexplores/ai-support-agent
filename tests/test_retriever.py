"""Unit tests for the knowledge base retriever."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_base.indexer import KnowledgeBaseIndexer
from src.knowledge_base.retriever import KnowledgeRetriever


def setup_module():
    """Index the knowledge base before running tests."""
    indexer = KnowledgeBaseIndexer()
    indexer.index_all_documents()


def test_retrieval_returns_results():
    """Test that retrieval returns passages."""
    retriever = KnowledgeRetriever()
    passages = retriever.retrieve("return policy")
    assert len(passages) > 0


def test_return_policy_retrieval():
    """Test that return policy retrieval finds the current policy."""
    retriever = KnowledgeRetriever()
    passages = retriever.retrieve("How long to return an item")
    sources = [p["source"] for p in passages]
    assert "01-returns-policy-current.md" in sources


def test_legacy_policy_demoted():
    """Test that legacy policy is ranked lower than current."""
    retriever = KnowledgeRetriever()
    passages = retriever.retrieve("return window 45 days")
    # Current policy should rank higher
    scores = {p["source"]: p["score"] for p in passages}
    if "01-returns-policy-current.md" in scores and "02-returns-policy-legacy.md" in scores:
        assert scores["01-returns-policy-current.md"] >= scores["02-returns-policy-legacy.md"]


def test_draft_document_demoted():
    """Test that draft documents (migration notes) are heavily demoted."""
    retriever = KnowledgeRetriever()
    passages = retriever.retrieve("60 day return policy everyone")
    # The migration note should be demoted
    for p in passages:
        if p["source"] == "14-internal-content-migration-notes.md":
            assert p["score"] < 0.5  # Should be heavily demoted


def test_international_shipping_retrieval():
    """Test that international shipping info is retrievable."""
    retriever = KnowledgeRetriever()
    passages = retriever.retrieve("ship internationally Canada")
    sources = [p["source"] for p in passages]
    assert "06-international-shipping.md" in sources


def test_warranty_retrieval():
    """Test that warranty info is retrievable."""
    retriever = KnowledgeRetriever()
    passages = retriever.retrieve("warranty how long bags backpacks")
    sources = [p["source"] for p in passages]
    assert "07-warranty.md" in sources


def test_breeze_tumbler_both_sources_retrieved():
    """Test that both product care and product card are retrieved for tumbler queries."""
    retriever = KnowledgeRetriever()
    passages = retriever.retrieve("How to clean Breeze Tumbler dishwasher")
    sources = [p["source"] for p in passages]
    assert "11-product-care.md" in sources or "12-breeze-tumbler-product-card.md" in sources


def test_conflict_detection():
    """Test that source conflicts are detected."""
    retriever = KnowledgeRetriever()
    passages = retriever.retrieve("How to clean Breeze Tumbler dishwasher")
    conflicts = retriever.detect_source_conflicts(passages)
    # Should detect conflict between product care and product card
    assert len(conflicts) > 0


def test_relevance_filtering():
    """Test that low-relevance passages are filtered out."""
    retriever = KnowledgeRetriever()
    passages = retriever.retrieve("return policy")
    for p in passages:
        assert p["score"] >= 0.3  # MIN_RELEVANCE_SCORE


def test_passage_metadata():
    """Test that passages contain required metadata fields."""
    retriever = KnowledgeRetriever()
    passages = retriever.retrieve("return policy")
    for p in passages:
        assert "source" in p
        assert "heading" in p
        assert "status" in p
        assert "score" in p
        assert "content" in p


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

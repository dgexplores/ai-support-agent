"""Unit tests for document metadata parsing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_base.metadata import (
    parse_front_matter, DocumentMeta, chunk_document,
)


def test_parse_active_document():
    """Test parsing an active document's front matter."""
    content = """---
document_id: RET-2026-01
title: Returns Policy
status: active
effective_date: 2026-04-01
audience: customer
policy_authority: official
supersedes: RET-2024-01
---

# Returns Policy

## Standard return window

Customers may return within 30 days.
"""
    meta, body = parse_front_matter(content)
    assert meta.document_id == "RET-2026-01"
    assert meta.status == "active"
    assert meta.is_active is True
    assert meta.is_superseded is False
    assert meta.is_official is True
    assert meta.is_customer_facing is True
    assert "Returns Policy" in body


def test_parse_superseded_document():
    """Test parsing a superseded document."""
    content = """---
document_id: RET-2024-01
title: Legacy Returns
status: superseded
effective_date: 2024-01-01
superseded_date: 2026-04-01
audience: customer
policy_authority: official
---

# Legacy Returns

## Return window

45 days to return.
"""
    meta, body = parse_front_matter(content)
    assert meta.status == "superseded"
    assert meta.is_superseded is True
    # Superseded docs are NOT customer-facing for answers (should not be used as authority)
    assert meta.is_customer_facing is False


def test_parse_draft_internal_document():
    """Test parsing a draft internal document (migration notes)."""
    content = """---
document_id: MIG-TEST-04
title: Migration Notes
status: draft
audience: internal
policy_authority: none
customer_answering: false
---

# Migration Notes

Draft content here.
"""
    meta, body = parse_front_matter(content)
    assert meta.status == "draft"
    assert meta.is_draft is True
    assert meta.audience == "internal"
    assert meta.customer_answering is False
    assert meta.is_customer_facing is False


def test_precedence_active_highest():
    """Test that active documents have highest precedence."""
    active = DocumentMeta(status="active", policy_authority="official", audience="customer")
    superseded = DocumentMeta(status="superseded", policy_authority="official", audience="customer")
    draft = DocumentMeta(status="draft", policy_authority="none", audience="internal")
    
    assert active.precedence_score > superseded.precedence_score
    assert superseded.precedence_score > draft.precedence_score


def test_chunk_document():
    """Test document chunking."""
    content = """# Returns Policy

## Standard return window

Customers may return within 30 days.

## Item condition

Items must be unused and in resalable condition.
"""
    meta = DocumentMeta(
        document_id="RET-2026-01",
        status="active",
        policy_authority="official",
        audience="customer",
    )
    chunks = chunk_document(content, meta, "01-returns-policy-current.md")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "source" in chunk
        assert "content" in chunk
        assert chunk["source"] == "01-returns-policy-current.md"
        assert chunk["status"] == "active"


def test_front_matter_custom_answer_flag():
    """Test parsing customer_answering flag."""
    content = """---
document_id: MIG-01
title: Notes
status: draft
customer_answering: false
---

# Notes
"""
    meta, _ = parse_front_matter(content)
    assert meta.customer_answering is False
    assert meta.is_customer_facing is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

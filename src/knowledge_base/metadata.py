"""Document metadata parsing and precedence rules for the knowledge base."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DocumentMeta:
    """Parsed front matter metadata from a knowledge base document."""
    document_id: str = ""
    title: str = ""
    status: str = ""  # active, superseded, draft
    effective_date: str = ""
    last_reviewed: str = ""
    audience: str = ""  # customer, internal
    policy_authority: str = ""  # official, none
    supersedes: str = ""
    superseded_by: str = ""
    superseded_date: str = ""
    customer_answering: Optional[bool] = None

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_superseded(self) -> bool:
        return self.status == "superseded"

    @property
    def is_draft(self) -> bool:
        return self.status == "draft"

    @property
    def is_official(self) -> bool:
        return self.policy_authority == "official"

    @property
    def is_customer_facing(self) -> bool:
        """Whether this document should be used for customer answers."""
        if self.audience == "internal" and self.customer_answering is False:
            return False
        if self.is_draft:
            return False
        if self.is_superseded:
            return False
        return True

    @property
    def precedence_score(self) -> float:
        """Higher score = higher precedence for retrieval ranking."""
        score = 0.0
        # Active documents rank higher
        if self.is_active:
            score += 10.0
        elif self.is_superseded:
            score -= 5.0
        elif self.is_draft:
            score -= 10.0

        # Official policy ranks higher
        if self.is_official:
            score += 5.0

        # Customer-facing documents rank higher for customer queries
        if self.is_customer_facing:
            score += 3.0

        # More recent effective date ranks slightly higher
        if self.effective_date:
            try:
                year = int(self.effective_date[:4])
                score += (year - 2024) * 0.5
            except (ValueError, IndexError):
                pass

        return score


def parse_front_matter(content: str) -> tuple[DocumentMeta, str]:
    """Parse YAML-like front matter from a markdown document.
    
    Returns:
        Tuple of (DocumentMeta, content_without_front_matter)
    """
    meta = DocumentMeta()
    
    # Match front matter between --- delimiters
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
    if not match:
        return meta, content
    
    front_matter = match.group(1)
    body = match.group(2)
    
    # Parse simple YAML-like key-value pairs
    for line in front_matter.split('\n'):
        line = line.strip()
        if ':' in line:
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip()
            
            if key == 'document_id':
                meta.document_id = value
            elif key == 'title':
                meta.title = value
            elif key == 'status':
                meta.status = value
            elif key == 'effective_date':
                meta.effective_date = value
            elif key == 'last_reviewed':
                meta.last_reviewed = value
            elif key == 'audience':
                meta.audience = value
            elif key == 'policy_authority':
                meta.policy_authority = value
            elif key == 'supersedes':
                meta.supersedes = value
            elif key == 'superseded_by':
                meta.superseded_by = value
            elif key == 'superseded_date':
                meta.superseded_date = value
            elif key == 'customer_answering':
                meta.customer_answering = value.lower() == 'true'
    
    return meta, body


def parse_heading_hierarchy(content: str) -> list[dict]:
    """Parse markdown headings to create a hierarchy of sections.
    
    Returns list of {level, heading, content} dicts.
    """
    sections = []
    current_section = None
    current_content_lines = []
    
    for line in content.split('\n'):
        heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading_match:
            # Save previous section
            if current_section is not None:
                current_section['content'] = '\n'.join(current_content_lines).strip()
                sections.append(current_section)
            
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            current_section = {'level': level, 'heading': heading, 'content': ''}
            current_content_lines = []
        else:
            current_content_lines.append(line)
    
    # Save last section
    if current_section is not None:
        current_section['content'] = '\n'.join(current_content_lines).strip()
        sections.append(current_section)
    
    return sections


def chunk_document(content: str, meta: DocumentMeta, source_file: str,
                   chunk_size: int = 800, chunk_overlap: int = 150) -> list[dict]:
    """Split a document into overlapping chunks with metadata.
    
    Chunks are created at paragraph/section boundaries when possible.
    Each chunk carries metadata about its source document.
    """
    sections = parse_heading_hierarchy(content)
    chunks = []
    
    # Build chunks from sections
    current_chunk = ""
    current_heading = ""
    current_heading_path = []
    
    for section in sections:
        heading_text = section['heading']
        section_content = section['content']
        
        # Build heading path
        while len(current_heading_path) >= section['level']:
            current_heading_path.pop()
        current_heading_path.append(heading_text)
        heading_path_str = " > ".join(current_heading_path)
        
        # If adding this section would exceed chunk size, save current chunk
        candidate = current_chunk + f"\n\n## {heading_text}\n\n{section_content}" if current_chunk else f"## {heading_text}\n\n{section_content}"
        
        if len(candidate) > chunk_size and current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "source": source_file,
                "document_id": meta.document_id,
                "heading": current_heading,
                "heading_path": " > ".join(current_heading_path[:-1]) if len(current_heading_path) > 1 else "",
                "status": meta.status,
                "policy_authority": meta.policy_authority,
                "audience": meta.audience,
                "effective_date": meta.effective_date,
                "precedence_score": meta.precedence_score,
                "is_customer_facing": meta.is_customer_facing,
            })
            # Start new chunk with overlap from end of previous
            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
            current_chunk = overlap_text + f"\n\n## {heading_text}\n\n{section_content}"
        else:
            current_chunk = candidate
        
        current_heading = heading_path_str
    
    # Save final chunk
    if current_chunk.strip():
        chunks.append({
            "content": current_chunk.strip(),
            "source": source_file,
            "document_id": meta.document_id,
            "heading": current_heading,
            "heading_path": " > ".join(current_heading_path[:-1]) if len(current_heading_path) > 1 else "",
            "status": meta.status,
            "policy_authority": meta.policy_authority,
            "audience": meta.audience,
            "effective_date": meta.effective_date,
            "precedence_score": meta.precedence_score,
            "is_customer_facing": meta.is_customer_facing,
        })
    
    return chunks

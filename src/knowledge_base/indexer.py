"""Knowledge base indexer - chunks documents and stores them in ChromaDB."""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.config import (
    KNOWLEDGE_BASE_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
)
from src.knowledge_base.metadata import (
    DocumentMeta, parse_front_matter, chunk_document,
)
from src.observability.logger import logger


class KnowledgeBaseIndexer:
    """Indexes knowledge base documents into ChromaDB with metadata."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        # Delete existing collection to ensure fresh indexing
        try:
            self.client.delete_collection("knowledge_base")
        except Exception:
            pass
        
        self.collection = self.client.create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )
        self.chunks_indexed = 0

    def index_all_documents(self) -> int:
        """Index all markdown files in the knowledge base directory.
        
        Returns the total number of chunks indexed.
        """
        md_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))
        logger.info(f"Indexing {len(md_files)} knowledge base documents...")

        all_chunks = []
        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            meta, body = parse_front_matter(content)
            chunks = chunk_document(
                body, meta, md_file.name,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            all_chunks.extend(chunks)
            logger.info(
                f"  {md_file.name}: {len(chunks)} chunks "
                f"(status={meta.status}, authority={meta.policy_authority}, "
                f"audience={meta.audience}, precedence={meta.precedence_score:.1f})"
            )

        if not all_chunks:
            logger.warning("No chunks found in knowledge base!")
            return 0

        # Batch insert into ChromaDB
        batch_size = 50
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            ids = [f"chunk_{i + j}" for j in range(len(batch))]
            documents = [c["content"] for c in batch]
            metadatas = [
                {
                    "source": c["source"],
                    "document_id": c["document_id"],
                    "heading": c["heading"],
                    "heading_path": c.get("heading_path", ""),
                    "status": c["status"],
                    "policy_authority": c["policy_authority"],
                    "audience": c["audience"],
                    "effective_date": c.get("effective_date", ""),
                    "precedence_score": c["precedence_score"],
                    "is_customer_facing": c["is_customer_facing"],
                }
                for c in batch
            ]
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        self.chunks_indexed = len(all_chunks)
        logger.info(f"Indexed {self.chunks_indexed} total chunks into ChromaDB.")
        return self.chunks_indexed


def get_or_create_indexer() -> KnowledgeBaseIndexer:
    """Get or create the knowledge base indexer."""
    indexer = KnowledgeBaseIndexer()
    if indexer.chunks_indexed == 0:
        indexer.index_all_documents()
    return indexer

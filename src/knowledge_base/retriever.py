"""RAG retriever with precedence-aware ranking."""

import chromadb
from chromadb.config import Settings
from typing import Optional

from src.config import (
    CHROMA_DIR, TOP_K_RESULTS, MIN_RELEVANCE_SCORE,
)
from src.observability.logger import logger


class KnowledgeRetriever:
    """Retrieves relevant passages from the indexed knowledge base."""

    def __init__(self):
        self._client = None

    def _get_collection(self):
        """Get a fresh collection reference each time."""
        try:
            if self._client is None:
                self._client = chromadb.PersistentClient(
                    path=str(CHROMA_DIR),
                    settings=Settings(anonymized_telemetry=False),
                )
            return self._client.get_collection("knowledge_base")
        except Exception as e:
            logger.warning(f"Knowledge base collection not available: {e}")
            return None

    def refresh(self):
        """Force refresh the client reference (call after indexing)."""
        self._client = None

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        where_filter: Optional[dict] = None,
    ) -> list[dict]:
        """Retrieve relevant passages for a query."""
        col = self._get_collection()
        if col is None:
            logger.error("Knowledge base not available for retrieval")
            return []

        n_candidates = min(top_k * 3, 50)

        try:
            results = col.query(
                query_texts=[query],
                n_results=n_candidates,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

        if not results["documents"] or not results["documents"][0]:
            return []

        passages = []
        for i, (doc, meta, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            score = 1.0 - distance

            is_customer_facing = meta.get("is_customer_facing", True)
            if not is_customer_facing:
                score *= 0.3

            if meta.get("status") == "active" and meta.get("policy_authority") == "official":
                score *= 1.1

            if meta.get("status") == "superseded":
                score *= 0.5

            if meta.get("status") == "draft":
                score *= 0.2

            passages.append({
                "content": doc,
                "source": meta.get("source", "unknown"),
                "document_id": meta.get("document_id", ""),
                "heading": meta.get("heading", ""),
                "heading_path": meta.get("heading_path", ""),
                "status": meta.get("status", ""),
                "policy_authority": meta.get("policy_authority", ""),
                "audience": meta.get("audience", ""),
                "effective_date": meta.get("effective_date", ""),
                "score": score,
                "is_customer_facing": meta.get("is_customer_facing", True),
            })

        passages.sort(key=lambda p: p["score"], reverse=True)
        passages = passages[:top_k]
        passages = [p for p in passages if p["score"] >= MIN_RELEVANCE_SCORE]

        return passages

    def retrieve_for_policy(self, query: str) -> list[dict]:
        """Retrieve only active, official, customer-facing policy documents."""
        return self.retrieve(
            query,
            where_filter={
                "$and": [
                    {"status": "active"},
                    {"policy_authority": "official"},
                    {"is_customer_facing": True},
                ]
            },
        )

    def detect_source_conflicts(self, passages: list[dict]) -> list[dict]:
        """Detect conflicts between retrieved passages from different sources."""
        conflicts = []

        heading_groups: dict[str, list[dict]] = {}
        for p in passages:
            heading = p.get("heading", "").lower()
            if heading:
                heading_groups.setdefault(heading, []).append(p)

        for heading, group in heading_groups.items():
            sources = set(p["source"] for p in group)
            if len(sources) > 1:
                statuses = set(p["status"] for p in group)
                if "active" in statuses and "superseded" in statuses:
                    continue
                if len(sources) > 1 and all(p["status"] == "active" for p in group):
                    conflicts.append({
                        "heading": heading,
                        "sources": list(sources),
                        "passages": group,
                    })

        return conflicts

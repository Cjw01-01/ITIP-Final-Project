"""
RAG retrieval from Qdrant A (job_postings, hr_policies, placement_briefs).

Collection-specific search functions for each specialist agent,
plus a general multi-collection search.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

COLLECTIONS = ("job_postings", "hr_policies", "placement_briefs")


def embed_query(client: Any, model: str, text: str) -> list[float]:
    r = client.embeddings.create(model=model, input=[text])
    return r.data[0].embedding


def _search(
    qdrant: QdrantClient,
    collection: str,
    vector: list[float],
    limit: int = 5,
    qdrant_filter: Filter | None = None,
) -> list[dict]:
    hits = qdrant.search(
        collection_name=collection,
        query_vector=vector,
        limit=limit,
        with_payload=True,
        query_filter=qdrant_filter,
    )
    out: list[dict] = []
    for h in hits:
        payload = h.payload or {}
        out.append({
            "score": h.score,
            "collection": collection,
            "text": payload.get("text", ""),
            "source_id": payload.get("source_id"),
            "section": payload.get("section"),
            "title": payload.get("title"),
            "track": payload.get("track"),
            "doc_type": payload.get("doc_type"),
            "company": payload.get("company"),
            "category": payload.get("category"),
        })
    return out


# --- Specialist-specific search functions (proposal §Agent A specialists) ---


def search_jobs(
    qdrant: QdrantClient,
    embed_client: Any,
    embed_model: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """Job Search Agent tool: RAG over job_postings collection."""
    vec = embed_query(embed_client, embed_model, query)
    return _search(qdrant, "job_postings", vec, limit=limit)


def search_policies(
    qdrant: QdrantClient,
    embed_client: Any,
    embed_model: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """Policy Agent tool: RAG over hr_policies collection. Strictly grounded."""
    vec = embed_query(embed_client, embed_model, query)
    return _search(qdrant, "hr_policies", vec, limit=limit)


def search_placement_briefs(
    qdrant: QdrantClient,
    embed_client: Any,
    embed_model: str,
    query: str,
    limit: int = 5,
    track: str | None = None,
) -> list[dict]:
    """BMW Placement Agent tool: RAG over placement_briefs with optional track metadata filter."""
    vec = embed_query(embed_client, embed_model, query)
    qf = None
    if track:
        qf = Filter(must=[FieldCondition(key="track", match=MatchValue(value=track))])
    return _search(qdrant, "placement_briefs", vec, limit=limit, qdrant_filter=qf)


def search_job_postings_for_screening(
    qdrant: QdrantClient,
    embed_client: Any,
    embed_model: str,
    query: str,
    limit: int = 3,
) -> list[dict]:
    """Candidate Screener: find relevant job postings to screen candidates against."""
    vec = embed_query(embed_client, embed_model, query)
    return _search(qdrant, "job_postings", vec, limit=limit)


# --- General multi-collection ---


def retrieve_all_collections(
    qdrant: QdrantClient,
    embed_client: Any,
    embed_model: str,
    query: str,
    per_collection: int = 4,
) -> tuple[list[dict], str]:
    vec = embed_query(embed_client, embed_model, query)
    notes: list[str] = []
    merged: list[dict] = []
    for name in COLLECTIONS:
        try:
            merged.extend(_search(qdrant, name, vec, limit=per_collection))
        except Exception as e:
            notes.append(f"{name}: {e!s}")
    merged.sort(key=lambda x: x.get("score") or 0, reverse=True)
    status = "ok" if not notes else "; ".join(notes)
    return merged, status


def format_context(hits: list[dict], max_chars: int = 12000) -> str:
    parts: list[str] = []
    n = 0
    for h in hits:
        block = (
            f"[{h['collection']} | source={h.get('source_id')} | score={h.get('score', 0):.3f}]\n"
            f"{h.get('text', '')}\n"
        )
        if n + len(block) > max_chars:
            break
        parts.append(block)
        n += len(block)
    return "\n---\n".join(parts) if parts else "(no retrieved context)"

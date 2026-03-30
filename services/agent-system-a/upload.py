"""
Live PDF upload handler for Agent System A.

Accepts a PDF file + category, extracts text, chunks, embeds, and upserts
to the appropriate Qdrant collection. Supports optional structured metadata
passed alongside the file (candidate_name, bmw_track_label, etc.).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import get_embed_client_and_model, qdrant_a_url

logger = logging.getLogger("itip.upload")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PDF = PROJECT_ROOT / "data" / "pdfs"
VECTOR_SIZE = 1536
ENCODING = tiktoken.get_encoding("cl100k_base")

CATEGORY_MAP = {
    "cvs": {"collection": "candidate_profiles", "qdrant": "b", "max_tokens": 400, "overlap": 80},
    "policies": {"collection": "hr_policies", "qdrant": "a", "max_tokens": 500, "overlap": 100},
    "job_listings": {"collection": "job_postings", "qdrant": "a", "max_tokens": 350, "overlap": 50},
    "placement_briefs": {"collection": "placement_briefs", "qdrant": "a", "max_tokens": 450, "overlap": 90},
}


def _get_qdrant_clients() -> tuple[QdrantClient, QdrantClient]:
    url_a = qdrant_a_url()
    url_b = (os.getenv("QDRANT_B_URL") or "http://localhost:6334").rstrip("/")
    return QdrantClient(url=url_a, timeout=10), QdrantClient(url=url_b, timeout=10)


def _ensure_collection(client: QdrantClient, name: str) -> None:
    try:
        client.get_collection(name)
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def _chunk_by_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    if not text.strip():
        return []
    tokens = ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return [ENCODING.decode(tokens)]
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(ENCODING.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = max(0, end - overlap)
    return chunks


def _embed_texts(embed_client: Any, embed_model: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    r = embed_client.embeddings.create(model=embed_model, input=texts)
    ordered = sorted(r.data, key=lambda x: x.index)
    return [row.embedding for row in ordered]


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    import io
    import pdfplumber

    pages_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = (page.extract_text() or "").strip()
            for tbl in page.extract_tables():
                if tbl and tbl[0]:
                    headers = [str(c or "") for c in tbl[0]]
                    md = "| " + " | ".join(headers) + " |\n"
                    md += "| " + " | ".join("---" for _ in headers) + " |\n"
                    for row in tbl[1:]:
                        cells = [str(c or "") for c in row]
                        md += "| " + " | ".join(cells) + " |\n"
                    text += "\n\n" + md
            pages_text.append(text)
    return "\n\n---\n\n".join(pages_text)


def _extract_skills_from_cv_text(text: str) -> list[str]:
    """Parse skills from a CV's SKILLS section (pipe or comma separated)."""
    import re
    skills_match = re.search(r"SKILLS\s*\n(.*?)(?:\n[A-Z]{2,}|\Z)", text, re.DOTALL | re.IGNORECASE)
    if not skills_match:
        return []
    raw = skills_match.group(1).strip()
    if "|" in raw:
        parts = [s.strip() for s in raw.split("|") if s.strip()]
    elif "," in raw:
        parts = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        parts = [s.strip() for s in raw.split("\n") if s.strip()]
    cleaned = []
    for p in parts:
        p = re.sub(r"^[-*]\s*", "", p).strip()
        if p and len(p) < 60:
            cleaned.append(p)
    return cleaned


def process_upload(
    filename: str,
    pdf_bytes: bytes,
    category: str,
    metadata: dict | None = None,
) -> dict:
    """
    Process a single uploaded PDF: extract text, chunk, embed, upsert.

    Returns a summary dict with status, points_added, collection.
    """
    if category not in CATEGORY_MAP:
        return {"status": "error", "detail": f"Unknown category: {category}. Use: {list(CATEGORY_MAP.keys())}"}

    cfg = CATEGORY_MAP[category]
    collection = cfg["collection"]
    max_tokens = cfg["max_tokens"]
    overlap = cfg["overlap"]

    full_text = _extract_pdf_text(pdf_bytes)
    if not full_text.strip():
        return {"status": "error", "detail": "PDF appears empty or unreadable"}

    save_dir = DATA_PDF / category
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    save_path.write_bytes(pdf_bytes)

    source_id = Path(filename).stem
    base_meta = {
        "source_file": filename,
        "collection": collection,
        "content_type": "pdf",
        "source_id": source_id,
    }
    if metadata:
        base_meta.update(metadata)

    if category == "cvs" and "skills" not in base_meta:
        extracted_skills = _extract_skills_from_cv_text(full_text)
        if extracted_skills:
            base_meta["skills"] = extracted_skills
            logger.info("Auto-extracted %d skills from CV: %s", len(extracted_skills), filename)

    chunks = _chunk_by_tokens(full_text, max_tokens, overlap)
    if not chunks:
        return {"status": "error", "detail": "No text chunks extracted"}

    embed_client, embed_model = get_embed_client_and_model()
    vectors = _embed_texts(embed_client, embed_model, chunks)

    qa, qb = _get_qdrant_clients()
    qclient = qb if cfg["qdrant"] == "b" else qa
    _ensure_collection(qclient, collection)

    points = []
    for i, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"itip:{collection}:{source_id}:body:{i}"))
        payload = {
            **base_meta,
            "source_id": source_id if i == 0 else f"{source_id}_chunk_{i}",
            "section": "body",
            "chunk_index": i,
            "text": chunk_text,
        }
        points.append(PointStruct(id=pid, vector=vec, payload=payload))

    qclient.upsert(collection_name=collection, points=points)

    logger.info("Uploaded %s -> %s: %d chunks", filename, collection, len(points))
    return {
        "status": "ok",
        "filename": filename,
        "category": category,
        "collection": collection,
        "chunks": len(points),
        "text_length": len(full_text),
    }

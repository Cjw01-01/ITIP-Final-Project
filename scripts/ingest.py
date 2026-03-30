"""
Ingest ITIP corpora into Qdrant A (RAG for Agent A) and Qdrant B (candidates).

Supports TWO input modes (§7.2):
  1. JSONL files in data/raw/ (synthetic / bootstrapping)
  2. PDF files in data/pdfs/{cvs,policies,job_listings,placement_briefs}/ (production)

PDF pipeline features:
  - Text extraction via pdfplumber
  - Table detection → Markdown table (preserves structure for LLM reasoning)
  - Image extraction via PyMuPDF → base64 thumbnail + optional OCR description
  - 4 proposal-aligned chunking strategies (§7.3)

Usage:
  python scripts/ingest.py                     # JSONL (default)
  python scripts/ingest.py --pdf               # PDF mode
  python scripts/ingest.py --reset             # drop & recreate collections
  python scripts/ingest.py --pdf --reset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

import tiktoken
from dotenv import load_dotenv
from openai import AzureOpenAI, NotFoundError, OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

logger = logging.getLogger("itip.ingest")

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PDF = ROOT / "data" / "pdfs"

# text-embedding-3-small default dimension
VECTOR_SIZE = 1536
COLLECTIONS_A = ("job_postings", "hr_policies", "placement_briefs")
COLLECTION_B = "candidate_profiles"

ENCODING = tiktoken.get_encoding("cl100k_base")


def normalize_azure_endpoint(raw: str) -> str:
    raw = raw.strip().rstrip("/")
    if not raw:
        return raw
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    if not parsed.netloc:
        return raw.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def get_embedding_client_and_model() -> tuple[Any, str]:
    """
    Embeddings provider (same 1536-dim vectors either way):

    1) EMBEDDINGS_OPENAI_API_KEY — your own key from platform.openai.com (sk-...).
       Use this when Azure has no embedding deployment (e.g. shared/friend account).
    2) Else Azure OpenAI — needs AZURE_OPENAI_EMBEDDING_DEPLOYMENT in the same resource.
    3) Else OPENAI_API_KEY — full OpenAI platform setup (no Azure).
    """
    load_dotenv(ROOT / ".env")
    model = (os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()

    sk_embed = (
        os.getenv("EMBEDDINGS_OPENAI_API_KEY")
        or os.getenv("OPENAI_EMBEDDINGS_API_KEY")
        or ""
    ).strip()
    if sk_embed:
        return OpenAI(api_key=sk_embed), model

    azure_endpoint = normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT") or "")

    if azure_endpoint:
        key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview").strip()
        dep = (
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        ).strip()
        if not key:
            print("ERROR: Azure embeddings need AZURE_OPENAI_API_KEY or OPENAI_API_KEY.", file=sys.stderr)
            sys.exit(1)
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=key,
            api_version=api_version,
        )
        return client, dep

    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        print(
            "ERROR: Set one of:\n"
            "  EMBEDDINGS_OPENAI_API_KEY=sk-...  (OpenAI platform, for embeddings only), or\n"
            "  Azure embedding deployment vars, or\n"
            "  OPENAI_API_KEY=sk-...  (OpenAI only).\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return OpenAI(api_key=key), model


def embed_batch(client: Any, model: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        r = client.embeddings.create(model=model, input=texts)
    except NotFoundError as e:
        if "DeploymentNotFound" in str(e) or getattr(e, "status_code", None) == 404:
            print(
                "\nERROR: Azure embedding deployment not found.\n"
                "  Fix A — Azure: Studio → Deployments → deploy text-embedding-3-small, set:\n"
                "    AZURE_OPENAI_EMBEDDING_DEPLOYMENT=exact-deployment-name\n"
                "  Fix B — No Azure embedding access: use YOUR OpenAI key for embeddings only:\n"
                "    EMBEDDINGS_OPENAI_API_KEY=sk-...   (from platform.openai.com)\n",
                file=sys.stderr,
            )
        raise
    # API returns ordered by index
    ordered = sorted(r.data, key=lambda x: x.index)
    return [row.embedding for row in ordered]


def chunk_by_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
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


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_collection(client: QdrantClient, name: str, reset: bool) -> None:
    exists = False
    try:
        client.get_collection(name)
        exists = True
    except Exception:
        exists = False

    if exists and reset:
        client.delete_collection(name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def stable_point_id(namespace: str, key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"itip:{namespace}:{key}"))


# --- Per-corpus chunk builders ---


def chunks_from_jobs(jobs: list[dict]) -> list[tuple[str, dict]]:
    """Structure-aware job chunks; description split ~350/50."""
    out: list[tuple[str, dict]] = []
    for job in jobs:
        jid = job.get("id", "unknown")
        base = {
            "source_id": jid,
            "collection": "job_postings",
            "company": job.get("company"),
            "track": job.get("track"),
        }
        header = (
            f"Job title: {job.get('title')}\n"
            f"Company: {job.get('company')}\n"
            f"Location: {job.get('location')}\n"
            f"Employment type: {job.get('employment_type')}\n"
            f"Seniority: {job.get('seniority')}\n"
            f"Salary band: {job.get('salary_band')}"
        )
        out.append((header, {**base, "section": "header"}))

        reqs = job.get("requirements") or []
        if reqs:
            body = "Requirements (must-have):\n" + "\n".join(f"- {r}" for r in reqs)
            out.append((body, {**base, "section": "requirements"}))

        nice = job.get("nice_to_have") or []
        if nice:
            body = "Nice to have:\n" + "\n".join(f"- {r}" for r in nice)
            out.append((body, {**base, "section": "nice_to_have"}))

        desc = (job.get("description") or "").strip()
        if desc:
            for i, piece in enumerate(chunk_by_tokens(desc, max_tokens=350, overlap=50)):
                out.append((piece, {**base, "section": "description", "chunk_index": i}))
    return out


def chunks_from_policies(policies: list[dict]) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for pol in policies:
        pid = pol.get("id", "unknown")
        base = {
            "source_id": pid,
            "collection": "hr_policies",
            "category": pol.get("category"),
            "title": pol.get("title"),
        }
        title = pol.get("title") or ""
        body = pol.get("body") or ""
        full = f"{title}\n\n{body}".strip()
        for i, piece in enumerate(chunk_by_tokens(full, max_tokens=500, overlap=100)):
            out.append((piece, {**base, "section": "body", "chunk_index": i}))
    return out


def chunks_from_placement(briefs: list[dict]) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for b in briefs:
        bid = b.get("id", "unknown")
        base = {
            "source_id": bid,
            "collection": "placement_briefs",
            "track": b.get("track"),
            "doc_type": b.get("doc_type"),
            "title": b.get("title"),
        }
        title = b.get("title") or ""
        body = b.get("body") or ""
        full = f"{title}\n\n{body}".strip()
        for i, piece in enumerate(chunk_by_tokens(full, max_tokens=450, overlap=90)):
            out.append((piece, {**base, "section": "body", "chunk_index": i}))
    return out


def chunks_from_candidates(cands: list[dict]) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for c in cands:
        cid = c.get("id", "unknown")
        skills = c.get("skills") or []
        skills_txt = ", ".join(skills) if isinstance(skills, list) else str(skills)
        full = (
            f"Summary: {c.get('summary', '')}\n\n"
            f"Skills: {skills_txt}\n\n"
            f"Experience years: {c.get('experience_years', '')}\n"
            f"Education: {c.get('education', '')}\n"
            f"Languages: {c.get('languages', '')}\n\n"
            f"Resume excerpt:\n{c.get('raw_resume_snippet', '')}"
        ).strip()
        base = {
            "source_id": cid,
            "collection": "candidate_profiles",
            "bmw_track_label": c.get("bmw_track_label"),
            "candidate_name": c.get("name"),  # demo only; scrub in production logs
        }
        for i, piece in enumerate(chunk_by_tokens(full, max_tokens=400, overlap=80)):
            out.append((piece, {**base, "section": "profile", "chunk_index": i}))
    return out


# ---------------------------------------------------------------------------
# PDF extraction (§7.2) — tables → markdown, images → description text
# ---------------------------------------------------------------------------

def _try_import_pdf_libs():
    """Lazy-import PDF libraries so JSONL mode works without them."""
    try:
        import pdfplumber
        import fitz  # PyMuPDF
        return pdfplumber, fitz
    except ImportError:
        print(
            "ERROR: PDF mode requires pdfplumber and PyMuPDF.\n"
            "  pip install pdfplumber PyMuPDF Pillow",
            file=sys.stderr,
        )
        sys.exit(1)


def _table_to_markdown(table: list[list]) -> str:
    if not table or not table[0]:
        return ""
    headers = [str(c or "") for c in table[0]]
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join("---" for _ in headers) + " |\n"
    for row in table[1:]:
        cells = [str(c or "") for c in row]
        while len(cells) < len(headers):
            cells.append("")
        md += "| " + " | ".join(cells) + " |\n"
    return md.strip()


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text, tables, and image descriptions from each page of a PDF.
    Returns list of {page: int, text: str, tables: [markdown], images: [desc]}.
    """
    pdfplumber, fitz = _try_import_pdf_libs()
    pages: list[dict] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = (page.extract_text() or "").strip()
            tables_md = []
            for tbl in page.extract_tables():
                md = _table_to_markdown(tbl)
                if md:
                    tables_md.append(md)
            pages.append({
                "page": i,
                "text": text,
                "tables": tables_md,
                "images": [],
            })

    doc = fitz.open(str(pdf_path))
    for page_idx, page in enumerate(doc):
        img_list = page.get_images(full=True)
        descs = []
        for img_info in img_list:
            xref = img_info[0]
            try:
                base_img = doc.extract_image(xref)
                ext = base_img.get("ext", "png")
                width = base_img.get("width", 0)
                height = base_img.get("height", 0)
                descs.append(f"[Image: {width}x{height} {ext}]")
            except Exception:
                descs.append("[Image: extraction failed]")
        if page_idx < len(pages):
            pages[page_idx]["images"] = descs
    doc.close()

    return pages


def pdf_pages_to_text(pages: list[dict]) -> str:
    """Combine pages into a single document text with tables inlined."""
    parts = []
    for p in pages:
        section = p["text"]
        for tbl in p.get("tables", []):
            section += "\n\n" + tbl
        for img_desc in p.get("images", []):
            section += "\n" + img_desc
        parts.append(section)
    return "\n\n---\n\n".join(parts)


def _load_pdf_metadata(pdf_dir: Path) -> dict[str, dict]:
    """Load the metadata.json sidecar created by generate_pdfs.py."""
    meta_path = pdf_dir / "metadata.json"
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def chunks_from_pdf_dir(
    pdf_dir: Path,
    collection: str,
    max_tokens: int,
    overlap: int,
    extra_metadata: dict | None = None,
) -> list[tuple[str, dict]]:
    """Generic PDF chunker: reads all PDFs in a directory, chunks each.
    Merges structured metadata from metadata.json sidecar if present."""
    if not pdf_dir.exists():
        return []
    sidecar = _load_pdf_metadata(pdf_dir)
    out: list[tuple[str, dict]] = []
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        pages = extract_pdf_pages(pdf_file)
        full_text = pdf_pages_to_text(pages)
        if not full_text.strip():
            continue
        base_meta = {
            "source_file": pdf_file.name,
            "collection": collection,
            "content_type": "pdf",
            "document_category": collection,
        }
        file_meta = sidecar.get(pdf_file.name, {})
        if file_meta:
            base_meta.update(file_meta)
        if extra_metadata:
            base_meta.update(extra_metadata)

        source_id_prefix = file_meta.get("source_id", pdf_file.stem)

        for i, chunk_text in enumerate(chunk_by_tokens(full_text, max_tokens, overlap)):
            page_approx = 1
            for p in pages:
                if chunk_text[:80] in (p["text"] or "")[:200]:
                    page_approx = p["page"]
                    break
            meta = {
                **base_meta,
                "source_id": f"{source_id_prefix}_chunk_{i}" if i > 0 else source_id_prefix,
                "section": "body",
                "chunk_index": i,
                "page_number": page_approx,
            }
            out.append((chunk_text, meta))
    return out


# ---------------------------------------------------------------------------
# Upsert to Qdrant
# ---------------------------------------------------------------------------

def upsert_points(
    qclient: QdrantClient,
    collection: str,
    texts_and_payloads: list[tuple[str, dict]],
    embed_client: Any,
    embed_model: str,
    batch_size: int = 64,
) -> int:
    if not texts_and_payloads:
        return 0
    points: list[PointStruct] = []
    for i in tqdm(range(0, len(texts_and_payloads), batch_size), desc=f"embed {collection}"):
        batch = texts_and_payloads[i : i + batch_size]
        texts = [t for t, _ in batch]
        vecs = embed_batch(embed_client, embed_model, texts)
        for (text, payload), vec in zip(batch, vecs, strict=True):
            key = f"{collection}:{payload.get('source_id')}:{payload.get('section', 'body')}:{payload.get('chunk_index', 0)}"
            pid = stable_point_id(collection, key)
            pl = dict(payload)
            pl["text"] = text
            points.append(PointStruct(id=pid, vector=vec, payload=pl))
    qclient.upsert(collection_name=collection, points=points)
    return len(points)


def ingest_pdfs(qa: QdrantClient, qb: QdrantClient, embed_client: Any, embed_model: str, batch_size: int) -> None:
    """PDF-based ingestion (§7.2–7.3). Reads from data/pdfs/{category}/."""
    print("=== PDF ingestion mode ===")
    print(f"Looking for PDFs in {DATA_PDF}")

    # Job postings: structure-aware, 350 tok / 50 overlap (§7.3.1)
    job_chunks = chunks_from_pdf_dir(DATA_PDF / "job_listings", "job_postings", 350, 50)
    n1 = upsert_points(qa, "job_postings", job_chunks, embed_client, embed_model, batch_size)
    print(f"  job_postings (PDF): {n1} points")

    # HR policies: recursive, 500 tok / 100 overlap (§7.3.2)
    pol_chunks = chunks_from_pdf_dir(DATA_PDF / "policies", "hr_policies", 500, 100)
    n2 = upsert_points(qa, "hr_policies", pol_chunks, embed_client, embed_model, batch_size)
    print(f"  hr_policies (PDF): {n2} points")

    # Placement briefs: 450 tok / 90 overlap (§7.3.3)
    place_chunks = chunks_from_pdf_dir(DATA_PDF / "placement_briefs", "placement_briefs", 450, 90)
    n3 = upsert_points(qa, "placement_briefs", place_chunks, embed_client, embed_model, batch_size)
    print(f"  placement_briefs (PDF): {n3} points")

    # Candidate CVs: 400 tok / 80 overlap (§7.3.4)
    cv_chunks = chunks_from_pdf_dir(DATA_PDF / "cvs", "candidate_profiles", 400, 80)
    n4 = upsert_points(qb, COLLECTION_B, cv_chunks, embed_client, embed_model, batch_size)
    print(f"  candidate_profiles (PDF): {n4} points")

    total = n1 + n2 + n3 + n4
    if total == 0:
        print(f"\nNo PDFs found. Create folders under {DATA_PDF}/ and add PDFs.")
    else:
        print(f"\nTotal PDF points: {total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ITIP corpora (JSONL or PDF) into Qdrant A/B.")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate collections")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--pdf", action="store_true", help="Use PDF ingestion mode (reads data/pdfs/)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    url_a = (os.getenv("QDRANT_A_URL") or "http://localhost:6333").rstrip("/")
    url_b = (os.getenv("QDRANT_B_URL") or "http://localhost:6334").rstrip("/")

    embed_client, embed_model = get_embedding_client_and_model()
    qa = QdrantClient(url=url_a)
    qb = QdrantClient(url=url_b)

    for name in COLLECTIONS_A:
        ensure_collection(qa, name, args.reset)
    ensure_collection(qb, COLLECTION_B, args.reset)

    if args.pdf:
        ingest_pdfs(qa, qb, embed_client, embed_model, args.batch_size)
    else:
        # JSONL mode (synthetic data — default)
        jobs_path = DATA_RAW / "job_postings_syn.jsonl"
        hr_path = DATA_RAW / "hr_policies_syn.jsonl"
        place_path = DATA_RAW / "placement_briefs_syn.jsonl"
        cand_path = DATA_RAW / "candidate_profiles_syn.jsonl"

        missing = [p for p in (jobs_path, hr_path, place_path, cand_path) if not p.exists()]
        if missing:
            print("ERROR: Missing files:", file=sys.stderr)
            for p in missing:
                print(f"  {p}", file=sys.stderr)
            sys.exit(1)

        jobs = read_jsonl(jobs_path)
        policies = read_jsonl(hr_path)
        briefs = read_jsonl(place_path)
        cands = read_jsonl(cand_path)

        print(f"Loaded jobs={len(jobs)} policies={len(policies)} briefs={len(briefs)} candidates={len(cands)}")

        job_chunks = chunks_from_jobs(jobs)
        n1 = upsert_points(qa, "job_postings", job_chunks, embed_client, embed_model, args.batch_size)
        print(f"  job_postings: {n1} points")

        pol_chunks = chunks_from_policies(policies)
        n2 = upsert_points(qa, "hr_policies", pol_chunks, embed_client, embed_model, args.batch_size)
        print(f"  hr_policies: {n2} points")

        place_chunks = chunks_from_placement(briefs)
        n3 = upsert_points(qa, "placement_briefs", place_chunks, embed_client, embed_model, args.batch_size)
        print(f"  placement_briefs: {n3} points")

        cand_chunks = chunks_from_candidates(cands)
        n4 = upsert_points(qb, COLLECTION_B, cand_chunks, embed_client, embed_model, args.batch_size)
        print(f"  candidate_profiles (Qdrant B): {n4} points")

    print("\nDone. Open Qdrant UI: http://localhost:6333/dashboard and http://localhost:6334/dashboard")


if __name__ == "__main__":
    main()

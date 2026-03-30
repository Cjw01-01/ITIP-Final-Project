"""
Skill-weighted candidate matcher (proposal §Agent System B).

Scoring formula:
  S = 0.4 × semantic_score × 100  +  0.6 × skill_intersection_score

Steps:
  1. GPT parses job description → must-have skills (weight 3×) + nice-to-have (weight 1×)
  2. Embed job description with text-embedding-3-small
  3. Query top-20 candidates from Qdrant B by cosine similarity
  4. Compute composite score per candidate
  5. Return top-K ranked candidates with skill gaps + GPT summary
"""

from __future__ import annotations

import json
import re
from typing import Any

from qdrant_client import QdrantClient

COLLECTION = "candidate_profiles"


# --- Step 1: Parse job description into skills ---


def parse_job_skills(chat_client: Any, chat_model: str, job_description: str) -> dict:
    """Use GPT to extract must-have and nice-to-have skills from a job description."""
    system = (
        "You extract skills from job descriptions. Output ONLY valid JSON, no markdown.\n"
        'Return: {"must_have": ["skill1", ...], "nice_to_have": ["skill1", ...]}\n'
        "Normalize skill names to common forms (e.g. 'Python' not 'python programming language')."
    )
    resp = chat_client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Extract skills from:\n\n{job_description}"},
        ],
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"must_have": [], "nice_to_have": []}
    return {
        "must_have": [s.strip() for s in parsed.get("must_have", []) if s.strip()],
        "nice_to_have": [s.strip() for s in parsed.get("nice_to_have", []) if s.strip()],
    }


# --- Step 2: Embed job description ---


def embed_text(embed_client: Any, embed_model: str, text: str) -> list[float]:
    r = embed_client.embeddings.create(model=embed_model, input=[text])
    return r.data[0].embedding


# --- Step 3: Query Qdrant B for top-N candidates ---


def search_candidates(
    qdrant: QdrantClient,
    vector: list[float],
    limit: int = 20,
) -> list[dict]:
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=limit,
        with_payload=True,
    )
    results: list[dict] = []
    seen_ids: set[str] = set()
    for h in hits:
        payload = h.payload or {}
        sid = payload.get("source_id", "")
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        results.append({
            "source_id": sid,
            "semantic_score": h.score,
            "candidate_name": payload.get("candidate_name", "Unknown"),
            "bmw_track_label": payload.get("bmw_track_label"),
            "text": payload.get("text", ""),
        })
    return results


# --- Step 4: Composite scoring ---


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9+#.]", "", s.lower())


def compute_skill_intersection(
    candidate_text: str,
    must_have: list[str],
    nice_to_have: list[str],
) -> tuple[float, list[str], list[str]]:
    """
    Weighted skill intersection.
    Must-have weight = 3, nice-to-have weight = 1.
    Returns (score_0_to_100, matched_skills, missing_skills).
    """
    text_lower = candidate_text.lower()
    total_weight = 0.0
    matched_weight = 0.0
    matched: list[str] = []
    missing: list[str] = []

    for skill in must_have:
        total_weight += 3.0
        if _normalize(skill) in _normalize(text_lower):
            matched_weight += 3.0
            matched.append(skill)
        else:
            missing.append(skill)

    for skill in nice_to_have:
        total_weight += 1.0
        if _normalize(skill) in _normalize(text_lower):
            matched_weight += 1.0
            matched.append(skill)

    if total_weight == 0:
        return 0.0, matched, missing

    return (matched_weight / total_weight) * 100.0, matched, missing


def rank_candidates(
    candidates: list[dict],
    must_have: list[str],
    nice_to_have: list[str],
    top_k: int = 5,
) -> list[dict]:
    """
    Proposal formula: S = 0.4 × semantic_score × 100 + 0.6 × skill_intersection_score
    """
    scored: list[dict] = []
    for c in candidates:
        skill_score, matched, missing = compute_skill_intersection(
            c["text"], must_have, nice_to_have
        )
        semantic_part = 0.4 * c["semantic_score"] * 100
        skill_part = 0.6 * skill_score
        composite = semantic_part + skill_part
        scored.append({
            "source_id": c["source_id"],
            "candidate_name": c["candidate_name"],
            "bmw_track_label": c["bmw_track_label"],
            "semantic_score": round(c["semantic_score"], 4),
            "skill_intersection_score": round(skill_score, 2),
            "composite_score": round(composite, 2),
            "matched_skills": matched,
            "missing_skills": missing,
        })
    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored[:top_k]


# --- Step 5: GPT summary per candidate ---


def generate_summaries(
    chat_client: Any,
    chat_model: str,
    job_description: str,
    ranked: list[dict],
) -> list[dict]:
    """Add a natural-language 'summary' field to each ranked candidate."""
    for c in ranked:
        prompt = (
            f"Job description (abbreviated):\n{job_description[:800]}\n\n"
            f"Candidate: {c['candidate_name']}\n"
            f"Composite score: {c['composite_score']}\n"
            f"Matched skills: {', '.join(c['matched_skills']) or 'none'}\n"
            f"Missing skills (gaps): {', '.join(c['missing_skills']) or 'none'}\n"
            f"Track: {c.get('bmw_track_label', 'N/A')}\n\n"
            "Write a 2-3 sentence recruiter summary: strengths, gaps, fit recommendation."
        )
        resp = chat_client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "You are a concise technical recruiter assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        c["summary"] = (resp.choices[0].message.content or "").strip()
    return ranked


# --- Full pipeline ---


def match_candidates(
    chat_client: Any,
    chat_model: str,
    embed_client: Any,
    embed_model: str,
    qdrant: QdrantClient,
    job_description: str,
    top_k: int = 5,
    semantic_limit: int = 20,
) -> dict:
    """
    End-to-end matching pipeline.
    Returns {"parsed_skills": {...}, "ranked": [...], "top_k": int}.
    """
    skills = parse_job_skills(chat_client, chat_model, job_description)
    vector = embed_text(embed_client, embed_model, job_description)
    candidates = search_candidates(qdrant, vector, limit=semantic_limit)
    ranked = rank_candidates(candidates, skills["must_have"], skills["nice_to_have"], top_k=top_k)
    ranked = generate_summaries(chat_client, chat_model, job_description, ranked)
    return {
        "parsed_skills": skills,
        "ranked": ranked,
        "top_k": top_k,
        "semantic_candidates_queried": len(candidates),
    }

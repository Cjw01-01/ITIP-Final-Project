"""
Google ADK Skills Matcher Agent (proposal §6).

Uses LiteLLM to route through GPT-4o (§6.2: "ADK's class-based agent definition
is cleaner for this single-responsibility pattern").

The agent has tools for each pipeline step: parse skills, search candidates,
rank them, and generate summaries.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

COLLECTION = "candidate_profiles"

# Word-boundary pattern cache for skill matching
_BOUNDARY_PATTERN_CACHE: dict[str, re.Pattern] = {}


# ---------------------------------------------------------------------------
# Standalone functions (same logic as before, usable outside ADK too)
# ---------------------------------------------------------------------------

def parse_job_skills_fn(chat_client: Any, chat_model: str, job_description: str) -> dict:
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
        logger.warning("Failed to parse skill extraction JSON: %s", text[:200])
        parsed = {"must_have": [], "nice_to_have": []}
    return {
        "must_have": [s.strip() for s in parsed.get("must_have", []) if s.strip()],
        "nice_to_have": [s.strip() for s in parsed.get("nice_to_have", []) if s.strip()],
    }


def embed_text(embed_client: Any, embed_model: str, text: str) -> list[float]:
    r = embed_client.embeddings.create(model=embed_model, input=[text])
    return r.data[0].embedding


def _name_search(qdrant: QdrantClient, query: str) -> list[dict]:
    """Scroll through candidates and find by name substring match."""
    capitalized = re.findall(r"\b[A-Z][a-z]{2,}\b", query)
    words = [w.lower() for w in capitalized]
    if not words:
        return []
    results: list[dict] = []
    try:
        all_pts, _ = qdrant.scroll(collection_name=COLLECTION, limit=200, with_payload=True)
        for p in all_pts:
            name = (p.payload.get("candidate_name") or "").lower()
            if name and any(w in name for w in words):
                results.append({
                    "source_id": p.payload.get("source_id", ""),
                    "semantic_score": 0.95,
                    "candidate_name": p.payload.get("candidate_name", "Unknown"),
                    "bmw_track_label": p.payload.get("bmw_track_label"),
                    "text": p.payload.get("text", ""),
                    "skills": p.payload.get("skills", []),
                    "name_matched": True,
                })
    except Exception:
        pass
    return results


def search_candidates(qdrant: QdrantClient, vector: list[float], limit: int = 20, query_text: str = "") -> list[dict]:
    hits = qdrant.search(
        collection_name=COLLECTION, query_vector=vector,
        limit=limit, with_payload=True,
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
            "skills": payload.get("skills", []),
            "name_matched": False,
        })

    if query_text:
        name_hits = _name_search(qdrant, query_text)
        name_hit_ids = {nh["source_id"] for nh in name_hits}
        for r in results:
            if r["source_id"] in name_hit_ids:
                r["name_matched"] = True
                r["semantic_score"] = max(r["semantic_score"], 0.95)
        for nh in name_hits:
            if nh["source_id"] not in seen_ids:
                seen_ids.add(nh["source_id"])
                results.append(nh)

    return results


def _skill_boundary_pattern(skill: str) -> re.Pattern:
    """Build a word-boundary regex for a skill name (avoids 'java' matching 'javascript')."""
    if skill not in _BOUNDARY_PATTERN_CACHE:
        escaped = re.escape(skill.lower())
        _BOUNDARY_PATTERN_CACHE[skill] = re.compile(
            rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE
        )
    return _BOUNDARY_PATTERN_CACHE[skill]


def compute_skill_intersection(
    candidate_text: str, must_have: list[str], nice_to_have: list[str],
) -> tuple[float, list[str], list[str]]:
    total_weight = matched_weight = 0.0
    matched: list[str] = []
    missing: list[str] = []
    for skill in must_have:
        total_weight += 3.0
        if _skill_boundary_pattern(skill).search(candidate_text):
            matched_weight += 3.0
            matched.append(skill)
        else:
            missing.append(skill)
    for skill in nice_to_have:
        total_weight += 1.0
        if _skill_boundary_pattern(skill).search(candidate_text):
            matched_weight += 1.0
            matched.append(skill)
    if total_weight == 0:
        return 0.0, matched, missing
    return (matched_weight / total_weight) * 100.0, matched, missing


def rank_candidates(
    candidates: list[dict], must_have: list[str], nice_to_have: list[str], top_k: int = 5,
) -> list[dict]:
    scored: list[dict] = []
    name_matched_ids: set[str] = set()
    for c in candidates:
        skill_score, matched, missing = compute_skill_intersection(c["text"], must_have, nice_to_have)
        composite = 0.4 * c["semantic_score"] * 100 + 0.6 * skill_score
        entry = {
            "source_id": c["source_id"],
            "candidate_name": c["candidate_name"],
            "bmw_track_label": c["bmw_track_label"],
            "semantic_score": round(c["semantic_score"], 4),
            "skill_intersection_score": round(skill_score, 2),
            "composite_score": round(composite, 2),
            "matched_skills": matched,
            "missing_skills": missing,
            "candidate_skills": c.get("skills", []),
        }
        scored.append(entry)
        if c.get("name_matched"):
            name_matched_ids.add(c["source_id"])
    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    result = scored[:top_k]
    result_ids = {r["source_id"] for r in result}
    for entry in scored:
        if entry["source_id"] in name_matched_ids and entry["source_id"] not in result_ids:
            result.insert(0, entry)
            result_ids.add(entry["source_id"])
    return result


def generate_summaries(
    chat_client: Any, chat_model: str, job_description: str, ranked: list[dict],
) -> list[dict]:
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
        try:
            resp = chat_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": "You are a concise technical recruiter assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3, max_tokens=200,
            )
            c["summary"] = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("Summary generation failed for %s: %s", c["candidate_name"], e)
            c["summary"] = "(summary unavailable)"
    return ranked


def match_candidates(
    chat_client: Any, chat_model: str,
    embed_client: Any, embed_model: str,
    qdrant: QdrantClient,
    job_description: str, top_k: int = 5, semantic_limit: int = 20,
) -> dict:
    skills = parse_job_skills_fn(chat_client, chat_model, job_description)
    vector = embed_text(embed_client, embed_model, job_description)
    candidates = search_candidates(qdrant, vector, limit=semantic_limit, query_text=job_description)
    ranked = rank_candidates(candidates, skills["must_have"], skills["nice_to_have"], top_k=top_k)
    ranked = generate_summaries(chat_client, chat_model, job_description, ranked)
    return {
        "parsed_skills": skills,
        "ranked": ranked,
        "top_k": top_k,
        "semantic_candidates_queried": len(candidates),
    }


# ---------------------------------------------------------------------------
# Google ADK Agent definition (§6.2) — with registered tools
# ---------------------------------------------------------------------------

_module_clients: dict[str, Any] = {}


def _set_module_clients(chat_client: Any, chat_model: str, embed_client: Any, embed_model: str, qdrant: QdrantClient) -> None:
    """Inject shared clients so ADK tool functions can access them."""
    _module_clients.update({
        "chat_client": chat_client,
        "chat_model": chat_model,
        "embed_client": embed_client,
        "embed_model": embed_model,
        "qdrant": qdrant,
    })


def adk_parse_skills(job_description: str) -> dict:
    """Parse a job description to extract must-have and nice-to-have skills."""
    return parse_job_skills_fn(
        _module_clients["chat_client"],
        _module_clients["chat_model"],
        job_description,
    )


def adk_match_and_rank(job_description: str, top_k: int = 5) -> dict:
    """Full matching pipeline: parse skills, search Qdrant, rank candidates, generate summaries."""
    return match_candidates(
        chat_client=_module_clients["chat_client"],
        chat_model=_module_clients["chat_model"],
        embed_client=_module_clients["embed_client"],
        embed_model=_module_clients["embed_model"],
        qdrant=_module_clients["qdrant"],
        job_description=job_description,
        top_k=top_k,
    )


def build_skills_agent() -> LlmAgent:
    """Build the ADK-based Skills Matcher agent with LiteLLM -> GPT-4o and registered tools."""
    parse_tool = FunctionTool(adk_parse_skills)
    match_tool = FunctionTool(adk_match_and_rank)

    return LlmAgent(
        model=LiteLlm(model="openai/gpt-4o"),
        name="skills_matcher",
        instruction=(
            "You are the Skills Matcher agent for the InMind Talent Intelligence Platform. "
            "Your purpose is to receive a job description, rank candidates from the database "
            "by skill match, and return structured results with match scores and skill gap analysis.\n\n"
            "Available tools:\n"
            "- adk_parse_skills: Extract must-have and nice-to-have skills from a job description\n"
            "- adk_match_and_rank: Full pipeline — parse, search, rank, summarize candidates\n\n"
            "For most requests, use adk_match_and_rank to run the complete pipeline."
        ),
        description=(
            "Independent skills matching service — ranks candidates against job requirements "
            "using embedding similarity and skill-weighted scoring."
        ),
        tools=[parse_tool, match_tool],
    )

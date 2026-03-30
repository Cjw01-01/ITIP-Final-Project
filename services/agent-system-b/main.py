"""
ITIP Agent System B — Skills Matcher (Google ADK + LiteLLM, proposal §6).

Framework: Google ADK with LiteLLM adapter for GPT-4o.
The ADK agent wraps the skill-weighted scoring pipeline (§6.4).

Endpoints: POST /chat, POST /match, POST /match/stream, GET /health.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent import (
    _set_module_clients,
    build_skills_agent,
    embed_text,
    generate_summaries,
    match_candidates,
    parse_job_skills_fn,
    rank_candidates,
    search_candidates,
)
from config import get_chat_client_and_model, get_embed_client_and_model, get_qdrant_b

logger = logging.getLogger("itip.agent_b")

app = FastAPI(
    title="InMind Talent Intelligence — Agent System B (Skills Matcher)",
    version="0.3.0",
    description="Google ADK agent with LiteLLM -> GPT-4o. Skill-weighted candidate ranking.",
)


# ---------------------------------------------------------------------------
# Startup: initialize shared clients and ADK agent
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _init_clients():
    try:
        chat_client, chat_model = get_chat_client_and_model()
        embed_client, embed_model = get_embed_client_and_model()
        qdrant = get_qdrant_b()
        _set_module_clients(chat_client, chat_model, embed_client, embed_model, qdrant)
        app.state.adk_agent = build_skills_agent()
        app.state.chat_client = chat_client
        app.state.chat_model = chat_model
        app.state.embed_client = embed_client
        app.state.embed_model = embed_model
        app.state.qdrant = qdrant
        logger.info("Agent B initialized: ADK agent + clients ready")
    except Exception as e:
        logger.error("Agent B startup warning — clients not ready: %s", e)
        app.state.adk_agent = None


class MatchRequest(BaseModel):
    job_description: str = Field(..., min_length=20, max_length=10000)
    top_k: int = Field(5, ge=1, le=20)
    semantic_limit: int = Field(20, ge=5, le=50)


class CandidateResult(BaseModel):
    source_id: str
    candidate_name: str
    bmw_track_label: str | None
    semantic_score: float
    skill_intersection_score: float
    composite_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    candidate_skills: list[str] = []
    summary: str


class MatchResponse(BaseModel):
    parsed_skills: dict[str, list[str]]
    ranked: list[CandidateResult]
    top_k: int
    semantic_candidates_queried: int


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        qb = get_qdrant_b()
        info = qb.get_collection("candidate_profiles")
        points = info.points_count
        reachable = True
    except Exception:
        points = 0
        reachable = False

    adk_ready = getattr(app.state, "adk_agent", None) is not None
    return {
        "status": "ok",
        "service": "agent-system-b",
        "version": "0.3.0",
        "framework": "google-adk",
        "model_backend": "litellm/openai/gpt-4o",
        "adk_agent_ready": adk_ready,
        "qdrant_b_reachable": reachable,
        "candidate_profiles_points": points,
    }


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    """Chat-style endpoint required by the API-layer rubric.

    Interprets the user message as a job-description query, runs the
    skill-matching pipeline, and returns a natural-language summary.
    """
    import uuid as _uuid

    sid = body.session_id or str(_uuid.uuid4())

    chat_client = getattr(app.state, "chat_client", None)
    if chat_client is None:
        try:
            chat_client, chat_model = get_chat_client_and_model()
            embed_client, embed_model = get_embed_client_and_model()
            qdrant = get_qdrant_b()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    else:
        chat_model = app.state.chat_model
        embed_client = app.state.embed_client
        embed_model = app.state.embed_model
        qdrant = app.state.qdrant

    try:
        result = match_candidates(
            chat_client=chat_client, chat_model=chat_model,
            embed_client=embed_client, embed_model=embed_model,
            qdrant=qdrant,
            job_description=body.message,
            top_k=5,
            semantic_limit=20,
        )
    except Exception as exc:
        logger.error("match_candidates failed: %s", exc)
        raise HTTPException(status_code=500, detail="Matching pipeline error") from exc

    ranked = result.get("ranked", [])
    if not ranked:
        reply = "No matching candidates found for the given query."
    else:
        lines = [f"Found {len(ranked)} matching candidate(s):\n"]
        for i, c in enumerate(ranked, 1):
            lines.append(
                f"{i}. **{c['candidate_name']}** — composite {c['composite_score']:.1f} "
                f"| skills matched: {', '.join(c['matched_skills'][:5]) or 'N/A'}\n"
                f"   {c.get('summary', '')}"
            )
        reply = "\n".join(lines)

    return ChatResponse(reply=reply, session_id=sid)


@app.post("/match", response_model=MatchResponse)
def match(body: MatchRequest) -> MatchResponse:
    chat_client = getattr(app.state, "chat_client", None)
    if chat_client is None:
        try:
            chat_client, chat_model = get_chat_client_and_model()
            embed_client, embed_model = get_embed_client_and_model()
            qdrant = get_qdrant_b()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
    else:
        chat_model = app.state.chat_model
        embed_client = app.state.embed_client
        embed_model = app.state.embed_model
        qdrant = app.state.qdrant

    try:
        result = match_candidates(
            chat_client=chat_client, chat_model=chat_model,
            embed_client=embed_client, embed_model=embed_model,
            qdrant=qdrant,
            job_description=body.job_description,
            top_k=body.top_k,
            semantic_limit=body.semantic_limit,
        )
    except Exception as exc:
        logger.error("match_candidates failed: %s", exc)
        raise HTTPException(status_code=500, detail="Matching pipeline error") from exc
    return MatchResponse(**result)


@app.post("/match/stream")
def match_stream(body: MatchRequest) -> StreamingResponse:
    def event_generator():
        chat_client = getattr(app.state, "chat_client", None)
        if chat_client is None:
            try:
                chat_client_l, chat_model_l = get_chat_client_and_model()
                embed_client_l, embed_model_l = get_embed_client_and_model()
                qdrant_l = get_qdrant_b()
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return
        else:
            chat_client_l = chat_client
            chat_model_l = app.state.chat_model
            embed_client_l = app.state.embed_client
            embed_model_l = app.state.embed_model
            qdrant_l = app.state.qdrant

        yield f"data: {json.dumps({'step': 'parsing_skills'})}\n\n"
        skills = parse_job_skills_fn(chat_client_l, chat_model_l, body.job_description)
        yield f"data: {json.dumps({'step': 'skills_parsed', 'parsed_skills': skills})}\n\n"

        yield f"data: {json.dumps({'step': 'embedding_query'})}\n\n"
        vector = embed_text(embed_client_l, embed_model_l, body.job_description)
        yield f"data: {json.dumps({'step': 'searching_candidates'})}\n\n"
        candidates = search_candidates(qdrant_l, vector, limit=body.semantic_limit)
        yield f"data: {json.dumps({'step': 'candidates_found', 'count': len(candidates)})}\n\n"

        yield f"data: {json.dumps({'step': 'ranking'})}\n\n"
        ranked = rank_candidates(
            candidates, skills["must_have"], skills["nice_to_have"], top_k=body.top_k,
        )
        yield f"data: {json.dumps({'step': 'ranked', 'top_k': len(ranked)})}\n\n"

        yield f"data: {json.dumps({'step': 'generating_summaries'})}\n\n"
        ranked = generate_summaries(chat_client_l, chat_model_l, body.job_description, ranked)

        yield f"data: {json.dumps({'step': 'done', 'ranked': ranked, 'parsed_skills': skills})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

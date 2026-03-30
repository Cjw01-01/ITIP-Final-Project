"""
ITIP Agent System A — LangGraph supervisor + 4 specialists (proposal §5).

Endpoints:
  POST /chat           — main conversational endpoint (LangGraph pipeline)
  POST /chat/stream    — SSE streaming (LangGraph stream_mode="updates")
  GET  /session/{id}   — retrieve session history (PII-scrubbed)
  DELETE /session/{id} — clear session
  GET  /health         — health check

Session persistence via Redis with 24h TTL (§5.6 / §9.3).
Rate-limited via slowapi (§9.4): 30 req/min on /chat, 5 req/min on /chat/voice.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis as redis_lib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from fastapi import UploadFile, File, Form

from agent.graph import compile_graph
from config import try_qdrant_a
from guardrails.input_guard import check_input
from guardrails.output_guard import apply_output_guardrails, scrub_pii_for_logs
from upload import process_upload

logger = logging.getLogger("itip.agent_a")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL = int(os.getenv("SESSION_TTL", "86400"))  # 24h
LOG_DIR = Path(os.getenv("LOG_DIR", "/logs"))

ROLE_SPECIALISTS: dict[str, list[str]] = {
    "job_seeker": ["job_search"],
    "hr": ["candidate_screener", "job_search"],
    "staff": ["policy"],
    "instructor": ["bmw_placement", "candidate_screener"],
    "admin": ["job_search", "policy", "candidate_screener", "bmw_placement"],
}

# ---------------------------------------------------------------------------
# Redis session store (§5.6) — replaces in-memory _sessions dict
# ---------------------------------------------------------------------------

_redis_pool = redis_lib.ConnectionPool.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=3)


def _redis() -> redis_lib.Redis:
    return redis_lib.Redis(connection_pool=_redis_pool)


def _session_key(sid: str) -> str:
    return f"itip:session:{sid}"


def load_session(sid: str) -> list[dict]:
    try:
        raw = _redis().get(_session_key(sid))
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return []


def save_session(sid: str, messages: list[dict]) -> None:
    try:
        _redis().setex(_session_key(sid), SESSION_TTL, json.dumps(messages, default=str))
    except Exception:
        logger.warning("Redis write failed for session %s — session not persisted", sid)


def delete_session_store(sid: str) -> None:
    try:
        _redis().delete(_session_key(sid))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Structured JSON logging (§12.7)
# ---------------------------------------------------------------------------

def _log_request(entry: dict) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "itip.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        logger.debug("Could not write structured log entry")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="InMind Talent Intelligence — Agent System A",
    version="0.3.0",
    description="Multi-agent LangGraph system: supervisor + job_search, policy, candidate_screener, bmw_placement",
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again shortly."})


_graph = compile_graph()


# --- Request / Response models ---

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(None, description="Optional session ID for conversation continuity")
    role: str = Field("admin", description="User role: job_seeker, hr, staff, instructor, admin")


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    route_taken: str
    specialist_used: str
    iterations: int
    qdrant_a_available: bool
    guardrail_blocked: bool


# --- Endpoints ---

@app.get("/health")
def health() -> dict[str, Any]:
    q = try_qdrant_a()
    redis_ok = False
    try:
        _redis().ping()
        redis_ok = True
    except Exception:
        pass
    return {
        "status": "ok",
        "service": "agent-system-a",
        "version": "0.3.0",
        "architecture": "langgraph_supervisor_specialist",
        "specialists": ["job_search", "policy", "candidate_screener", "bmw_placement"],
        "qdrant_a_reachable": q is not None,
        "redis_reachable": redis_ok,
    }


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    t0 = time.time()
    request_id = str(uuid.uuid4())
    allowed = ROLE_SPECIALISTS.get(body.role, ROLE_SPECIALISTS["admin"])

    block_reason = check_input(body.message)
    if block_reason:
        sid = body.session_id or str(uuid.uuid4())
        _log_request({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id, "session_id": sid,
            "routing_path": "guardrail_block", "guardrail_result": "blocked",
            "latency_ms": round((time.time() - t0) * 1000),
        })
        return ChatResponse(
            reply=block_reason, session_id=sid,
            route_taken="guardrail_block", specialist_used="none",
            iterations=0, qdrant_a_available=try_qdrant_a() is not None,
            guardrail_blocked=True,
        )

    sid = body.session_id or str(uuid.uuid4())
    history = load_session(sid)
    history.append({"role": "user", "content": body.message})

    initial_state = {
        "messages": history,
        "next_route": "",
        "specialist_used": "",
        "iteration": 0,
        "session_id": sid,
        "guardrail_block": "",
        "retrieved_context": "",
        "allowed_specialists": allowed,
    }

    try:
        result = _graph.invoke(initial_state)
    except Exception as exc:
        logger.error("LangGraph invoke failed (request=%s): %s", request_id, exc)
        latency = round((time.time() - t0) * 1000)
        _log_request({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id, "session_id": sid,
            "routing_path": "error", "guardrail_result": "pass",
            "latency_ms": latency, "error": str(exc),
        })
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request. Please try again.",
        ) from exc

    all_msgs = result.get("messages", [])
    assistant_reply = ""
    for m in reversed(all_msgs):
        if m.get("role") == "assistant" and m.get("content", "").strip():
            assistant_reply = m["content"]
            break

    retrieved_ctx = result.get("retrieved_context", "")
    assistant_reply = apply_output_guardrails(assistant_reply, retrieved_context=retrieved_ctx)

    save_session(sid, all_msgs)

    latency = round((time.time() - t0) * 1000)
    _log_request({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id, "session_id": sid,
        "routing_path": result.get("specialist_used", ""),
        "route_taken": result.get("next_route", "FINISH"),
        "iterations": result.get("iteration", 0),
        "guardrail_result": "pass",
        "latency_ms": latency,
    })

    return ChatResponse(
        reply=assistant_reply, session_id=sid,
        route_taken=result.get("next_route", "FINISH"),
        specialist_used=result.get("specialist_used", ""),
        iterations=result.get("iteration", 0),
        qdrant_a_available=try_qdrant_a() is not None,
        guardrail_blocked=False,
    )


@app.post("/chat/stream")
@limiter.limit("30/minute")
def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """SSE streaming — emits updates as LangGraph nodes execute."""
    allowed = ROLE_SPECIALISTS.get(body.role, ROLE_SPECIALISTS["admin"])

    def event_generator():
        block_reason = check_input(body.message)
        if block_reason:
            yield f"data: {json.dumps({'type': 'guardrail_block', 'content': block_reason})}\n\n"
            return

        sid = body.session_id or str(uuid.uuid4())
        history = load_session(sid)
        history.append({"role": "user", "content": body.message})

        initial_state = {
            "messages": history,
            "next_route": "",
            "iteration": 0,
            "session_id": sid,
            "guardrail_block": "",
            "retrieved_context": "",
            "allowed_specialists": allowed,
        }

        yield f"data: {json.dumps({'type': 'start', 'session_id': sid})}\n\n"

        accumulated_messages = list(history)
        final_route = ""
        final_iteration = 0
        retrieved_ctx = ""

        for event in _graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in event.items():
                step_data: dict[str, Any] = {"type": "node_update", "node": node_name}
                if node_name == "supervisor":
                    step_data["route"] = update.get("next_route", "")
                    step_data["iteration"] = update.get("iteration", 0)
                    final_route = update.get("next_route", final_route)
                    final_iteration = update.get("iteration", final_iteration)
                new_msgs = update.get("messages", [])
                if new_msgs:
                    accumulated_messages.extend(new_msgs)
                    if node_name != "supervisor":
                        last = new_msgs[-1]
                        step_data["content_preview"] = (last.get("content", ""))[:200]
                if update.get("retrieved_context"):
                    retrieved_ctx = update["retrieved_context"]
                yield f"data: {json.dumps(step_data)}\n\n"

        assistant_reply = ""
        for m in reversed(accumulated_messages):
            if m.get("role") == "assistant" and m.get("content", "").strip():
                assistant_reply = m["content"]
                break

        assistant_reply = apply_output_guardrails(assistant_reply, retrieved_context=retrieved_ctx)
        save_session(sid, accumulated_messages)

        yield f"data: {json.dumps({'type': 'done', 'reply': assistant_reply, 'session_id': sid})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Voice interface (§5.7) — Whisper STT + pipeline + OpenAI TTS
# ---------------------------------------------------------------------------

_WHISPER_FILE_EXTS = frozenset(
    {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"}
)


def _voice_temp_suffix(upload: Any) -> str:
    """Whisper uses the file extension to detect format; do not save M4A as .wav."""
    raw = (getattr(upload, "filename", None) or "").strip()
    ext = Path(raw).suffix.lower()
    if ext in _WHISPER_FILE_EXTS:
        return ext
    return ".webm"


def _http_detail_exc(prefix: str, e: BaseException) -> str:
    c = getattr(e, "__cause__", None)
    extra = f" | cause: {type(c).__name__}: {c}" if c else ""
    return f"{prefix}: {e}{extra}"


def _get_openai_platform_client():
    """Get an OpenAI platform client for Whisper STT + TTS (not available on Azure)."""
    from guardrails.input_guard import _get_openai_platform_client as _get_client
    return _get_client()


@app.post("/chat/voice")
@limiter.limit("5/minute")
async def chat_voice(request: Request):
    """Accept audio file, transcribe via Whisper, run pipeline, return TTS audio."""
    import tempfile
    from starlette.responses import Response

    form = await request.form()
    audio_file = form.get("audio")
    session_id = form.get("session_id", None)
    role = form.get("role", "admin")

    if audio_file is None:
        raise HTTPException(status_code=400, detail="No audio file provided")

    openai_client = _get_openai_platform_client()
    if openai_client is None:
        raise HTTPException(status_code=503, detail="OpenAI platform key not configured for Whisper/TTS")

    audio_bytes = await audio_file.read()
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file exceeds 25MB limit")

    suffix = _voice_temp_suffix(audio_file)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as af:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1", file=af, timeout=120.0
            )
        user_text = (transcript.text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=_http_detail_exc("Whisper transcription failed", e))
    finally:
        os.unlink(tmp_path)

    if not user_text:
        raise HTTPException(status_code=400, detail="Could not transcribe any speech from the audio. Please try again with clearer audio.")

    allowed = ROLE_SPECIALISTS.get(role, ROLE_SPECIALISTS["admin"])
    sid = session_id or str(uuid.uuid4())
    block_reason = check_input(user_text)
    if block_reason:
        reply_text = block_reason
    else:
        history = load_session(sid)
        history.append({"role": "user", "content": user_text})
        result = _graph.invoke({
            "messages": history, "next_route": "", "specialist_used": "",
            "iteration": 0, "session_id": sid, "guardrail_block": "",
            "retrieved_context": "", "allowed_specialists": allowed,
        })
        all_msgs = result.get("messages", [])
        reply_text = ""
        for m in reversed(all_msgs):
            if m.get("role") == "assistant" and m.get("content", "").strip():
                reply_text = m["content"]
                break
        retrieved_ctx = result.get("retrieved_context", "")
        reply_text = apply_output_guardrails(reply_text, retrieved_context=retrieved_ctx)
        save_session(sid, all_msgs)

    try:
        tts_response = openai_client.audio.speech.create(
            model="tts-1", voice="nova", input=reply_text[:4096], timeout=120.0
        )
        audio_content = tts_response.content
        # Header values must be latin-1; Unicode in replies (smart quotes, bullets, etc.)
        # breaks Starlette. Use UTF-8 base64 for text previews (UI decodes).
        def _b64_hdr(text: str, max_chars: int) -> str:
            return base64.standard_b64encode((text or "")[:max_chars].encode("utf-8")).decode("ascii")

        return Response(
            content=audio_content,
            media_type="audio/mpeg",
            headers={
                "X-Transcript-B64": _b64_hdr(user_text, 2000),
                "X-Reply-B64": _b64_hdr(reply_text, 4500),
                "X-Session-Id": sid,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=_http_detail_exc("TTS generation failed", e))


@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    category: str = Form(...),
    metadata_json: str = Form("{}"),
):
    """Upload a PDF and ingest it into the appropriate Qdrant collection."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    if category not in ("cvs", "policies", "job_listings", "placement_briefs"):
        raise HTTPException(status_code=400, detail="category must be: cvs, policies, job_listings, or placement_briefs")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF exceeds 20MB limit")

    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        metadata = {}

    result = process_upload(file.filename, pdf_bytes, category, metadata)
    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("detail", "Upload failed"))

    return result


@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    msgs = load_session(session_id)
    if not msgs:
        raise HTTPException(status_code=404, detail="Session not found")
    safe_msgs = [
        {"role": m.get("role"), "content": scrub_pii_for_logs(m.get("content", ""))}
        for m in msgs
    ]
    return {"session_id": session_id, "messages": safe_msgs, "message_count": len(safe_msgs)}


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    delete_session_store(session_id)
    return {"status": "deleted", "session_id": session_id}

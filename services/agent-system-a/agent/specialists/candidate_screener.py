"""
Specialist 3 — Candidate Screener Agent (proposal §5.4.3).

Workflow:
  1. (Optional) Call local DistilBERT classifier → predict BMW track
  2. RAG screen against job posting in Qdrant A
  3. HTTP call to Agent B /match for skill-weighted ranking
  4. If recruiter approves, call MCP tools to book interview / send assessment

Agent B on port 8001, MCP server on port 8002.
"""

from __future__ import annotations

import httpx

from agent.state import AgentState
from classifier.predict import predict_track
from config import chat_completion_create, get_chat_client_and_model, get_embed_client_and_model, try_qdrant_a
from rag.retrieve import format_context, search_job_postings_for_screening

import logging
import os

logger = logging.getLogger(__name__)

AGENT_B_URL = os.getenv("AGENT_B_URL", "http://localhost:8001").rstrip("/")
MCP_URL = os.getenv("MCP_URL", "http://localhost:8002").rstrip("/")

SYSTEM = """You are the Candidate Screening specialist for the InMind Talent Intelligence Platform.
You help recruiters evaluate, rank, and shortlist candidates for open positions.

You have access to:
1. Job posting context from the knowledge base (below).
2. Candidate ranking results from the Skills Matcher (Agent B), if available.
3. Interview scheduling tools (MCP server): you can find interviewers, book interviews, and send assessments.

When presenting candidate rankings, ALWAYS include this disclaimer:
"Note: AI-generated rankings are advisory only. Final hiring decisions should involve human review."

If the user asks to schedule an interview or send an assessment, include the MCP tool results in your response.

JOB CONTEXT:
{job_context}

CANDIDATE RANKING:
{ranking_context}

SCHEDULING INFO:
{scheduling_context}"""


def _call_agent_b(job_description: str, top_k: int = 5) -> dict | None:
    """HTTP call to Agent B /match endpoint."""
    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(
                f"{AGENT_B_URL}/match",
                json={"job_description": job_description, "top_k": top_k},
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning("Agent B returned HTTP %d: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("Agent B call failed (%s): %s", AGENT_B_URL, e)
    return None


def _call_mcp_find_interviewers(role: str, date: str, track: str | None = None) -> list[dict]:
    """Call MCP server to find available interviewers."""
    try:
        with httpx.Client(timeout=5.0) as client:
            body: dict = {"role": role, "date": date}
            if track:
                body["track"] = track
            resp = client.post(f"{MCP_URL}/tools/find_available_interviewers", json=body)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("MCP find_interviewers returned HTTP %d", resp.status_code)
    except Exception as e:
        logger.warning("MCP find_interviewers failed (%s): %s", MCP_URL, e)
    return []


def _call_mcp_schedule(
    candidate_id: str, interviewer_id: str, dt: str, interview_type: str = "technical"
) -> dict | None:
    """Call MCP server to book an interview."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{MCP_URL}/tools/schedule_interview",
                json={
                    "candidate_id": candidate_id,
                    "interviewer_id": interviewer_id,
                    "datetime": dt,
                    "interview_type": interview_type,
                },
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning("MCP schedule_interview returned HTTP %d", resp.status_code)
    except Exception as e:
        logger.warning("MCP schedule_interview failed: %s", e)
    return None


def _call_mcp_assessment(candidate_id: str, assessment_type: str = "coding") -> dict | None:
    """Call MCP server to dispatch an assessment."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{MCP_URL}/tools/send_assessment",
                json={"candidate_id": candidate_id, "assessment_type": assessment_type},
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning("MCP send_assessment returned HTTP %d", resp.status_code)
    except Exception as e:
        logger.warning("MCP send_assessment failed: %s", e)
    return None


def _format_ranking(result: dict | None) -> str:
    if not result:
        return "(Agent B skills matcher is not available — start it on port 8001)"

    ranked = result.get("ranked", [])
    if not ranked:
        return "(No matching candidates found)"

    lines: list[str] = []
    for i, c in enumerate(ranked, 1):
        cand_skills = c.get("candidate_skills", [])
        skills_line = f"   Candidate's skills: {', '.join(cand_skills)}\n" if cand_skills else ""
        lines.append(
            f"{i}. {c.get('candidate_name', 'Unknown')} "
            f"(score: {c.get('composite_score', 0):.1f}, "
            f"track: {c.get('bmw_track_label', 'N/A')})\n"
            f"{skills_line}"
            f"   Matched: {', '.join(c.get('matched_skills', []))}\n"
            f"   Gaps: {', '.join(c.get('missing_skills', []))}\n"
            f"   {c.get('summary', '')}"
        )
    skills = result.get("parsed_skills", {})
    header = (
        f"Must-have skills: {', '.join(skills.get('must_have', []))}\n"
        f"Nice-to-have: {', '.join(skills.get('nice_to_have', []))}\n\n"
    )
    return header + "\n\n".join(lines)


def _detect_scheduling_intent(user_msg: str) -> dict:
    """Simple keyword detection for scheduling/assessment requests."""
    msg_lower = user_msg.lower()
    return {
        "wants_interview": any(kw in msg_lower for kw in ["book interview", "schedule interview", "book a", "schedule a"]),
        "wants_assessment": any(kw in msg_lower for kw in ["send assessment", "dispatch assessment", "technical assessment"]),
        "wants_interviewers": any(kw in msg_lower for kw in ["available interviewer", "find interviewer", "who can interview"]),
    }


def _build_scheduling_context(user_msg: str, top_candidate: dict | None) -> str:
    intent = _detect_scheduling_intent(user_msg)
    parts: list[str] = []

    if intent["wants_interviewers"] or intent["wants_interview"]:
        track = top_candidate.get("bmw_track_label") if top_candidate else None
        interviewers = _call_mcp_find_interviewers("Technical Role", "2026-03-25", track=track)
        if interviewers:
            lines = []
            for iv in interviewers:
                slots = ", ".join(iv.get("available_slots", []))
                lines.append(f"  {iv['name']} ({iv['specialisation']}) — slots: {slots}")
            parts.append("Available interviewers for 2026-03-25:\n" + "\n".join(lines))
        else:
            parts.append("(No interviewers available or MCP server not running on port 8002)")

    if intent["wants_interview"] and top_candidate:
        interviewers = _call_mcp_find_interviewers("Technical Role", "2026-03-25")
        if interviewers:
            result = _call_mcp_schedule(
                candidate_id=top_candidate.get("source_id", "unknown"),
                interviewer_id=interviewers[0].get("interviewer_id", "int_001"),
                dt="2026-03-25T10:00:00",
                interview_type="technical",
            )
            if result:
                parts.append(
                    f"Interview booked: {result['confirmation_id']}\n"
                    f"  Candidate: {result['candidate']}\n"
                    f"  Interviewer: {result['interviewer']}\n"
                    f"  Time: {result['datetime']}\n"
                    f"  Zoom: {result['zoom_link']}\n"
                    f"  Status: {result['status']}"
                )

    if intent["wants_assessment"] and top_candidate:
        result = _call_mcp_assessment(
            candidate_id=top_candidate.get("source_id", "unknown"),
            assessment_type="coding",
        )
        if result:
            parts.append(
                f"Assessment dispatched: {result['tracking_id']}\n"
                f"  Type: {result['assessment_type']}\n"
                f"  Deadline: {result['deadline']}\n"
                f"  Link: {result['link']}\n"
                f"  Status: {result['status']}"
            )

    return "\n\n".join(parts) if parts else "(no scheduling actions requested)"


def _classify_candidates(ranking_result: dict | None) -> str:
    """Run DistilBERT track classifier on ranked candidates (§5.5)."""
    if not ranking_result or not ranking_result.get("ranked"):
        return ""

    lines: list[str] = []
    for cand in ranking_result["ranked"][:5]:
        text = f"{cand.get('summary', '')} Skills: {', '.join(cand.get('matched_skills', []))}"
        preds = predict_track(text, top_k=2)
        if preds:
            top = preds[0]
            lines.append(
                f"  {cand.get('candidate_name', 'Unknown')}: "
                f"predicted={top['track']} ({top['confidence']:.0%}), "
                f"labeled={cand.get('bmw_track_label', 'N/A')}"
            )
        else:
            lines.append(
                f"  {cand.get('candidate_name', 'Unknown')}: "
                f"(DistilBERT classifier unavailable, using labeled track: {cand.get('bmw_track_label', 'N/A')})"
            )

    if lines:
        return "DistilBERT Track Predictions:\n" + "\n".join(lines)
    return ""


def candidate_screener_node(state: AgentState) -> dict:
    user_msg = ""
    for m in reversed(state.get("messages", [])):
        if m["role"] == "user":
            user_msg = m.get("content", "")
            break

    if not user_msg:
        return {"messages": [{"role": "assistant", "content": "Please describe the role or candidate screening criteria."}]}

    qdrant = try_qdrant_a()
    embed_client, embed_model = get_embed_client_and_model()

    job_hits: list[dict] = []
    if qdrant:
        job_hits = search_job_postings_for_screening(qdrant, embed_client, embed_model, user_msg, limit=3)
    job_context = format_context(job_hits) if job_hits else "(no job posting context available)"

    job_desc_for_b = user_msg
    logger.debug("SCREENER -> Agent B | job_desc_for_b (first 300 chars): %s", job_desc_for_b[:300])
    ranking_result = _call_agent_b(job_desc_for_b, top_k=5)
    if ranking_result:
        names = [c.get("candidate_name", "?") for c in ranking_result.get("ranked", [])]
        logger.debug("SCREENER <- Agent B | returned %d candidates: %s", len(names), names)
    else:
        logger.debug("SCREENER <- Agent B | returned None")
    ranking_context = _format_ranking(ranking_result)

    classifier_context = _classify_candidates(ranking_result)
    if classifier_context:
        ranking_context = f"{ranking_context}\n\n{classifier_context}"

    top_candidate = None
    if ranking_result and ranking_result.get("ranked"):
        top_candidate = ranking_result["ranked"][0]

    scheduling_context = _build_scheduling_context(user_msg, top_candidate)

    chat_client, chat_model = get_chat_client_and_model()
    resp = chat_completion_create(
        chat_client,
        chat_model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM.format(
                    job_context=job_context,
                    ranking_context=ranking_context,
                    scheduling_context=scheduling_context,
                ),
            },
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1500,
    )
    reply = (resp.choices[0].message.content or "").strip()
    return {"messages": [{"role": "assistant", "content": reply}], "retrieved_context": job_context}

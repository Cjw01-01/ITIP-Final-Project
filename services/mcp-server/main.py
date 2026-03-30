"""
ITIP MCP Server — Interview scheduling tools (proposal §8, Table 4).

Built with fastmcp. Exposes 3 tools:
  - find_available_interviewers
  - schedule_interview
  - send_assessment

Also exposes REST endpoints via FastAPI for direct HTTP access from Agent A.

Run:
  cd services/mcp-server
  pip install -r requirements.txt
  python -m uvicorn main:app --reload --port 8002
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from mock_data import book_interview, dispatch_assessment, find_interviewers

# --- FastMCP tool definitions (MCP protocol) ---

mcp = FastMCP(
    "ITIP Interview Scheduler",
    instructions=(
        "You are the interview scheduling service for the InMind Talent Intelligence Platform. "
        "You can find available interviewers, schedule interviews, and dispatch assessments."
    ),
)


@mcp.tool()
def find_available_interviewers(
    role: str,
    date: str,
    track: str | None = None,
) -> list[dict]:
    """Find available interviewers for a given role and date.

    Args:
        role: Job title to interview for (e.g. 'ML Engineer', 'Backend Developer')
        date: ISO 8601 date string (e.g. '2026-03-25')
        track: Optional BMW track filter (AI, Backend, Frontend, Robotics, Simulation)

    Returns:
        List of interviewer objects with name, specialisation, and available_slots.
    """
    return find_interviewers(role, date, track)


@mcp.tool()
def schedule_interview(
    candidate_id: str,
    interviewer_id: str,
    datetime: str,
    interview_type: str,
) -> dict:
    """Schedule an interview for a candidate.

    Args:
        candidate_id: Candidate identifier (e.g. 'cand_syn_001')
        interviewer_id: Interviewer identifier (e.g. 'int_001')
        datetime: ISO 8601 datetime (e.g. '2026-03-25T10:00:00')
        interview_type: One of 'technical', 'cultural', 'final'

    Returns:
        Confirmation object with confirmation_id, zoom_link, and status.
    """
    return book_interview(candidate_id, interviewer_id, datetime, interview_type)


@mcp.tool()
def send_assessment(
    candidate_id: str,
    assessment_type: str,
) -> dict:
    """Dispatch a technical assessment to a candidate.

    Args:
        candidate_id: Candidate identifier (e.g. 'cand_syn_001')
        assessment_type: One of 'coding', 'systems_design', 'bmw_technical', 'behavioural'

    Returns:
        Tracking object with tracking_id, deadline, link, and status.
    """
    return dispatch_assessment(candidate_id, assessment_type)


# --- FastAPI REST wrapper (for direct HTTP calls from Agent A) ---

mcp_asgi = mcp.http_app(path="/mcp")

app = FastAPI(
    title="InMind Talent Intelligence — MCP Server",
    version="0.1.0",
    description="Interview scheduling tools exposed via MCP protocol and REST.",
    lifespan=mcp_asgi.lifespan,
)


class FindInterviewersRequest(BaseModel):
    role: str
    date: str
    track: str | None = None


class ScheduleRequest(BaseModel):
    candidate_id: str
    interviewer_id: str
    datetime: str
    interview_type: str = Field(pattern="^(technical|cultural|final)$")


class AssessmentRequest(BaseModel):
    candidate_id: str
    assessment_type: str = Field(pattern="^(coding|systems_design|bmw_technical|behavioural)$")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "mcp-server",
        "tools": ["find_available_interviewers", "schedule_interview", "send_assessment"],
    }


@app.post("/tools/find_available_interviewers")
def http_find_interviewers(body: FindInterviewersRequest) -> list[dict]:
    return find_interviewers(body.role, body.date, body.track)


@app.post("/tools/schedule_interview")
def http_schedule_interview(body: ScheduleRequest) -> dict:
    return book_interview(body.candidate_id, body.interviewer_id, body.datetime, body.interview_type)


@app.post("/tools/send_assessment")
def http_send_assessment(body: AssessmentRequest) -> dict:
    return dispatch_assessment(body.candidate_id, body.assessment_type)


# Mount MCP protocol transport alongside REST
app.mount("/mcp", mcp_asgi)

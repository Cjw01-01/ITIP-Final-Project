"""LangGraph state for the supervisor-specialist pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    next_route: str
    specialist_used: str
    iteration: int
    session_id: str
    guardrail_block: str
    retrieved_context: str
    allowed_specialists: list[str]


MAX_ITERATIONS = 8

ROUTES = ("job_search", "policy", "candidate_screener", "bmw_placement", "FINISH")

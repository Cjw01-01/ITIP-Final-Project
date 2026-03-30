"""
Supervisor Agent — GPT-4o intent classifier (proposal §Supervisor Agent).

Classifies user intent and routes to one of 4 specialists:
  job_search, policy, candidate_screener, bmw_placement, or FINISH.

Max 8 routing iterations before forced FINISH.
"""

from __future__ import annotations

import json

from agent.state import AgentState, MAX_ITERATIONS, ROUTES
from config import chat_completion_create, get_chat_client_and_model

SUPERVISOR_SYSTEM = """You are the routing supervisor for the InMind Talent Intelligence Platform (ITIP).

Your ONLY job is to classify the user's latest message into exactly ONE route.

Routes:
- "job_search": questions about open positions, job requirements, salary, seniority, applying
- "policy": HR policy questions (leave, benefits, conduct, remote work, working hours, termination)
- "candidate_screener": evaluate, rank, screen, compare, or shortlist candidates for a role
- "bmw_placement": BMW Group / idealworks placement queries, internship tracks, Munich logistics, past cohorts
- "FINISH": the conversation is complete, the user said goodbye, OR the query is clearly out of scope for this platform

Rules:
- If the previous specialist already answered and the user hasn't asked a new question, return "FINISH".
- If the user's message is ambiguous, pick the single best route.
- Return ONLY a JSON object: {"route": "<route_name>"}
- No explanation, no markdown, ONLY the JSON object."""


def _last_role(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") in ("user", "assistant"):
            return m["role"]
    return ""


def supervisor_node(state: AgentState) -> dict:
    iteration = state.get("iteration", 0)
    messages = state.get("messages", [])
    allowed = state.get("allowed_specialists") or list(ROUTES[:-1])

    if iteration >= MAX_ITERATIONS:
        return {
            "next_route": "FINISH",
            "iteration": iteration,
            "messages": [{
                "role": "assistant",
                "content": "I've reached the maximum number of routing steps. Let me know if you need anything else.",
            }],
        }

    if iteration > 0 and _last_role(messages) == "assistant":
        return {
            "next_route": "FINISH",
            "iteration": iteration,
            "messages": [],
        }

    chat_client, chat_model = get_chat_client_and_model()

    system_prompt = SUPERVISOR_SYSTEM

    classify_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        classify_messages.append({"role": m["role"], "content": m.get("content", "")})

    resp = chat_completion_create(
        chat_client,
        chat_model,
        messages=classify_messages,
        temperature=0.0,
        max_tokens=50,
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
        route = parsed.get("route", "FINISH")
    except (json.JSONDecodeError, AttributeError):
        for r in ROUTES:
            if r.lower() in text.lower():
                route = r
                break
        else:
            route = "FINISH"

    if route not in ROUTES:
        route = "FINISH"

    # Enforce role-based access: if LLM picked a specialist outside allowed list, block it
    if route != "FINISH" and route not in allowed:
        return {
            "next_route": "FINISH",
            "iteration": iteration + 1,
            "messages": [{
                "role": "assistant",
                "content": "I'm sorry, but your current role doesn't have access to that functionality. Please contact your administrator if you need access.",
            }],
        }

    if route == "FINISH" and iteration == 0 and not state.get("specialist_used"):
        return {
            "next_route": "FINISH",
            "iteration": iteration + 1,
            "messages": [{
                "role": "assistant",
                "content": "I'm not sure how to help with that. Could you rephrase, or ask me something about open positions, HR policies, candidate screening, or BMW placements?",
            }],
        }

    update: dict = {
        "next_route": route,
        "iteration": iteration + 1,
        "messages": [],
    }
    if route != "FINISH" and not state.get("specialist_used"):
        update["specialist_used"] = route

    return update

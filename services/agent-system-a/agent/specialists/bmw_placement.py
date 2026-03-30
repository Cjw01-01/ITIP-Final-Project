"""
Specialist 4 — BMW Placement Agent (proposal §Specialist 4).

Handles BMW/idealworks placement queries, track requirements, past cohort data.
Tool: search_placement_briefs() → RAG over placement_briefs collection,
with optional track metadata filter (AI, Backend, Frontend, Robotics, Simulation).
"""

from __future__ import annotations

import json

from agent.state import AgentState
from config import chat_completion_create, get_chat_client_and_model, get_embed_client_and_model, try_qdrant_a
from rag.retrieve import format_context, search_placement_briefs

BMW_TRACKS = ("AI", "Backend", "Frontend", "Robotics", "Simulation")

SYSTEM = """You are the BMW/idealworks Placement specialist for the InMind Talent Intelligence Platform.
You help academy graduates understand the 6-month internship placement program in Munich, Germany.

Available tracks: AI, Backend, Frontend, Robotics, Simulation.

Use ONLY the retrieved placement briefs below. Include details about technical stacks, expectations, logistics, and assessment criteria when available.
If a track is specified, focus on that track's information.

CONTEXT:
{context}"""

TRACK_DETECT_SYSTEM = """Given this user message about BMW/idealworks placement, extract the track if mentioned.
Tracks: AI, Backend, Frontend, Robotics, Simulation.
Return ONLY JSON: {"track": "<track_name>"} or {"track": null} if no specific track.
No explanation."""


def _detect_track(chat_client, chat_model: str, user_msg: str) -> str | None:
    try:
        resp = chat_completion_create(
            chat_client,
            chat_model,
            messages=[
                {"role": "system", "content": TRACK_DETECT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=30,
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(text)
        track = parsed.get("track")
        if track and track in BMW_TRACKS:
            return track
    except Exception:
        pass

    msg_upper = user_msg.upper()
    for t in BMW_TRACKS:
        if t.upper() in msg_upper:
            return t
    return None


def bmw_placement_node(state: AgentState) -> dict:
    user_msg = ""
    for m in reversed(state.get("messages", [])):
        if m["role"] == "user":
            user_msg = m.get("content", "")
            break

    if not user_msg:
        return {"messages": [{"role": "assistant", "content": "What would you like to know about BMW/idealworks placement?"}]}

    qdrant = try_qdrant_a()
    if qdrant is None:
        return {"messages": [{"role": "assistant", "content": "The placement knowledge base is currently unavailable."}]}

    chat_client, chat_model = get_chat_client_and_model()
    embed_client, embed_model = get_embed_client_and_model()

    track = _detect_track(chat_client, chat_model, user_msg)
    hits = search_placement_briefs(qdrant, embed_client, embed_model, user_msg, limit=5, track=track)
    context = format_context(hits)

    track_note = f"\n[Filtered to track: {track}]" if track else ""

    resp = chat_completion_create(
        chat_client,
        chat_model,
        messages=[
            {"role": "system", "content": SYSTEM.format(context=context) + track_note},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    reply = (resp.choices[0].message.content or "").strip()
    return {"messages": [{"role": "assistant", "content": reply}], "retrieved_context": context}

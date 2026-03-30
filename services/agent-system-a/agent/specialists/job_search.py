"""
Specialist 1 — Job Search Agent (proposal §Specialist 1).

Handles queries about open positions, requirements, salary, seniority.
Tool: search_jobs() → RAG over job_postings Qdrant collection.
"""

from __future__ import annotations

from agent.state import AgentState
from config import chat_completion_create, get_chat_client_and_model, get_embed_client_and_model, try_qdrant_a
from rag.retrieve import format_context, search_jobs

SYSTEM = """You are the Job Search specialist for the InMind Talent Intelligence Platform.
You help recruiters and candidates find information about open positions at inmind.ai, BMW Group, and idealworks GmbH.

Use ONLY the retrieved context below to answer. Include specific details like job titles, requirements, seniority levels, and salary bands when available.
If the context doesn't contain the answer, say so clearly — do not invent job postings.

CONTEXT:
{context}"""


def job_search_node(state: AgentState) -> dict:
    user_msg = ""
    for m in reversed(state.get("messages", [])):
        if m["role"] == "user":
            user_msg = m.get("content", "")
            break

    if not user_msg:
        return {"messages": [{"role": "assistant", "content": "Could you clarify what job you're looking for?"}]}

    qdrant = try_qdrant_a()
    if qdrant is None:
        return {"messages": [{"role": "assistant", "content": "The job search knowledge base is currently unavailable. Please try again later."}]}

    embed_client, embed_model = get_embed_client_and_model()
    hits = search_jobs(qdrant, embed_client, embed_model, user_msg, limit=5)
    context = format_context(hits)

    chat_client, chat_model = get_chat_client_and_model()
    resp = chat_completion_create(
        chat_client,
        chat_model,
        messages=[
            {"role": "system", "content": SYSTEM.format(context=context)},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    reply = (resp.choices[0].message.content or "").strip()
    return {"messages": [{"role": "assistant", "content": reply}], "retrieved_context": context}

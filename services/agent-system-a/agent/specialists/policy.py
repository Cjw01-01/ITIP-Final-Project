"""
Specialist 2 — Policy Agent (proposal §Specialist 2).

Handles HR policy questions (leave, benefits, conduct, remote work).
Tool: search_policies() → RAG over hr_policies collection.
Strictly grounded — never extrapolates beyond retrieved policy text.
"""

from __future__ import annotations

from agent.state import AgentState
from config import chat_completion_create, get_chat_client_and_model, get_embed_client_and_model, try_qdrant_a
from rag.retrieve import format_context, search_policies

SYSTEM = """You are the HR Policy specialist for the InMind Talent Intelligence Platform.
You answer questions about company HR policies grounded in Lebanese Labour Law themes.

STRICT RULES:
- Use ONLY the retrieved policy documents below. Do NOT extrapolate or infer beyond what is written.
- If the context does not contain the answer, say: "This is not covered in the available policy documents."
- Quote or paraphrase the policy text directly. Never invent policy provisions.
- If a question involves legal advice, recommend consulting HR or legal counsel.

CONTEXT:
{context}"""


def policy_node(state: AgentState) -> dict:
    user_msg = ""
    for m in reversed(state.get("messages", [])):
        if m["role"] == "user":
            user_msg = m.get("content", "")
            break

    if not user_msg:
        return {"messages": [{"role": "assistant", "content": "What HR policy would you like to know about?"}]}

    qdrant = try_qdrant_a()
    if qdrant is None:
        return {"messages": [{"role": "assistant", "content": "The HR policy knowledge base is currently unavailable."}]}

    embed_client, embed_model = get_embed_client_and_model()
    hits = search_policies(qdrant, embed_client, embed_model, user_msg, limit=5)
    context = format_context(hits)

    chat_client, chat_model = get_chat_client_and_model()
    resp = chat_completion_create(
        chat_client,
        chat_model,
        messages=[
            {"role": "system", "content": SYSTEM.format(context=context)},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=1000,
    )
    reply = (resp.choices[0].message.content or "").strip()
    return {"messages": [{"role": "assistant", "content": reply}], "retrieved_context": context}

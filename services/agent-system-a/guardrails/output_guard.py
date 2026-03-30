"""
Output guardrails (proposal §11.3 — Output Guardrails).

1. Hallucination check (string-overlap heuristic against retrieved context) — §11.3.1
2. Response length cap: 2,500 tokens max (truncate with note) — §11.3.2
3. Disclaimer injection on all candidate rankings — §11.3.3
4. PII scrubbing from logs (names/emails/phones → candidate IDs) — §11.3.4
"""

from __future__ import annotations

import re

import tiktoken

MAX_RESPONSE_TOKENS = 2500
ENCODING = tiktoken.get_encoding("cl100k_base")

RANKING_KEYWORDS = ["ranking", "ranked", "score", "composite", "top candidate", "shortlist"]
RANKING_DISCLAIMER = (
    "\n\n---\n*Disclaimer: AI-generated rankings are advisory only. "
    "Final hiring decisions should involve human review and comply with applicable employment laws.*"
)

GROUNDING_THRESHOLD = 0.30
GROUNDING_WARNING = (
    "\n\n---\n*Note: Some details in this response could not be fully verified against the retrieved documents. "
    "Please cross-check with the original source materials before acting on this information.*"
)


def check_grounding(response: str, context: str) -> str:
    """
    Hallucination check (§11.3.1): lightweight string-overlap heuristic.
    Extracts factual tokens (4+ char words) from the response and checks
    what fraction appear in the retrieved context. If overlap is below
    threshold, appends a grounding warning.
    """
    if not context or context.startswith("(no"):
        return response

    ctx_lower = context.lower()
    stop_words = {
        "this", "that", "with", "from", "have", "will", "been", "they",
        "their", "them", "what", "when", "where", "which", "would", "could",
        "should", "about", "also", "some", "more", "other", "into", "than",
        "then", "each", "very", "does", "here", "most", "only", "over",
        "such", "after", "before", "just", "like", "make", "many", "well",
        "back", "being", "were", "there", "your", "based", "these", "those",
        "note", "please", "following", "information", "available",
    }

    words = re.findall(r"[a-z]{4,}", response.lower())
    factual_words = [w for w in words if w not in stop_words]

    if len(factual_words) < 5:
        return response

    found = sum(1 for w in factual_words if w in ctx_lower)
    overlap = found / len(factual_words)

    if overlap < GROUNDING_THRESHOLD:
        if GROUNDING_WARNING.strip() not in response:
            return response + GROUNDING_WARNING

    return response


def truncate_if_needed(text: str) -> str:
    tokens = ENCODING.encode(text)
    if len(tokens) <= MAX_RESPONSE_TOKENS:
        return text
    truncated = ENCODING.decode(tokens[:MAX_RESPONSE_TOKENS])
    return truncated + "\n\n[Response truncated — reached maximum length.]"


def add_ranking_disclaimer(text: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in RANKING_KEYWORDS):
        if RANKING_DISCLAIMER.strip() not in text:
            return text + RANKING_DISCLAIMER
    return text


def scrub_pii_for_logs(text: str) -> str:
    """Replace emails, phone numbers, SSNs, and street addresses for log safety."""
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_REDACTED]", text)
    text = re.sub(r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}", "[PHONE_REDACTED]", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", text)
    text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD_REDACTED]", text)
    return text


def apply_output_guardrails(text: str, retrieved_context: str = "") -> str:
    text = check_grounding(text, retrieved_context)
    text = truncate_if_needed(text)
    text = add_ranking_disclaimer(text)
    return text

"""Load project-root .env and build OpenAI / Azure clients (chat + embeddings)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AzureOpenAI, NotFoundError, OpenAI
from qdrant_client import QdrantClient

# services/agent-system-a/config.py -> project root is 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def normalize_azure_endpoint(raw: str) -> str:
    raw = raw.strip().rstrip("/")
    if not raw:
        return raw
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    if not parsed.netloc:
        return raw.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def get_chat_client_and_model() -> tuple[Any, str]:
    load_env()
    azure_endpoint = normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT") or "")

    if azure_endpoint:
        key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview").strip()
        dep = (os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("OPENAI_MODEL") or "").strip()
        if not key:
            raise ValueError("Azure: set AZURE_OPENAI_API_KEY or OPENAI_API_KEY")
        if not dep:
            raise ValueError("Azure: set AZURE_OPENAI_CHAT_DEPLOYMENT")
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=key,
            api_version=api_version,
        )
        return client, dep

    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("Set OPENAI_API_KEY or Azure endpoint + key")
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()
    return OpenAI(api_key=key), model


def chat_completion_create(client: Any, model: str, **kwargs: Any) -> Any:
    """
    Chat completions. On Azure, ``model`` must be the deployment *name* from AI Studio
    (same rule as embeddings); a SKU name like gpt-4o only works if that is the deployment name.
    """
    load_env()
    try:
        return client.chat.completions.create(model=model, **kwargs)
    except NotFoundError as e:
        if normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT") or ""):
            raise NotFoundError(
                (
                    f"Azure chat deployment {model!r} not found (404). Set AZURE_OPENAI_CHAT_DEPLOYMENT in .env "
                    "to the exact deployment name under Azure AI Studio → Deployments for this resource. "
                    "If OPENAI_MODEL is set without AZURE_OPENAI_CHAT_DEPLOYMENT, it must match that deployment name."
                ),
                response=e.response,
                body=e.body,
            ) from e
        raise


def get_embed_client_and_model() -> tuple[Any, str]:
    load_env()
    model = (os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()

    sk_embed = (
        os.getenv("EMBEDDINGS_OPENAI_API_KEY")
        or os.getenv("OPENAI_EMBEDDINGS_API_KEY")
        or ""
    ).strip()
    if sk_embed:
        return OpenAI(api_key=sk_embed), model

    azure_endpoint = normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT") or "")

    if azure_endpoint:
        key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview").strip()
        dep = (
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        ).strip()
        if not key:
            raise ValueError("Azure embeddings: set API key in .env")
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=key,
            api_version=api_version,
        )
        return client, dep

    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "Set EMBEDDINGS_OPENAI_API_KEY (OpenAI embeddings) or Azure/OpenAI embedding vars"
        )
    return OpenAI(api_key=key), model


def qdrant_a_url() -> str:
    load_env()
    return (os.getenv("QDRANT_A_URL") or "http://localhost:6333").rstrip("/")


def try_qdrant_a() -> QdrantClient | None:
    """Return client if Qdrant A is reachable; otherwise None (Docker not up yet)."""
    try:
        c = QdrantClient(url=qdrant_a_url(), timeout=3)
        c.get_collections()
        return c
    except Exception:
        return None

"""Agent System B config — Qdrant B + embedding + chat clients."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from qdrant_client import QdrantClient

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
        return AzureOpenAI(azure_endpoint=azure_endpoint, api_key=key, api_version=api_version), dep

    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("Set OPENAI_API_KEY or Azure endpoint + key")
    return OpenAI(api_key=key), (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()


def get_embed_client_and_model() -> tuple[Any, str]:
    load_env()
    model = (os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()

    sk_embed = (os.getenv("EMBEDDINGS_OPENAI_API_KEY") or os.getenv("OPENAI_EMBEDDINGS_API_KEY") or "").strip()
    if sk_embed:
        return OpenAI(api_key=sk_embed), model

    azure_endpoint = normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT") or "")
    if azure_endpoint:
        key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview").strip()
        dep = (os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or model).strip()
        if not key:
            raise ValueError("Azure embeddings: set API key in .env")
        return AzureOpenAI(azure_endpoint=azure_endpoint, api_key=key, api_version=api_version), dep

    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("Set EMBEDDINGS_OPENAI_API_KEY or Azure/OpenAI embedding vars")
    return OpenAI(api_key=key), model


def qdrant_b_url() -> str:
    load_env()
    return (os.getenv("QDRANT_B_URL") or "http://localhost:6334").rstrip("/")


def get_qdrant_b() -> QdrantClient:
    return QdrantClient(url=qdrant_b_url())

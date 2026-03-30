"""
Generate synthetic corpora for InMind Talent Intelligence Platform (ITIP).

Targets match **Project Proposal (ITIP)** — synthetic slice of the RAG corpora:
  - Job postings (synthetic): 21  → collection total 28 with 2+5 real later
  - HR policies: 9
  - Placement briefs: 14
  - Candidate profiles (synthetic): 33  → Qdrant-B total ~58 with ~25 Kaggle later
  - BERT extra snippets: 20 × 5 tracks = 100 (part of wider 500-sample BERT plan)

Prerequisites:
  1. Copy .env.example to .env
  2. Set either OpenAI (OPENAI_API_KEY) OR Azure OpenAI (endpoint + key + version + deployment)
  3. pip install -r requirements.txt
  4. python scripts/generate_data.py

Outputs under data/raw/ (JSONL, one JSON object per line).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"

# --- Proposal-aligned defaults (synthetic counts only) ---
N_SYNTHETIC_JOBS = 21
N_HR_POLICIES = 9
N_PLACEMENT_BRIEFS = 14
N_SYNTHETIC_CANDIDATES = 33
N_BERT_EXTRA_PER_TRACK = 20
CANDIDATE_GEN_BATCH_SIZE = 8  # smaller batches → model hits exact counts reliably


def normalize_azure_endpoint(raw: str) -> str:
    """
    Azure Chat Completions expects base URL only, e.g.
    https://YOUR-RESOURCE.openai.azure.com
    Do NOT use /openai/realtime — that is a different API.
    """
    raw = raw.strip().rstrip("/")
    if not raw:
        return raw
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    if not parsed.netloc:
        return raw.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def get_client() -> Any:
    """OpenAI (platform) or Azure OpenAI, depending on .env."""
    load_dotenv(ROOT / ".env")
    azure_endpoint = normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT") or "")

    if azure_endpoint:
        key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview").strip()
        if not key:
            print(
                "ERROR: Azure mode: set AZURE_OPENAI_API_KEY (or OPENAI_API_KEY) in .env.",
                file=sys.stderr,
            )
            sys.exit(1)
        return AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=key,
            api_version=api_version,
        )

    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key or key == "sk-your-key-here":
        print(
            "ERROR: Set OPENAI_API_KEY in .env, OR set AZURE_OPENAI_ENDPOINT for Azure OpenAI.",
            file=sys.stderr,
        )
        sys.exit(1)
    return OpenAI(api_key=key)


def chat_deployment_or_model() -> str:
    """Azure: deployment name. OpenAI: model id (e.g. gpt-4o)."""
    if (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip():
        dep = (os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("OPENAI_MODEL") or "").strip()
        if not dep:
            print(
                "ERROR: Azure mode: set AZURE_OPENAI_CHAT_DEPLOYMENT to your chat deployment name.",
                file=sys.stderr,
            )
            sys.exit(1)
        return dep
    return os.getenv("OPENAI_MODEL", "gpt-4o")


def chat_json(client: Any, system: str, user: str, max_retries: int = 3) -> dict:
    """Call GPT and parse JSON from the assistant message."""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=chat_deployment_or_model(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
            )
            text = (resp.choices[0].message.content or "").strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            return json.loads(text)
        except Exception as e:
            last_err = e
            err_text = str(e).lower()
            if "404" in err_text or "resource not found" in err_text:
                if (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip():
                    raise RuntimeError(
                        "Azure returned 404 (resource not found). Usually this means:\n"
                        "  1) AZURE_OPENAI_CHAT_DEPLOYMENT must be the exact name of a **Chat** deployment "
                        "in Azure OpenAI (Model deployments), NOT a Realtime-only deployment.\n"
                        "  2) Deploy e.g. gpt-4o or gpt-4o-mini as a normal chat model and put that "
                        "deployment name in .env.\n"
                        "  3) Check AZURE_OPENAI_API_VERSION matches what your resource supports "
                        "(try 2024-08-01-preview).\n"
                        f"Original error: {last_err}"
                    ) from e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after retries: {last_err}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- Generators ---


def gen_job_postings(client: Any, n: int) -> list[dict]:
    system = (
        "You output ONLY valid JSON. No markdown. "
        "Each job must be realistic for a tech academy / AI company (inmind.ai style) "
        "or BMW Group / idealworks Munich internship-style roles."
    )
    user = f"""Generate exactly {n} distinct job postings as a JSON object with key "jobs" whose value is an array.
Each job object must have:
  "id": string unique id like job_syn_001
  "title": string
  "company": one of "inmind.ai", "BMW Group", "idealworks GmbH"
  "location": string
  "employment_type": string
  "seniority": string
  "salary_band": string or null
  "requirements": array of strings (must-haves)
  "nice_to_have": array of strings
  "description": string (2-4 short paragraphs)
  "track": for BMW/idealworks only, one of "AI","Backend","Frontend","Robotics","Simulation" or null for pure inmind roles

Mix: some inmind internal (curriculum, DevOps, ML engineer), some BMW/idealworks internship tracks.
"""
    data = chat_json(client, system, user)
    return data["jobs"]


def gen_hr_policies(client: Any, n: int) -> list[dict]:
    system = (
        "You output ONLY valid JSON. Policies must be plausible HR documents grounded in "
        "Lebanese Labour Law themes (leave, working hours, termination, conduct, remote work, benefits). "
        "Do not invent specific article numbers unless clearly marked as illustrative."
    )
    user = f"""Generate exactly {n} HR policy documents as JSON: {{ "policies": [ ... ] }}
Each policy:
  "id": string e.g. hr_pol_001
  "title": string
  "category": one of leave, conduct, remote_work, benefits, working_hours, termination, general
  "body": string, full policy text, 400-900 words, professional tone
"""
    data = chat_json(client, system, user)
    return data["policies"]


def gen_placement_briefs(client: Any, n: int) -> list[dict]:
    system = (
        "You output ONLY valid JSON. Briefs describe BMW Group / idealworks academy placement: "
        "stack, expectations, evaluation, logistics for Munich internships."
    )
    user = f"""Generate exactly {n} placement briefs as JSON: {{ "briefs": [ ... ] }}
Each brief:
  "id": string e.g. place_001
  "title": string
  "track": one of AI, Backend, Frontend, Robotics, Simulation
  "doc_type": one of overview, technical_stack, expectations, logistics, assessment
  "body": string, 350-800 words
"""
    data = chat_json(client, system, user)
    return data["briefs"]


def _gen_candidate_profiles_batch(client: Any, batch_n: int, id_start: int) -> list[dict]:
    """Single API call for batch_n profiles; ids cand_syn_{id_start:03d} ..."""
    id_end = id_start + batch_n - 1
    system = (
        "You output ONLY valid JSON. Synthetic candidate profiles for tech recruitment. "
        "Vary skills, experience, education, languages; no real person names (use plausible fictitious names). "
        f"The array MUST contain exactly {batch_n} objects, no fewer."
    )
    user = f"""Generate exactly {batch_n} candidate profiles as JSON: {{ "candidates": [ ... ] }}

Each candidate object:
  "id": use these exact ids in order: cand_syn_{id_start:03d} through cand_syn_{id_end:03d}
  "name": fictitious full name
  "email": fictitious email
  "summary": string, 2-4 sentences
  "skills": array of strings (15-40 skills mix hard/soft)
  "experience_years": number
  "education": string
  "languages": array of strings
  "bmw_track_label": one of AI, Backend, Frontend, Robotics, Simulation (ground truth for BERT training)
  "raw_resume_snippet": string, 200-500 words like a resume excerpt

Return exactly {batch_n} candidates. Balance roughly across bmw_track_label values.
"""
    data = chat_json(client, system, user)
    return list(data.get("candidates") or [])


def gen_candidate_profiles(client: Any, n: int, batch_size: int = CANDIDATE_GEN_BATCH_SIZE) -> list[dict]:
    """
    Build n profiles using multiple smaller requests. One big JSON often truncates (e.g. 6 instead of 33).
    """
    all_rows: list[dict] = []
    seen_ids: set[str] = set()
    next_index = 1
    max_rounds = max(30, (n // max(batch_size, 1)) * 3)
    rounds = 0

    while len(all_rows) < n and rounds < max_rounds:
        rounds += 1
        need = n - len(all_rows)
        bn = min(batch_size, need)
        batch = _gen_candidate_profiles_batch(client, bn, next_index)
        if not batch:
            continue
        for row in batch:
            if len(all_rows) >= n:
                break
            rid = str(row.get("id", "")).strip()
            if not rid or rid in seen_ids:
                rid = f"cand_syn_{len(all_rows) + 1:03d}"
                row["id"] = rid
            seen_ids.add(rid)
            all_rows.append(row)
        next_index = len(all_rows) + 1

    if len(all_rows) < n:
        raise RuntimeError(
            f"Proposal target is {n} synthetic candidates; only got {len(all_rows)} after {rounds} rounds. "
            "Try again or lower --candidate-batch-size / check API limits."
        )
    return all_rows[:n]


def gen_bert_extra_samples(client: Any, per_track: int) -> list[dict]:
    tracks = ["AI", "Backend", "Frontend", "Robotics", "Simulation"]
    rows: list[dict] = []
    system = (
        "You output ONLY valid JSON. Short resume-style text snippets for multi-class classification."
    )
    for track in tracks:
        user = f"""Generate exactly {per_track} objects in JSON: {{ "samples": [ ... ] }}
Each sample:
  "text": string, 80-220 words, resume excerpt clearly indicative of {track} track
  "label": "{track}"
  "id": unique string e.g. bert_{track.lower()}_001 (increment numbers)
Vary writing style and seniority. No real names; use placeholders like Candidate A."""
        data = chat_json(client, system, user)
        rows.extend(data["samples"])
    random.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ITIP datasets via OpenAI.")
    parser.add_argument("--jobs", type=int, default=N_SYNTHETIC_JOBS, help="Synthetic job postings (proposal: 21)")
    parser.add_argument("--policies", type=int, default=N_HR_POLICIES, help="HR policy docs (proposal: 9)")
    parser.add_argument("--briefs", type=int, default=N_PLACEMENT_BRIEFS, help="Placement briefs (proposal: 14)")
    parser.add_argument(
        "--candidates",
        type=int,
        default=N_SYNTHETIC_CANDIDATES,
        help="Synthetic candidate profiles (proposal: 33 for Qdrant-B with ~25 Kaggle)",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=CANDIDATE_GEN_BATCH_SIZE,
        help="Profiles per API call (smaller = more reliable full count)",
    )
    parser.add_argument(
        "--bert-per-track",
        type=int,
        default=N_BERT_EXTRA_PER_TRACK,
        help="Extra BERT snippets per track (proposal layer: 20×5=100)",
    )
    parser.add_argument("--skip-bert-extra", action="store_true", help="Skip extra BERT snippets")
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        help="Only regenerate candidate_profiles_syn.jsonl (saves API calls)",
    )
    args = parser.parse_args()

    client = get_client()
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if args.candidates_only:
        steps = [
            (
                "candidate_profiles_syn.jsonl",
                lambda: gen_candidate_profiles(
                    client, args.candidates, batch_size=args.candidate_batch_size
                ),
            ),
        ]
    else:
        steps = [
            ("job_postings_syn.jsonl", lambda: gen_job_postings(client, args.jobs)),
            ("hr_policies_syn.jsonl", lambda: gen_hr_policies(client, args.policies)),
            ("placement_briefs_syn.jsonl", lambda: gen_placement_briefs(client, args.briefs)),
            (
                "candidate_profiles_syn.jsonl",
                lambda: gen_candidate_profiles(
                    client, args.candidates, batch_size=args.candidate_batch_size
                ),
            ),
        ]
        if not args.skip_bert_extra:
            steps.append(
                (
                    "bert_train_extra.jsonl",
                    lambda: gen_bert_extra_samples(client, args.bert_per_track),
                )
            )

    for filename, fn in tqdm(steps, desc="datasets"):
        path = DATA_RAW / filename
        tqdm.write(f"Generating {filename} ...")
        rows = fn()
        write_jsonl(path, rows)
        tqdm.write(f"  Wrote {len(rows)} rows -> {path}")

    print("\nDone. Next: add your real job PDFs/text under data/raw/ and run ingest (when ready).")


if __name__ == "__main__":
    main()

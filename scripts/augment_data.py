"""
Augment ITIP synthetic data to meet rubric requirements:

  - Candidate profiles: 33 → 58  (need 25 more)
  - Job postings:       21 → 28  (need 7 more)
  - HR policies:        expand bodies to ~700-900 words each (9 × ~800 words ≈ 24 pages)

Reads existing JSONL, generates only the missing entries, writes updated files.

Usage:
  python scripts/augment_data.py
  python scripts/augment_data.py --skip-candidates   # only expand policies + jobs
  python scripts/augment_data.py --skip-policies      # only add candidates + jobs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"


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


def get_client_and_model() -> tuple:
    load_dotenv(ROOT / ".env")
    azure_endpoint = normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT") or "")

    if azure_endpoint:
        key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview").strip()
        dep = (os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("OPENAI_MODEL") or "").strip()
        if not key or not dep:
            print("ERROR: Azure config incomplete. Check .env", file=sys.stderr)
            sys.exit(1)
        return AzureOpenAI(azure_endpoint=azure_endpoint, api_key=key, api_version=api_version), dep

    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        print("ERROR: Set OPENAI_API_KEY or Azure vars in .env", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=key), os.getenv("OPENAI_MODEL", "gpt-4o")


def chat_json(client, model: str, system: str, user: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            return json.loads(text)
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed after retries")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 1. Generate additional candidate profiles
# ---------------------------------------------------------------------------

def gen_extra_candidates(client, model: str, existing: list[dict], target: int = 58) -> list[dict]:
    need = target - len(existing)
    if need <= 0:
        print(f"  Candidates: already at {len(existing)}, no augmentation needed.")
        return existing

    print(f"  Candidates: {len(existing)} -> {target} (generating {need} more)")
    existing_names = {c["name"] for c in existing}
    existing_tracks = [c.get("bmw_track_label", "AI") for c in existing]

    all_new: list[dict] = []
    batch_size = 8
    next_id = len(existing) + 1

    while len(all_new) < need:
        bn = min(batch_size, need - len(all_new))
        id_start = next_id + len(all_new)
        id_end = id_start + bn - 1

        system = (
            "You output ONLY valid JSON. Synthetic candidate profiles for tech recruitment at inmind.ai / BMW Group. "
            "Vary skills, experience, education, universities (include LAU, AUB, NDU, USJ, BAU for Lebanese candidates). "
            "Use plausible fictitious names. Mix nationalities. "
            f"The array MUST contain exactly {bn} objects."
        )
        user = f"""Generate exactly {bn} candidate profiles as JSON: {{ "candidates": [ ... ] }}

Each candidate:
  "id": use ids cand_syn_{id_start:03d} through cand_syn_{id_end:03d}
  "name": fictitious full name (DO NOT reuse: {', '.join(list(existing_names)[:10])}...)
  "email": fictitious email
  "summary": 2-4 sentences about their background
  "skills": array of 15-30 technical and soft skills
  "experience_years": number (0-10, vary it)
  "education": degree and university
  "languages": array of languages
  "bmw_track_label": one of AI, Backend, Frontend, Robotics, Simulation (balance across tracks)
  "raw_resume_snippet": 200-400 words like a real resume excerpt with projects, achievements, tools used

Return exactly {bn} candidates."""

        try:
            data = chat_json(client, model, system, user)
            batch = data.get("candidates", [])
            for row in batch:
                if len(all_new) >= need:
                    break
                rid = row.get("id", f"cand_syn_{id_start + len(all_new):03d}")
                row["id"] = rid
                existing_names.add(row.get("name", ""))
                all_new.append(row)
            print(f"    Got {len(batch)} profiles, total new: {len(all_new)}/{need}")
        except Exception as e:
            print(f"    Batch failed: {e}")

        time.sleep(1)

    return existing + all_new[:need]


# ---------------------------------------------------------------------------
# 2. Generate additional job postings
# ---------------------------------------------------------------------------

def gen_extra_jobs(client, model: str, existing: list[dict], target: int = 28) -> list[dict]:
    need = target - len(existing)
    if need <= 0:
        print(f"  Jobs: already at {len(existing)}, no augmentation needed.")
        return existing

    print(f"  Jobs: {len(existing)} -> {target} (generating {need} more)")
    existing_titles = [j["title"] for j in existing]
    next_id = len(existing) + 1

    system = (
        "You output ONLY valid JSON. Job postings for inmind.ai (tech academy / AI company), "
        "BMW Group, or idealworks GmbH. Make them realistic and varied."
    )
    user = f"""Generate exactly {need} NEW job postings as JSON: {{ "jobs": [ ... ] }}

DO NOT duplicate these existing titles: {json.dumps(existing_titles)}

Each job:
  "id": use ids job_syn_{next_id:03d} through job_syn_{next_id + need - 1:03d}
  "title": unique role title
  "company": one of "inmind.ai", "BMW Group", "idealworks GmbH"
  "location": city, country
  "employment_type": Full-time, Part-time, or Internship
  "seniority": Junior, Mid-level, Senior, Internship, or Manager
  "salary_band": string like "60,000 - 80,000 EUR/year" or null for internships
  "requirements": array of 3-5 must-have requirements
  "nice_to_have": array of 2-4 nice-to-have skills
  "description": 2-4 paragraphs (150-250 words)
  "track": for BMW/idealworks internships use AI/Backend/Frontend/Robotics/Simulation, null for inmind roles

Mix: some inmind (e.g. QA engineer, technical writer, academy ops), some BMW/idealworks tracks."""

    data = chat_json(client, model, system, user)
    new_jobs = data.get("jobs", [])
    print(f"    Got {len(new_jobs)} new job postings")
    return existing + new_jobs


# ---------------------------------------------------------------------------
# 3. Expand HR policy bodies to 700-900 words each
# ---------------------------------------------------------------------------

def expand_hr_policies(client, model: str, existing: list[dict]) -> list[dict]:
    print(f"  HR Policies: expanding {len(existing)} policies to 700-900 words each (~20+ pages total)")
    expanded = []

    for pol in tqdm(existing, desc="    Expanding policies"):
        current_words = len(pol["body"].split())
        if current_words >= 650:
            expanded.append(pol)
            continue

        system = (
            "You output ONLY valid JSON. You are an HR policy writer for a Lebanese tech company (inmind.ai). "
            "Policies must be grounded in Lebanese Labour Law themes. Write in formal, professional language. "
            "Include specific procedures, eligibility criteria, exceptions, and employee/employer responsibilities."
        )
        user = f"""Expand this HR policy to 700-900 words. Keep the same title and category.
Add more detail: specific procedures, eligibility criteria, timelines, exceptions, responsibilities, and compliance notes.

Current policy:
Title: {pol['title']}
Category: {pol['category']}
Body ({current_words} words): {pol['body']}

Return JSON: {{ "id": "{pol['id']}", "title": "{pol['title']}", "category": "{pol['category']}", "body": "expanded text here 700-900 words" }}"""

        try:
            data = chat_json(client, model, system, user)
            data["id"] = pol["id"]
            data["title"] = pol["title"]
            data["category"] = pol["category"]
            new_words = len(data.get("body", "").split())
            print(f"      {pol['id']} ({pol['title']}): {current_words} -> {new_words} words")
            expanded.append(data)
        except Exception as e:
            print(f"      Failed to expand {pol['id']}: {e}, keeping original")
            expanded.append(pol)

        time.sleep(0.5)

    total_words = sum(len(p["body"].split()) for p in expanded)
    est_pages = total_words / 300
    print(f"    Total: {total_words} words ~ {est_pages:.1f} pages")
    return expanded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Augment ITIP data to meet rubric requirements")
    parser.add_argument("--skip-candidates", action="store_true")
    parser.add_argument("--skip-jobs", action="store_true")
    parser.add_argument("--skip-policies", action="store_true")
    parser.add_argument("--target-candidates", type=int, default=58)
    parser.add_argument("--target-jobs", type=int, default=28)
    args = parser.parse_args()

    client, model = get_client_and_model()
    print(f"Using model: {model}")
    print(f"Data dir: {DATA_RAW}\n")

    # --- Candidates ---
    if not args.skip_candidates:
        cands = read_jsonl(DATA_RAW / "candidate_profiles_syn.jsonl")
        cands = gen_extra_candidates(client, model, cands, target=args.target_candidates)
        write_jsonl(DATA_RAW / "candidate_profiles_syn.jsonl", cands)
        print(f"  OK - Wrote {len(cands)} candidates\n")

    # --- Job postings ---
    if not args.skip_jobs:
        jobs = read_jsonl(DATA_RAW / "job_postings_syn.jsonl")
        jobs = gen_extra_jobs(client, model, jobs, target=args.target_jobs)
        write_jsonl(DATA_RAW / "job_postings_syn.jsonl", jobs)
        print(f"  OK - Wrote {len(jobs)} job postings\n")

    # --- HR policies ---
    if not args.skip_policies:
        policies = read_jsonl(DATA_RAW / "hr_policies_syn.jsonl")
        policies = expand_hr_policies(client, model, policies)
        write_jsonl(DATA_RAW / "hr_policies_syn.jsonl", policies)
        print(f"  OK - Wrote {len(policies)} expanded policies\n")

    print("=" * 50)
    print("AUGMENTATION COMPLETE")
    print("=" * 50)

    cands = read_jsonl(DATA_RAW / "candidate_profiles_syn.jsonl")
    jobs = read_jsonl(DATA_RAW / "job_postings_syn.jsonl")
    pols = read_jsonl(DATA_RAW / "hr_policies_syn.jsonl")
    briefs = read_jsonl(DATA_RAW / "placement_briefs_syn.jsonl")

    total_pol_words = sum(len(p["body"].split()) for p in pols)
    print(f"  Candidates:  {len(cands)} (target: 50-60)")
    print(f"  Job postings: {len(jobs)} (target: 25-30)")
    print(f"  HR policies:  {len(pols)} ({total_pol_words} words ~ {total_pol_words//300} pages, target: 20+)")
    print(f"  Placement:    {len(briefs)}")
    print("\nNext: python scripts/generate_pdfs.py")


if __name__ == "__main__":
    main()

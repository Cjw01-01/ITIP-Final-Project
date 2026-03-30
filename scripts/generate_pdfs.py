"""
Generate professional PDFs from augmented JSONL data.

Output directories (matching ingest.py --pdf expectations):
  data/pdfs/cvs/             - 1-page candidate CVs
  data/pdfs/policies/        - HR policies (multi-page per policy)
  data/pdfs/job_listings/    - 1-page job postings
  data/pdfs/placement_briefs/- placement briefs

Usage:
  python scripts/generate_pdfs.py
  python scripts/generate_pdfs.py --only cvs
  python scripts/generate_pdfs.py --only policies
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PDF = ROOT / "data" / "pdfs"


def _write_metadata(path: Path, meta: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sanitize(text) -> str:
    if text is None:
        return ""
    if isinstance(text, (dict, list)):
        text = json.dumps(text)
    text = str(text)
    if not text:
        return ""
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
        .replace("\u2022", "-")
        .replace("\u00b7", "-")
        .replace("\u00e9", "e")
        .replace("\u00e8", "e")
        .replace("\u00e0", "a")
        .replace("\u00fc", "u")
        .replace("\u00f6", "o")
        .replace("\u00e4", "a")
        .replace("\u00df", "ss")
        .replace("\u00f1", "n")
        .replace("\ufffd", "'")
    )
    import re
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def safe_text(pdf: FPDF, w: float, h: float, text: str, **kwargs) -> None:
    clean = sanitize(text)
    pdf.set_x(pdf.l_margin)
    try:
        pdf.multi_cell(w, h, clean, **kwargs)
    except Exception:
        ascii_text = clean.encode("ascii", errors="replace").decode("ascii")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w, h, ascii_text, **kwargs)


# ---------------------------------------------------------------------------
# 1. Candidate CVs - 1-page, well-structured
# ---------------------------------------------------------------------------

class CvPdf(FPDF):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 5, f"CV - {sanitize(self._name)}", align="C")


def generate_cv(cand: dict, out_dir: Path) -> Path:
    pdf = CvPdf(cand.get("name", "Candidate"))
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    name = sanitize(cand.get("name", "Unknown"))
    email = sanitize(cand.get("email", ""))
    track = sanitize(cand.get("bmw_track_label", ""))

    # --- Name header ---
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 9, name, new_x="LMARGIN", new_y="NEXT")

    # --- Contact & track bar ---
    pdf.set_draw_color(30, 60, 120)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    contact_parts = [email]
    langs = cand.get("languages", [])
    if isinstance(langs, list) and langs:
        contact_parts.append("Languages: " + ", ".join(str(l) for l in langs))
    if track:
        contact_parts.append(f"Track: {track}")
    exp = cand.get("experience_years", "")
    if exp:
        contact_parts.append(f"Experience: {exp} years")
    pdf.cell(0, 5, sanitize(" | ".join(contact_parts)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- Professional Summary ---
    summary = cand.get("summary", "")
    if summary:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 60, 120)
        pdf.cell(0, 5, "PROFESSIONAL SUMMARY", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(40, 40, 40)
        safe_text(pdf, 0, 4, summary)
        pdf.ln(2)

    # --- Education ---
    edu = cand.get("education", "")
    if edu:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 60, 120)
        pdf.cell(0, 5, "EDUCATION", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(40, 40, 40)
        safe_text(pdf, 0, 4, edu)
        pdf.ln(2)

    # --- Skills (compact comma-separated) ---
    skills = cand.get("skills", [])
    if skills:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 60, 120)
        pdf.cell(0, 5, "SKILLS", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(40, 40, 40)
        skills_str = [sanitize(str(s)) for s in skills]
        safe_text(pdf, 0, 3.5, " | ".join(skills_str))
        pdf.ln(2)

    # --- Experience / Resume Excerpt ---
    snippet = cand.get("raw_resume_snippet", "")
    if isinstance(snippet, dict):
        snippet = json.dumps(snippet)
    elif isinstance(snippet, list):
        snippet = " ".join(str(s) for s in snippet)
    snippet = str(snippet or "")
    if snippet:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 60, 120)
        pdf.cell(0, 5, "EXPERIENCE", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(40, 40, 40)
        safe_text(pdf, 0, 3.5, snippet)

    out_path = out_dir / f"{cand['id']}.pdf"
    pdf.output(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# 2. HR Policies - multi-page documents
# ---------------------------------------------------------------------------

class PolicyPdf(FPDF):
    def __init__(self):
        super().__init__()
        self._title = "HR Policy"

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, sanitize(f"inmind.ai - {self._title}"), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")


def generate_policy_pdf(pol: dict, out_dir: Path) -> Path:
    pdf = PolicyPdf()
    pdf._title = sanitize(pol.get("title", "HR Policy"))
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Company header
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 7, "inmind.ai", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 4, "Human Resources Department", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Divider
    pdf.set_draw_color(30, 60, 120)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Policy title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    safe_text(pdf, 0, 8, sanitize(pol.get("title", "Policy")))
    pdf.ln(2)

    # Category badge
    category = pol.get("category", "general")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, sanitize(f"Category: {category.replace('_', ' ').title()} | Effective: January 2025 | Version 1.0"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Body text - split into paragraphs
    body = pol.get("body", "")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)

    paragraphs = body.split("\n\n") if "\n\n" in body else body.split(". ")
    if len(paragraphs) <= 2:
        sentences = body.split(". ")
        paragraphs = []
        chunk_size = max(3, len(sentences) // 6)
        for i in range(0, len(sentences), chunk_size):
            para = ". ".join(sentences[i : i + chunk_size])
            if para and not para.endswith("."):
                para += "."
            paragraphs.append(para)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) < 60 and not para.endswith("."):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 60, 120)
            safe_text(pdf, 0, 5, para)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
        else:
            safe_text(pdf, 0, 5, para)
            pdf.ln(2)

    out_path = out_dir / f"{pol['id']}.pdf"
    pdf.output(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# 3. Job Postings - 1-page professional layout
# ---------------------------------------------------------------------------

class JobPdf(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 5, "inmind.ai | BMW Group | idealworks GmbH - Talent Intelligence Platform", align="C")


def generate_job_pdf(job: dict, out_dir: Path) -> Path:
    pdf = JobPdf()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    company = sanitize(job.get("company", "inmind.ai"))
    title = sanitize(job.get("title", "Open Position"))

    # Company header
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 7, company, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 60, 120)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Job title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Meta info
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    location = sanitize(job.get("location", ""))
    etype = sanitize(job.get("employment_type", ""))
    seniority = sanitize(job.get("seniority", ""))
    salary = sanitize(job.get("salary_band") or "Competitive")
    track = sanitize(job.get("track") or "")

    meta = f"Location: {location} | Type: {etype} | Level: {seniority} | Salary: {salary}"
    if track:
        meta += f" | Track: {track}"
    pdf.cell(0, 5, meta, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Description
    desc = job.get("description", "")
    if desc:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 60, 120)
        pdf.cell(0, 6, "About the Role", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        safe_text(pdf, 0, 4.5, desc)
        pdf.ln(4)

    # Requirements
    reqs = job.get("requirements", [])
    if reqs:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 60, 120)
        pdf.cell(0, 6, "Requirements", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        req_text = "\n".join(f"- {sanitize(str(r))}" for r in reqs)
        safe_text(pdf, 0, 4.5, req_text)
        pdf.ln(3)

    # Nice to have
    nice = job.get("nice_to_have", [])
    if nice:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 60, 120)
        pdf.cell(0, 6, "Nice to Have", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        nice_text = "\n".join(f"- {sanitize(str(n))}" for n in nice)
        safe_text(pdf, 0, 4.5, nice_text)

    out_path = out_dir / f"{job['id']}.pdf"
    pdf.output(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# 4. Placement Briefs
# ---------------------------------------------------------------------------

class BriefPdf(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 5, "BMW Group & idealworks GmbH - Internship Programme", align="C")


def generate_brief_pdf(brief: dict, out_dir: Path) -> Path:
    pdf = BriefPdf()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    title = sanitize(brief.get("title", "Placement Brief"))
    track = sanitize(brief.get("track", ""))
    doc_type = sanitize(brief.get("doc_type", "brief"))

    # Header
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 7, "BMW Group - Internship Programme", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 60, 120)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Title
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 30, 30)
    safe_text(pdf, 0, 7, title)
    pdf.ln(2)

    # Meta
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, f"Track: {track} | Document Type: {doc_type.replace('_', ' ').title()}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Body
    body = brief.get("body", "")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    safe_text(pdf, 0, 5, body)

    out_path = out_dir / f"{brief['id']}.pdf"
    pdf.output(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PDFs from augmented JSONL")
    parser.add_argument("--only", choices=["cvs", "policies", "jobs", "briefs"],
                        help="Generate only one category")
    args = parser.parse_args()

    cv_dir = DATA_PDF / "cvs"
    pol_dir = DATA_PDF / "policies"
    job_dir = DATA_PDF / "job_listings"
    brief_dir = DATA_PDF / "placement_briefs"

    for d in [cv_dir, pol_dir, job_dir, brief_dir]:
        d.mkdir(parents=True, exist_ok=True)

    do_all = args.only is None
    total = 0

    # --- CVs ---
    if do_all or args.only == "cvs":
        cands = read_jsonl(DATA_RAW / "candidate_profiles_syn.jsonl")
        print(f"Generating {len(cands)} candidate CVs (1 page each)...")
        cv_meta = {}
        for cand in cands:
            try:
                p = generate_cv(cand, cv_dir)
                cv_meta[p.name] = {
                    "source_id": cand["id"],
                    "candidate_name": cand.get("name", "Unknown"),
                    "bmw_track_label": cand.get("bmw_track_label"),
                    "skills": cand.get("skills", []),
                    "experience_years": cand.get("experience_years"),
                    "education": cand.get("education"),
                }
            except Exception as e:
                print(f"  WARN: {cand.get('id')} failed: {e}")
        _write_metadata(cv_dir / "metadata.json", cv_meta)
        total += len(cands)
        print(f"  -> {len(cands)} CVs in {cv_dir}")

    # --- HR Policies ---
    if do_all or args.only == "policies":
        policies = read_jsonl(DATA_RAW / "hr_policies_syn.jsonl")
        print(f"Generating {len(policies)} HR policy PDFs...")
        pol_meta = {}
        total_pages = 0
        for pol in policies:
            try:
                p = generate_policy_pdf(pol, pol_dir)
                pol_meta[p.name] = {
                    "source_id": pol["id"],
                    "title": pol.get("title"),
                    "category": pol.get("category"),
                }
                try:
                    import pymupdf
                    doc = pymupdf.open(str(p))
                    pages = len(doc)
                    doc.close()
                    total_pages += pages
                    print(f"  {pol['id']} ({sanitize(pol['title'])}): {pages} page(s)")
                except ImportError:
                    total_pages += 3
            except Exception as e:
                print(f"  WARN: {pol.get('id')} failed: {e}")
        _write_metadata(pol_dir / "metadata.json", pol_meta)
        total += len(policies)
        print(f"  -> {len(policies)} policy PDFs ({total_pages} pages total) in {pol_dir}")

    # --- Job Postings ---
    if do_all or args.only == "jobs":
        jobs = read_jsonl(DATA_RAW / "job_postings_syn.jsonl")
        print(f"Generating {len(jobs)} job posting PDFs...")
        job_meta = {}
        for job in jobs:
            try:
                p = generate_job_pdf(job, job_dir)
                job_meta[p.name] = {
                    "source_id": job["id"],
                    "title": job.get("title"),
                    "company": job.get("company"),
                    "track": job.get("track"),
                    "employment_type": job.get("employment_type"),
                    "seniority": job.get("seniority"),
                }
            except Exception as e:
                print(f"  WARN: {job.get('id')} failed: {e}")
        _write_metadata(job_dir / "metadata.json", job_meta)
        total += len(jobs)
        print(f"  -> {len(jobs)} job PDFs in {job_dir}")

    # --- Placement Briefs ---
    if do_all or args.only == "briefs":
        briefs = read_jsonl(DATA_RAW / "placement_briefs_syn.jsonl")
        print(f"Generating {len(briefs)} placement brief PDFs...")
        brief_meta = {}
        for brief in briefs:
            try:
                p = generate_brief_pdf(brief, brief_dir)
                brief_meta[p.name] = {
                    "source_id": brief["id"],
                    "title": brief.get("title"),
                    "track": brief.get("track"),
                    "doc_type": brief.get("doc_type"),
                }
            except Exception as e:
                print(f"  WARN: {brief.get('id')} failed: {e}")
        _write_metadata(brief_dir / "metadata.json", brief_meta)
        total += len(briefs)
        print(f"  -> {len(briefs)} brief PDFs in {brief_dir}")

    print(f"\n{'='*50}")
    print(f"TOTAL: {total} PDFs generated (with metadata.json sidecars)")
    print(f"{'='*50}")
    print(f"\nAll PDFs ready in: {DATA_PDF}")
    print("Next step: python scripts/ingest.py --pdf --reset")


if __name__ == "__main__":
    main()

# InMind Talent Intelligence Platform (ITIP)

A production-ready, containerised, multi-agent AI system for talent acquisition and BMW placement intelligence at inmind.ai.

**Author:** Carl Wakim · Gen AI Track · Spring 2026

> Full technical report (architecture, design decisions, evaluation results, etc.) is submitted separately on Overleaf.

---

## Quick Start

### Prerequisites

- **Docker Desktop** installed and running ([download](https://www.docker.com/products/docker-desktop/))
- **Python 3.11+** installed ([download](https://www.python.org/downloads/))
- **Git** installed (for cloning)
- An **OpenAI API key** or **Azure OpenAI** endpoint + key

### Step 1 — Clone the repository

```bash
git clone https://github.com/Cjw01-01/ITIP-Final-Project.git
cd ITIP-Final-Project
```

> If Git LFS is not installed, run `git lfs install` first (needed for the 267 MB DistilBERT model weights).

### Step 2 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` in any text editor and fill in **your** credentials:

- **Option A (OpenAI platform):** Set `OPENAI_API_KEY=sk-...`
- **Option B (Azure OpenAI):** Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_CHAT_DEPLOYMENT`, and `AZURE_OPENAI_API_VERSION`
- **Embeddings:** Set `EMBEDDINGS_OPENAI_API_KEY=sk-...` (can be the same OpenAI key)

Leave the `QDRANT_*`, `REDIS_URL`, `AGENT_B_URL`, and `MCP_URL` values as-is — they default to `localhost` and Docker Compose overrides them internally.

### Step 3 — Start all 6 Docker containers

Make sure **Docker Desktop is running**, then from the repo root:

```bash
docker compose up --build -d
```

This builds and starts:

| Container | Port | What it does |
|-----------|------|-------------|
| `itip-qdrant-a` | 6333 | Vector DB for jobs, policies, placements |
| `itip-qdrant-b` | 6334 | Vector DB for candidate profiles |
| `itip-redis` | 6379 | Session store (24h TTL) |
| `itip-mcp-server` | 8002 | Interview scheduling tools |
| `itip-agent-system-b` | 8001 | Skills Matcher (Google ADK) |
| `itip-agent-system-a` | 8000 | Supervisor + 4 specialists (LangGraph) |

Wait until all containers show **healthy**:

```bash
docker ps
```

All 6 should show `(healthy)` in the STATUS column. First build takes ~10 minutes due to PyTorch download.

### Step 4 — Ingest data into Qdrant

Synthetic JSONL data is included in `data/raw/`. Install dependencies and run ingestion:

```bash
pip install -r requirements.txt
python scripts/ingest.py
```

You should see output like:

```
Loaded jobs=28 policies=12 briefs=14 candidates=58
  job_postings: 112 points
  hr_policies: 36 points
  placement_briefs: 14 points
  candidate_profiles (Qdrant B): 60 points
```

### Step 5 — Launch the UI

```bash
pip install -r ui/requirements.txt
streamlit run ui/app.py --server.port 8501
```

### Step 6 — Open and test

Open **http://localhost:8501** in your browser and sign in with any of these demo accounts:

| Username | Password | Role | Access |
|----------|----------|------|--------|
| `jobseeker` | `pass123` | Job Seeker | Job search |
| `hr` | `pass123` | HR / Recruiter | Candidate screening, job search |
| `staff` | `pass123` | InMind Staff | HR policies |
| `instructor` | `pass123` | Academy Instructor | BMW placement, candidate screening |

### Stopping everything

```bash
docker compose down       # stop containers (keeps data in volumes)
docker compose down -v    # stop containers AND wipe all Qdrant/Redis data
```

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (:8501)                               │
│                     Login → Role-based Chat Interface                      │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │ HTTP
┌─────────────────────────────────▼──────────────────────────────────────────┐
│                     Agent System A (:8000)                                  │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌────────────┐  ┌─────────┐ │
│  │Supervisor│→ │Job Search │  │  Policy    │  │ Screener   │  │BMW Place│ │
│  │(GPT-4o)  │  │    Agent  │  │   Agent   │  │   Agent    │  │  Agent  │ │
│  └──────────┘  └─────┬─────┘  └─────┬─────┘  └─────┬──────┘  └────┬────┘ │
│       │              │              │              │               │       │
│  ┌────▼────┐    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐    ┌────▼────┐  │
│  │DistilBERT│   │Qdrant A │   │Qdrant A │   │Agent B  │    │Qdrant A │  │
│  │Classifier│   │job_posts│   │hr_policy│   │via HTTP │    │placemnt │  │
│  └──────────┘   └─────────┘   └─────────┘   └────┬────┘    └─────────┘  │
│                                                    │                      │
│  ┌──────────┐                              ┌──────▼──────┐               │
│  │ Guardrails│                              │  MCP Server │               │
│  │(in+out)  │                              │  (scheduling)│               │
│  └──────────┘                              └─────────────┘               │
└──────────────────────────────────────────────────────────────────────────┘
                                  │ HTTP
┌─────────────────────────────────▼──────────────────────────────────────────┐
│            Agent System B (:8001) — Google ADK + LiteLLM                   │
│            Skills Matcher: embedding similarity + skill-weighted scoring   │
│            Own Qdrant instance (Qdrant B :6334, candidate_profiles)        │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
InMind Project/
├── docker-compose.yml          # 6-container orchestration
├── .env.example                # Environment template
├── requirements.txt            # Root deps (ingest, evaluate)
├── README.md                   # This file
├── EVALUATION.md               # Evaluation results
├── data/
│   ├── raw/                    # Synthetic JSONL datasets
│   └── pdfs/                   # PDF ingestion (cvs, policies, ...)
├── evaluation/
│   ├── test_set.json           # 29-question test set
│   └── results/                # Evaluation outputs
├── logs/                       # Structured JSON logs (itip.jsonl)
├── models/
│   └── track_classifier/       # Fine-tuned DistilBERT weights (LFS)
├── scripts/
│   ├── generate_data.py        # GPT-4o synthetic data generation
│   ├── ingest.py               # Chunking + embedding → Qdrant
│   ├── evaluate.py             # RAGAS + retrieval + routing eval
│   └── train_classifier.py     # DistilBERT fine-tuning
├── services/
│   ├── agent-system-a/         # LangGraph supervisor + 4 specialists
│   ├── agent-system-b/         # Google ADK Skills Matcher
│   └── mcp-server/             # FastMCP scheduling tools
└── ui/
    ├── app.py                  # Streamlit UI (login + role chat)
    └── .streamlit/config.toml  # Theme config
```

---

## Changes Tracking

| Date | Change | Rationale |
|------|--------|-----------|
| Mar 19 | Initial architecture: 6-container stack, LangGraph + Google ADK | Proposal submission |
| Mar 20 | RAG pipeline: structure-aware chunking, text-embedding-3-small | Section-specific chunks improve precision for job postings |
| Mar 21 | DistilBERT track classifier: 5-fold CV, 97% accuracy | Fast CPU-based first-pass routing for candidate screening |
| Mar 22 | MCP server: 3 scheduling tools (find, schedule, assess) | Decoupled interview scheduling from agent logic |
| Mar 23 | Guardrails: regex + GPT-4o-mini two-layer input/output | Blocks injection, discrimination, PII extraction |
| Mar 24 | Session persistence: Redis with 24h TTL replacing in-memory | Survives container restarts |
| Mar 25 | Voice interface: Whisper STT → pipeline → OpenAI TTS | End-to-end voice demo |
| Mar 25 | Evaluation pipeline: 29 questions, RAGAS, config comparisons | Automated retrieval + generation metrics |
| Mar 26 | Agent B `/chat` endpoint, improved error handling on Agent A | API layer completeness — both systems expose POST /chat |
| Mar 27 | PDF ingestion pipeline: pdfplumber + PyMuPDF + per-corpus chunking | Production data path alongside synthetic JSONL |

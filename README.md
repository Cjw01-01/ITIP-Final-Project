# InMind Talent Intelligence Platform (ITIP)

A production-ready, containerised, multi-agent AI system for talent acquisition and BMW placement intelligence at inmind.ai.

**Author:** Carl Wakim · Gen AI Track · Spring 2026

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

Wait until all containers show **healthy** (Agent A is last — it depends on all others):

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

> Optional: To regenerate synthetic data from scratch, run `python scripts/generate_data.py` first (requires API key).

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

## Architecture

### 6-Container Docker Stack

| Container | Service | Port | Purpose |
|-----------|---------|------|---------|
| `agent-system-a` | Agent A (LangGraph) | 8000 | Supervisor + 4 specialists, DistilBERT, voice |
| `agent-system-b` | Agent B (Google ADK) | 8001 | Skills Matcher (embedding + skill scoring) |
| `mcp-server` | MCP (FastMCP) | 8002 | Interview scheduling tools |
| `qdrant-a` | Vector DB | 6333 | job_postings, hr_policies, placement_briefs |
| `qdrant-b` | Vector DB | 6334 | candidate_profiles |
| `redis` | Session Store | 6379 | Conversation history (24h TTL) |

### Agent System A — LangGraph (§5)

- **Supervisor**: GPT-4o intent classifier → routes to one of 4 specialists
- **Job Search Agent**: RAG over `job_postings` collection
- **Policy Agent**: RAG over `hr_policies` with strict grounding
- **Candidate Screener**: DistilBERT track classification + Agent B HTTP call + MCP tools
- **BMW Placement Agent**: RAG over `placement_briefs` with track metadata filtering
- **Role-based routing**: Supervisor respects `allowed_specialists` per user role
- **Session persistence**: Redis with 24h TTL (replaces in-memory storage)

### Agent System B — Google ADK (§6)

- Built on Google ADK with LiteLLM adapter → GPT-4o
- Scoring formula: `S = 0.4 × semantic × 100 + 0.6 × skill_intersection`
- Completely independent: own container, own Qdrant, own requirements
- Communication with Agent A is **HTTP only** (no Python imports)

### MCP Server (§8)

Three tools exposed via FastMCP protocol + REST:
- `find_available_interviewers(role, date, track?)`
- `schedule_interview(candidate_id, interviewer_id, datetime, type)`
- `send_assessment(candidate_id, assessment_type)`

## Key Technical Decisions

### Chunking Strategy (§7.3)

| Corpus | Strategy | Chunk Size | Overlap | Rationale |
|--------|----------|-----------|---------|-----------|
| Job Postings | Structure-aware sections | 350 tok | 50 tok | Section-specific queries need section-specific chunks |
| HR Policies | Recursive character split | 500 tok | 100 tok | Policy prose has natural paragraph boundaries |
| Placement Briefs | Recursive + metadata-tiered | 450 tok | 90 tok | Mixed doc types; metadata enables pre-filtering |
| Candidate Profiles | Fixed-size + skill tags | 400 tok | 80 tok | Semi-structured, dense vocabulary |

### Embedding Model

`text-embedding-3-small` (1536 dims, 8191-token context, $0.02/1M tokens) — same API as GPT-4o, multilingual, negligible cost at project scale.

### DistilBERT Track Classifier (§5.5)

Fine-tuned `distilbert-base-uncased` on 5 BMW tracks (AI, Backend, Frontend, Robotics, Simulation). 5-fold cross-validation, <50ms inference. Used by Candidate Screener for first-pass track routing.

### Guardrails (§11)

**Input**: Length cap (2000 chars), regex injection detection, GPT-4o-mini injection check, discriminatory query blocking, PII extraction prevention, English-only check.

**Output**: Grounding verification (context overlap), 2500-token cap, ranking disclaimer, PII scrub for logs.

### Voice Interface (§5.7)

`POST /chat/voice` — Whisper STT → LangGraph pipeline → OpenAI TTS (nova). Returns MP3 audio.

## Data Ingestion

Two modes:

```bash
# Synthetic JSONL (default — for demo)
python scripts/ingest.py

# PDF mode (production — reads from data/pdfs/)
python scripts/ingest.py --pdf

# Reset collections first
python scripts/ingest.py --reset
```

### PDF Pipeline

Place PDFs in the appropriate directory:
- `data/pdfs/cvs/` — Candidate resumes
- `data/pdfs/policies/` — HR policy documents
- `data/pdfs/job_listings/` — Job descriptions
- `data/pdfs/placement_briefs/` — BMW placement material

Features: text extraction (pdfplumber), table detection → Markdown, image extraction (PyMuPDF), per-corpus chunking strategies.

## Evaluation

```bash
python scripts/evaluate.py
```

See `EVALUATION.md` for full results including:
- Retrieval metrics (P@K, R@K, MRR per collection)
- RAGAS scores (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
- Configuration comparisons (chunk size, top-K)
- DistilBERT classifier evaluation (accuracy, F1, confusion matrix)
- Routing accuracy per category

## Environment Variables

```env
# OpenAI or Azure (see .env.example for full options)
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o

# Qdrant
QDRANT_A_URL=http://localhost:6333
QDRANT_B_URL=http://localhost:6334

# Embeddings
EMBEDDINGS_OPENAI_API_KEY=sk-...

# Internal (set by docker-compose)
AGENT_B_URL=http://agent-system-b:8001
MCP_URL=http://mcp-server:8002
REDIS_URL=redis://redis:6379/0
```

## Repository Structure

```
InMind Project/
├── docker-compose.yml          # 6-container orchestration
├── .env.example                # Environment template
├── requirements.txt            # Root deps (ingest, evaluate)
├── README.md                   # This file
├── EVALUATION.md               # Evaluation report
├── Proposal/                   # Project proposal PDF + figures
├── data/
│   ├── raw/                    # Synthetic JSONL datasets
│   └── pdfs/                   # PDF ingestion (cvs, policies, ...)
├── evaluation/
│   ├── test_set.json           # 29-question test set
│   └── results/                # Evaluation outputs
├── logs/                       # Structured JSON logs (itip.jsonl)
├── models/
│   └── track_classifier/       # Fine-tuned DistilBERT weights
├── scripts/
│   ├── generate_data.py        # GPT-4o synthetic data generation
│   ├── ingest.py               # Chunking + embedding → Qdrant
│   ├── evaluate.py             # RAGAS + retrieval + routing eval
│   └── train_classifier.py     # DistilBERT fine-tuning
├── services/
│   ├── agent-system-a/         # LangGraph supervisor + 4 specialists
│   │   ├── agent/              # Supervisor, specialists, state
│   │   ├── classifier/         # DistilBERT inference wrapper
│   │   ├── guardrails/         # Input + output guardrails
│   │   └── rag/                # Retrieval pipeline
│   ├── agent-system-b/         # Google ADK Skills Matcher
│   └── mcp-server/             # FastMCP scheduling tools
└── ui/
    ├── app.py                  # Streamlit UI (login + role chat)
    └── .streamlit/config.toml  # Theme config
```

## Known Limitations

- **Synthetic data**: Most corpora are GPT-4o generated. Grounded in real sources (inmind.ai careers, BMW/idealworks specs, Lebanese Labour Law) but not production data.
- **English only**: Non-English queries are blocked by input guardrail.
- **Mock scheduling**: MCP server uses in-memory mock data, not a real HRIS API.
- **No real authentication**: Login uses hardcoded demo credentials.
- **CPU inference**: DistilBERT runs on CPU inside Docker (~50ms per call).

## Changes Tracking

| Date | Change | Rationale |
|------|--------|-----------|
| Mar 19 | Initial architecture: 6-container stack, LangGraph + Google ADK | Proposal submission |
| Mar 20 | RAG pipeline: structure-aware chunking, text-embedding-3-small | §7.3 — section-specific chunks improve precision for job postings |
| Mar 21 | DistilBERT track classifier: 5-fold CV, 97% accuracy | §5.5 — fast CPU-based first-pass routing for candidate screening |
| Mar 22 | MCP server: 3 scheduling tools (find, schedule, assess) | §8 — decoupled interview scheduling from agent logic |
| Mar 23 | Guardrails: regex + GPT-4o-mini two-layer input/output | §11 — blocks injection, discrimination, PII extraction |
| Mar 24 | Session persistence: Redis with 24h TTL replacing in-memory | §9.3 — survives container restarts |
| Mar 25 | Voice interface: Whisper STT → pipeline → OpenAI TTS | §5.7 — end-to-end voice demo |
| Mar 25 | Evaluation pipeline: 29 questions, RAGAS, config comparisons | §10 — automated retrieval + generation metrics |
| Mar 26 | Agent B `/chat` endpoint, improved error handling on Agent A | API layer completeness — both systems expose POST /chat |
| Mar 27 | PDF ingestion pipeline: pdfplumber + PyMuPDF + per-corpus chunking | §7.4 — production data path alongside synthetic JSONL |

## Proposal

See `Proposal/PorjectProposalCarlWakim.pdf` for the complete 33-page system proposal.

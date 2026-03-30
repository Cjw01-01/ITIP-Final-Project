"""
InMind Talent Intelligence Platform — Premium Web UI

Features:
  - Login page with 4 role-based accounts
  - Role-specific specialist access, quick prompts, and branding
  - Glass-morphism dark UI with aurora gradients
  - Native Streamlit chat with custom Material avatars
  - Voice: sidebar recorder + audio upload → /chat/voice (Whisper + TTS)
  - Agent B health status in sidebar

Run:
  cd ui && pip install -r requirements.txt && streamlit run app.py
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

AGENT_A_URL = os.getenv("AGENT_A_URL", "http://localhost:8000").rstrip("/")
AGENT_B_URL = os.getenv("AGENT_B_URL", "http://localhost:8001").rstrip("/")

# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

ROLES = {
    "job_seeker": {
        "label": "Job Seeker",
        "icon": "search",
        "emoji": "\U0001f50d",
        "accent": "#00d4aa",
        "accent_rgb": "0,212,170",
        "gradient": "linear-gradient(135deg, #00d4aa, #0d9488)",
        "description": "Search open positions, explore role requirements, and find your next opportunity.",
        "quick_prompts": [
            ("ML roles at inmind.ai", "What open machine learning positions are at inmind.ai?"),
            ("DevOps positions", "What DevOps positions are available?"),
            ("Remote roles", "Are there any remote engineering roles open?"),
            ("Role requirements", "What are the requirements for the Senior AI Engineer role?"),
        ],
        "username": "jobseeker",
        "password": "pass123",
    },
    "hr": {
        "label": "HR / Recruiter",
        "icon": "group",
        "emoji": "\U0001f465",
        "accent": "#f472b6",
        "accent_rgb": "244,114,182",
        "gradient": "linear-gradient(135deg, #f472b6, #db2777)",
        "description": "Screen candidates, rank applicants, and manage the hiring pipeline.",
        "quick_prompts": [
            ("Top ML candidates", "Who are the top candidates for the Machine Learning Engineer role?"),
            ("Screen for DevOps", "Screen candidates for the DevOps Engineer position."),
            ("Compare candidates", "Compare the top 3 candidates for the Backend Developer role."),
            ("Open positions", "What positions are currently open?"),
        ],
        "username": "hr",
        "password": "pass123",
    },
    "staff": {
        "label": "InMind Staff",
        "icon": "badge",
        "emoji": "\U0001f4cb",
        "accent": "#a78bfa",
        "accent_rgb": "167,139,250",
        "gradient": "linear-gradient(135deg, #a78bfa, #7c3aed)",
        "description": "Access HR policies, benefits, leave rules, and company guidelines.",
        "quick_prompts": [
            ("Annual leave", "How many annual leave days do I get after one year?"),
            ("Remote work policy", "Can I work remotely full-time?"),
            ("Probation period", "What is the probation period for new hires?"),
            ("Sick leave", "How does the sick leave policy work?"),
        ],
        "username": "staff",
        "password": "pass123",
    },
    "instructor": {
        "label": "Academy Instructor",
        "icon": "school",
        "emoji": "\U0001f393",
        "accent": "#38bdf8",
        "accent_rgb": "56,189,248",
        "gradient": "linear-gradient(135deg, #38bdf8, #0284c7)",
        "description": "View BMW placement tracks, candidate dashboards, and track assignments.",
        "quick_prompts": [
            ("BMW AI track", "What technical skills are required for the BMW AI internship track?"),
            ("Robotics placement", "What does the idealworks Robotics track require?"),
            ("Track candidates", "Who are the top candidates for the BMW AI track placement?"),
            ("Placement GPA", "What is the minimum GPA for BMW placement consideration?"),
        ],
        "username": "instructor",
        "password": "pass123",
    },
}

SPECIALIST_STYLES = {
    "job_search": ("Job Search", "#00d4aa", "work"),
    "policy": ("HR Policy", "#a78bfa", "policy"),
    "candidate_screener": ("Screener", "#f472b6", "person_search"),
    "bmw_placement": ("BMW Placement", "#38bdf8", "school"),
    "guardrail_block": ("Guardrail", "#fb7185", "shield"),
    "FINISH": ("Complete", "#94a3b8", "check_circle"),
    "none": ("\u2014", "#64748b", "circle"),
}

CHAT_AVATAR_USER = ":material/person:"
CHAT_AVATAR_ASSISTANT = ":material/auto_awesome:"


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def inject_css(role_key: str | None = None) -> None:
    role = ROLES.get(role_key) if role_key else None
    accent = role["accent"] if role else "#00d4aa"
    accent_rgb = role["accent_rgb"] if role else "0,212,170"
    role_gradient = role["gradient"] if role else "linear-gradient(135deg, #00d4aa, #0d9488)"

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* --- Reset Streamlit chrome --- */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{background: transparent;}}
div[data-testid="stToolbar"] {{visibility: hidden;}}

/* Force sidebar visible */
section[data-testid="stSidebar"] {{
  width: 760px !important;
  min-width: 760px !important;
  transform: translateX(0px) !important;
  transition: none !important;
}}
section[data-testid="stSidebar"] > div {{
  width: 760px !important;
}}

.block-container {{
  padding-top: 1.2rem !important;
  padding-bottom: 3rem !important;
  max-width: 1100px !important;
}}

/* --- Page canvas: aurora mesh --- */
.stApp {{
  background: #050508 !important;
  background-image:
    radial-gradient(ellipse 120% 80% at 10% -10%, rgba({accent_rgb}, 0.18), transparent 50%),
    radial-gradient(ellipse 90% 60% at 90% 10%, rgba(167, 139, 250, 0.12), transparent 45%),
    radial-gradient(ellipse 70% 50% at 50% 100%, rgba(56, 189, 248, 0.08), transparent 50%),
    linear-gradient(180deg, #050508 0%, #0a0a12 40%, #050508 100%) !important;
}}

/* Subtle grid overlay */
.stApp::before {{
  content: "";
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
  z-index: 0;
  mask-image: radial-gradient(ellipse 80% 70% at 50% 30%, black 20%, transparent 75%);
}}

/* Material Symbols */
.material-symbols-rounded {{
  font-family: 'Material Symbols Rounded';
  font-weight: normal;
  font-style: normal;
  font-size: 20px;
  display: inline-block;
  line-height: 1;
  text-transform: none;
  letter-spacing: normal;
  word-wrap: normal;
  white-space: nowrap;
  direction: ltr;
  -webkit-font-smoothing: antialiased;
}}

/* --- Hero / header --- */
.itip-hero {{
  font-family: 'Outfit', sans-serif;
  text-align: center;
  padding: 0.4rem 0 1.2rem;
  position: relative;
  z-index: 1;
}}
.itip-badge {{
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.8rem;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  background: linear-gradient(135deg, rgba({accent_rgb}, 0.18), rgba(167,139,250,0.12));
  border: 1px solid rgba(255,255,255,0.08);
  color: {accent};
  margin-bottom: 0.6rem;
  animation: pulse-glow 4s ease-in-out infinite;
}}
@keyframes pulse-glow {{
  0%, 100% {{ box-shadow: 0 0 20px rgba({accent_rgb}, 0.15); }}
  50% {{ box-shadow: 0 0 32px rgba({accent_rgb}, 0.25); }}
}}
.itip-title {{
  font-size: clamp(1.8rem, 4.5vw, 2.5rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.1;
  margin: 0;
  background: linear-gradient(120deg, #f8fafc 0%, {accent} 45%, #c4b5fd 80%, #7dd3fc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.itip-sub {{
  margin-top: 0.5rem;
  font-size: 0.95rem;
  font-weight: 400;
  color: rgba(226,232,240,0.5);
  max-width: 500px;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.45;
}}

/* --- Chat shell --- */
.itip-chat-shell {{
  position: relative;
  z-index: 1;
  border-radius: 24px;
  padding: 1px;
  background: linear-gradient(135deg, rgba({accent_rgb}, 0.35), rgba(167,139,250,0.2), rgba(56,189,248,0.15));
  margin-bottom: 0.8rem;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.06) inset, 0 24px 80px -20px rgba(0,0,0,0.65);
}}
.itip-chat-inner {{
  border-radius: 23px;
  background: rgba(10, 10, 18, 0.72);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  padding: 1rem 1rem 0.5rem;
  min-height: 400px;
  max-height: min(56vh, 600px);
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: rgba({accent_rgb}, 0.35) rgba(255,255,255,0.04);
}}
.itip-chat-inner::-webkit-scrollbar {{ width: 7px; }}
.itip-chat-inner::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.03); border-radius: 8px; }}
.itip-chat-inner::-webkit-scrollbar-thumb {{
  background: linear-gradient(180deg, rgba({accent_rgb}, 0.45), rgba(99,102,241,0.3));
  border-radius: 8px;
}}

/* --- Empty state --- */
.itip-empty {{
  text-align: center;
  padding: 2.5rem 1rem 2rem;
  color: rgba(148,163,184,0.85);
  font-family: 'Outfit', sans-serif;
}}
.itip-empty-icon {{
  font-size: 2.2rem;
  margin-bottom: 0.6rem;
  opacity: 0.8;
}}

/* --- Native Streamlit chat messages --- */
[data-testid="stChatMessage"] {{
  width: fit-content !important;
  max-width: min(50rem, 100%) !important;
  background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: 18px !important;
  padding: 0.45rem 0.6rem !important;
  margin-bottom: 0.65rem !important;
  box-shadow: 0 0 0 1px rgba({accent_rgb},0.05) inset, 0 16px 40px -20px rgba(0,0,0,0.5) !important;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  transition: border-color 0.25s ease;
}}
[data-testid="stChatMessage"]:hover {{
  border-color: rgba(255,255,255,0.14) !important;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label^="Chat message from user"]) {{
  margin-left: auto !important;
  margin-right: 0 !important;
  border-color: rgba({accent_rgb}, 0.22) !important;
  background: linear-gradient(145deg, rgba({accent_rgb}, 0.1) 0%, rgba(255,255,255,0.02) 100%) !important;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label^="Chat message from assistant"]) {{
  margin-left: 0 !important;
  margin-right: auto !important;
  border-color: rgba(99,102,241,0.2) !important;
  background: linear-gradient(145deg, rgba(99,102,241,0.07) 0%, rgba(167,139,250,0.04) 45%, rgba(255,255,255,0.02) 100%) !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {{
  font-family: 'Outfit', sans-serif !important;
  color: #e8e8ed !important;
  line-height: 1.6 !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {{
  color: #f1f5f9 !important;
  font-weight: 600 !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.85em;
  background: rgba({accent_rgb}, 0.12);
  padding: 0.12rem 0.35rem;
  border-radius: 5px;
  color: #a5f3fc;
}}
/* Avatar overrides */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessage"]:has([aria-label^="Chat message from user"]) [data-testid="stChatMessageAvatarCustom"] {{
  background: {role_gradient} !important;
  color: #042f2e !important;
  box-shadow: 0 6px 20px rgba({accent_rgb}, 0.3) !important;
  border: none !important;
}}
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessage"]:has([aria-label^="Chat message from assistant"]) [data-testid="stChatMessageAvatarCustom"] {{
  background: linear-gradient(145deg, #6366f1, #8b5cf6) !important;
  color: #f8fafc !important;
  box-shadow: 0 6px 20px rgba(99,102,241,0.3) !important;
  border: none !important;
}}

/* Specialist badge */
.spec-badge {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: rgba(148,163,184,0.9);
  letter-spacing: 0.02em;
  margin-top: 0.3rem;
  padding: 0.25rem 0.55rem;
  border-radius: 8px;
  background: rgba(0,0,0,0.22);
  border: 1px solid rgba(255,255,255,0.05);
}}
.spec-badge .material-symbols-rounded {{ font-size: 14px; }}

/* --- Input area --- */
.itip-input-wrap {{
  position: relative;
  z-index: 1;
  margin-top: 0.4rem;
}}
textarea {{
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.95rem !important;
}}

/* Primary button */
.stButton > button[kind="primary"],
button[data-testid="stFormSubmitButton"] {{
  background: {role_gradient} !important;
  border: none !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
  border-radius: 12px !important;
  padding: 0.5rem 1.2rem !important;
  box-shadow: 0 8px 28px rgba({accent_rgb}, 0.3) !important;
  transition: transform 0.2s, box-shadow 0.2s !important;
}}
.stButton > button[kind="primary"]:hover {{
  transform: translateY(-1px);
  box-shadow: 0 12px 36px rgba({accent_rgb}, 0.4) !important;
}}

/* Secondary / quick prompt buttons */
.stButton > button[kind="secondary"] {{
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  color: #cbd5e1 !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.8rem !important;
  border-radius: 999px !important;
  transition: border-color 0.2s, background 0.2s !important;
}}
.stButton > button[kind="secondary"]:hover {{
  background: rgba({accent_rgb}, 0.08) !important;
  border-color: rgba({accent_rgb}, 0.3) !important;
}}

/* --- Sidebar --- */
[data-testid="stSidebar"],
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0c0c14 0%, #08080e 100%) !important;
  border-right: 1px solid rgba(255,255,255,0.06) !important;
  width: 760px !important;
  min-width: 760px !important;
  transform: translateX(0px) !important;
  transition: none !important;
  display: flex !important;
  visibility: visible !important;
  z-index: 999 !important;
}}
[data-testid="stSidebar"] .block-container {{
  padding-top: 1.5rem !important;
}}
[data-testid="stSidebar"] h3 {{
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: rgba(148,163,184,0.9) !important;
  margin-bottom: 0.5rem !important;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  padding-bottom: 0.35rem;
}}
[data-testid="stSidebar"] hr {{
  border: none !important;
  border-top: 1px solid rgba(255,255,255,0.06) !important;
  margin: 0.8rem 0 !important;
}}
[data-testid="stSidebar"] [data-testid="stCode"] {{
  font-size: 0.62rem !important;
  border-radius: 8px !important;
  border: 1px solid rgba({accent_rgb}, 0.2) !important;
  background: rgba(0,0,0,0.3) !important;
}}

/* Status dot */
.status-live {{
  width: 8px; height: 8px; border-radius: 50%;
  background: #22c55e;
  box-shadow: 0 0 12px #22c55e;
  animation: blink 2s ease-in-out infinite;
  display: inline-block;
}}
@keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}
.status-down {{
  width: 8px; height: 8px; border-radius: 50%;
  background: #ef4444;
  box-shadow: 0 0 12px #ef4444;
  display: inline-block;
}}

/* --- Login page --- */
.login-container {{
  max-width: 860px;
  margin: 0 auto;
  padding: 2rem 1rem;
  font-family: 'Outfit', sans-serif;
}}
.login-title {{
  text-align: center;
  font-size: clamp(2rem, 5vw, 3rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(120deg, #f8fafc 0%, #a8f5e5 35%, #c4b5fd 70%, #7dd3fc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.3rem;
}}
.login-subtitle {{
  text-align: center;
  color: rgba(148,163,184,0.7);
  font-size: 1rem;
  margin-bottom: 2.5rem;
}}
.role-card {{
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px;
  padding: 1.5rem 1.2rem;
  text-align: center;
  transition: all 0.3s ease;
  cursor: pointer;
  min-height: 200px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(8px);
}}
.role-card:hover {{
  border-color: rgba(255,255,255,0.18);
  transform: translateY(-4px);
  box-shadow: 0 20px 60px -20px rgba(0,0,0,0.6);
}}
.role-icon {{
  font-size: 2.5rem;
  margin-bottom: 0.8rem;
}}
.role-name {{
  font-size: 1.1rem;
  font-weight: 700;
  color: #f1f5f9;
  margin-bottom: 0.4rem;
}}
.role-desc {{
  font-size: 0.78rem;
  color: rgba(148,163,184,0.8);
  line-height: 1.45;
}}

/* Form inputs on login */
[data-testid="stTextInput"] input {{
  font-family: 'Outfit', sans-serif !important;
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 10px !important;
  color: #e2e8f0 !important;
}}
[data-testid="stTextInput"] input:focus {{
  border-color: rgba({accent_rgb}, 0.5) !important;
  box-shadow: 0 0 0 2px rgba({accent_rgb}, 0.15) !important;
}}

/* Expander styling */
[data-testid="stExpander"] {{
  border: 1px solid rgba(255,255,255,0.06) !important;
  border-radius: 12px !important;
  background: rgba(255,255,255,0.02) !important;
}}
[data-testid="stExpander"] summary {{
  font-family: 'Outfit', sans-serif !important;
  color: rgba(148,163,184,0.9) !important;
  font-size: 0.85rem !important;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_agent(message: str, session_id: str | None, role: str) -> dict:
    body: dict = {"message": message, "role": role}
    if session_id:
        body["session_id"] = session_id
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{AGENT_A_URL}/chat", json=body)
        r.raise_for_status()
        return r.json()


def call_agent_stream(message: str, session_id: str | None, role: str, status_placeholder):
    """SSE streaming call to /chat/stream with live UI updates."""
    body: dict = {"message": message, "role": role}
    if session_id:
        body["session_id"] = session_id

    result: dict = {
        "reply": "",
        "session_id": session_id or "",
        "route_taken": "",
        "specialist_used": "",
        "iterations": 0,
        "guardrail_blocked": False,
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", f"{AGENT_A_URL}/chat/stream", json=body) as resp:
                resp.raise_for_status()
                buffer = ""
                for chunk in resp.iter_text():
                    buffer += chunk
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        for line in event_str.strip().split("\n"):
                            if line.startswith("data: "):
                                try:
                                    evt = json.loads(line[6:])
                                except json.JSONDecodeError:
                                    continue

                                evt_type = evt.get("type", "")

                                if evt_type == "start":
                                    result["session_id"] = evt.get("session_id", result["session_id"])
                                    status_placeholder.caption("Starting pipeline...")

                                elif evt_type == "node_update":
                                    node = evt.get("node", "")
                                    route = evt.get("route", "")
                                    if node == "supervisor" and route:
                                        result["route_taken"] = route
                                        result["iterations"] = evt.get("iteration", 0)
                                        result["specialist_used"] = route
                                        status_placeholder.caption(f"Routing to **{route}**...")
                                    elif node != "supervisor":
                                        preview = evt.get("content_preview", "")
                                        label = node.replace("_", " ").title()
                                        if preview:
                                            status_placeholder.caption(f"{label}: generating response...")
                                        else:
                                            status_placeholder.caption(f"Running {label}...")

                                elif evt_type == "guardrail_block":
                                    result["reply"] = evt.get("content", "Blocked by guardrail.")
                                    result["guardrail_blocked"] = True
                                    status_placeholder.empty()

                                elif evt_type == "done":
                                    result["reply"] = evt.get("reply", "")
                                    result["session_id"] = evt.get("session_id", result["session_id"])
                                    status_placeholder.empty()

    except Exception as e:
        result["reply"] = f"**Connection error:** {e}\n\nIs Agent A running at `{AGENT_A_URL}`?"
        result["specialist_used"] = "none"
        result["route_taken"] = "error"
        status_placeholder.empty()

    return result


def fetch_health(url: str) -> dict | None:
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{url}/health")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


def _voice_header_utf8(headers: httpx.Headers, b64_name: str, plain_name: str) -> str:
    """Decode X-*-B64 (UTF-8) or fall back to plain Latin-1 header for older Agent A."""
    b64 = headers.get(b64_name.lower())
    if b64:
        try:
            return base64.standard_b64decode(b64).decode("utf-8")
        except Exception:
            pass
    return (headers.get(plain_name.lower()) or "").strip()


def call_agent_voice(
    audio_bytes: bytes,
    filename: str,
    mime: str,
    session_id: str | None,
    role: str,
) -> dict:
    """POST /chat/voice — returns reply_mp3 bytes + transcript + headers metadata."""
    files = {"audio": (filename, audio_bytes, mime or "audio/wav")}
    data: dict[str, str] = {"role": role}
    if session_id:
        data["session_id"] = session_id
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{AGENT_A_URL}/chat/voice", files=files, data=data)
    h = r.headers
    out: dict = {
        "ok": r.status_code == 200,
        "status_code": r.status_code,
        "detail": r.text[:2000] if r.status_code != 200 else "",
        "transcript": _voice_header_utf8(h, "x-transcript-b64", "x-transcript"),
        "reply_text": _voice_header_utf8(h, "x-reply-b64", "x-reply-text"),
        "session_id": (h.get("x-session-id") or "").strip(),
        "mp3": r.content if r.status_code == 200 else b"",
    }
    return out


# ---------------------------------------------------------------------------
# Helper components
# ---------------------------------------------------------------------------

def specialist_badge_html(spec_key: str) -> str:
    key = spec_key or "none"
    if key not in SPECIALIST_STYLES:
        key = "none"
    name, color, icon = SPECIALIST_STYLES[key]
    return (
        f'<span class="spec-badge" style="color:{color};">'
        f'<span class="material-symbols-rounded" style="color:{color};">{icon}</span>'
        f'{name}</span>'
    )


def render_chat_messages(messages: list[dict]) -> None:
    for m in messages:
        role = "user" if m["role"] == "user" else "assistant"
        av = CHAT_AVATAR_USER if role == "user" else CHAT_AVATAR_ASSISTANT
        with st.chat_message(role, avatar=av):
            st.markdown(m["content"])
            if role == "assistant":
                meta = m.get("meta") or {}
                spec = meta.get("specialist_used") or "none"
                route = meta.get("route_taken", "\u2014")
                iters = meta.get("iterations", "\u2014")
                guard = meta.get("guardrail_blocked")
                gtxt = "blocked" if guard else "cleared"
                badge = specialist_badge_html(spec)
                st.markdown(
                    f'{badge} &nbsp; '
                    f'<span class="spec-badge">route: {route}</span> &nbsp; '
                    f'<span class="spec-badge">steps: {iters}</span> &nbsp; '
                    f'<span class="spec-badge">guardrail: {gtxt}</span>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

def render_login() -> None:
    inject_css()

    st.markdown(
        """
<div class="login-container">
  <div class="login-title">InMind Talent Intelligence</div>
  <div class="login-subtitle">Sign in to access your workspace</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4, gap="medium")
    for i, (role_key, role) in enumerate(ROLES.items()):
        with cols[i]:
            st.markdown(
                f"""
<div class="role-card" style="border-color: rgba({role['accent_rgb']}, 0.15);">
  <div class="role-icon"><span class="material-symbols-rounded" style="font-size:2.5rem;color:{role['accent']};">{role['icon']}</span></div>
  <div class="role-name" style="color: {role['accent']};">{role['label']}</div>
  <div class="role-desc">{role['description']}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("login_form"):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            username = st.text_input("Username", placeholder="e.g. jobseeker, hr, staff, instructor")
        with c2:
            password = st.text_input("Password", type="password", placeholder="pass123")
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            login_btn = st.form_submit_button("Sign In", type="primary", use_container_width=True)

    if login_btn:
        matched_role = None
        for rk, rv in ROLES.items():
            if rv["username"] == username.strip().lower() and rv["password"] == password:
                matched_role = rk
                break
        if matched_role:
            st.session_state.logged_in = True
            st.session_state.role = matched_role
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()
        else:
            st.error("Invalid credentials. Try: jobseeker/pass123, hr/pass123, staff/pass123, or instructor/pass123")

    st.markdown(
        """
<div style="text-align:center;margin-top:2rem;font-family:'Outfit',sans-serif;font-size:0.75rem;color:rgba(148,163,184,0.5);">
  InMind Talent Intelligence Platform &middot; Multi-Agent &middot; RAG &middot; Guardrails &middot; DistilBERT
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chat page (per role)
# ---------------------------------------------------------------------------

def render_chat() -> None:
    role_key = st.session_state.get("role", "job_seeker")
    role = ROLES[role_key]

    inject_css(role_key)

    import streamlit.components.v1 as components
    components.html(
        """<script>
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.setAttribute('aria-expanded', 'true');
            sidebar.style.setProperty('width', '760px', 'important');
            sidebar.style.setProperty('min-width', '760px', 'important');
            sidebar.style.setProperty('transform', 'translateX(0px)', 'important');
            sidebar.style.setProperty('display', 'flex', 'important');
            sidebar.style.setProperty('visibility', 'visible', 'important');
        }
        const collapsed = window.parent.document.querySelector('[data-testid="stSidebarCollapsedControl"]');
        if (collapsed) { collapsed.style.display = 'none'; }
        </script>""",
        height=0,
    )

    # --- Sidebar ---
    st.sidebar.markdown(
        f"""
<div style="text-align:center;padding:0.5rem 0 0.8rem;">
  <span class="material-symbols-rounded" style="font-size:2.2rem;color:{role['accent']};">{role['icon']}</span>
  <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:1rem;color:{role['accent']};margin-top:0.3rem;">{role['label']}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # System status
    st.sidebar.markdown("### System Status")
    h_a = fetch_health(AGENT_A_URL)
    h_b = fetch_health(AGENT_B_URL)
    ok_a = h_a is not None and h_a.get("status") == "ok"
    ok_b = h_b is not None and h_b.get("status") == "ok"

    dot_a = "status-live" if ok_a else "status-down"
    dot_b = "status-live" if ok_b else "status-down"
    st.sidebar.markdown(
        f'<div style="display:flex;flex-direction:column;gap:6px;margin-bottom:8px;">'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span class="{dot_a}"></span>'
        f'<span style="font-family:Outfit,sans-serif;font-weight:600;color:#e2e8f0;font-size:0.8rem;">'
        f'Agent A {"Connected" if ok_a else "Offline"}</span></div>'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span class="{dot_b}"></span>'
        f'<span style="font-family:Outfit,sans-serif;font-weight:600;color:#e2e8f0;font-size:0.8rem;">'
        f'Agent B {"Connected" if ok_b else "Offline"}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if h_a:
        qdrant_status = "\u2713" if h_a.get("qdrant_a_reachable") else "\u2717"
        redis_status = "\u2713" if h_a.get("redis_reachable") else "\u2717"
        st.sidebar.caption(f"Qdrant: {qdrant_status} | Redis: {redis_status}")
    if h_b:
        qdrant_b = "\u2713" if h_b.get("qdrant_b_reachable") else "\u2717"
        pts = h_b.get("candidate_profiles_points", 0)
        st.sidebar.caption(f"Qdrant B: {qdrant_b} | Candidates: {pts}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Upload PDF")
    upload_cat = st.sidebar.selectbox(
        "Category",
        options=["cvs", "policies", "job_listings", "placement_briefs"],
        format_func=lambda x: {
            "cvs": "Candidate CV",
            "policies": "HR Policy",
            "job_listings": "Job Posting",
            "placement_briefs": "Placement Brief",
        }[x],
        label_visibility="collapsed",
    )

    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    upload_meta = {}
    if upload_cat == "cvs" and uploaded_file:
        with st.sidebar.expander("CV Metadata (optional)"):
            cv_name = st.text_input("Candidate Name", key="upload_cv_name")
            cv_track = st.selectbox("BMW Track", ["AI", "Backend", "Frontend", "Robotics", "Simulation"], key="upload_cv_track")
            if cv_name:
                upload_meta["candidate_name"] = cv_name
            upload_meta["bmw_track_label"] = cv_track
    elif upload_cat == "policies" and uploaded_file:
        with st.sidebar.expander("Policy Metadata (optional)"):
            pol_title = st.text_input("Policy Title", key="upload_pol_title")
            pol_category = st.selectbox("Category", ["leave", "conduct", "remote_work", "benefits", "termination", "working_hours", "performance_review", "onboarding", "expenses", "general"], key="upload_pol_cat")
            if pol_title:
                upload_meta["title"] = pol_title
            upload_meta["category"] = pol_category

    if uploaded_file and st.sidebar.button("Ingest PDF", type="primary", use_container_width=True):
        with st.sidebar:
            with st.spinner("Uploading & ingesting..."):
                try:
                    import io
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    form_data = {"category": upload_cat, "metadata_json": json.dumps(upload_meta)}
                    with httpx.Client(timeout=60.0) as client:
                        r = client.post(f"{AGENT_A_URL}/ingest/upload", files=files, data=form_data)
                        r.raise_for_status()
                        result = r.json()
                    st.success(f"Ingested **{result['filename']}** into `{result['collection']}` ({result['chunks']} chunks)")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session")
    sid = st.session_state.get("session_id")
    if sid:
        st.sidebar.code(sid, language=None)
        if st.sidebar.button("New session", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state._clear_chat_input = True
            st.rerun()
    else:
        st.sidebar.caption("Send a message to start a session.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Voice")
    st.sidebar.caption(
        "Whisper STT → Agent A → TTS (MP3). Needs a platform **sk-** OpenAI key in `.env`. "
        "Limit: 5 voice requests/min."
    )
    voice_recording = None
    if hasattr(st, "audio_input"):
        voice_recording = st.sidebar.audio_input("Record a question", key="itip_audio_in")
    voice_file = st.sidebar.file_uploader(
        "Or upload audio",
        type=["wav", "mp3", "webm", "m4a"],
        key="itip_voice_upload",
        label_visibility="visible",
    )
    if st.sidebar.button("Send voice to agent", key="itip_voice_send", use_container_width=True):
        audio_bytes: bytes | None = None
        fname = "voice.wav"
        mime = "audio/wav"
        if voice_recording is not None:
            audio_bytes = voice_recording.getvalue()
            fname = "recording.webm"
            mime = getattr(voice_recording, "type", None) or "audio/webm"
        elif voice_file is not None:
            audio_bytes = voice_file.getvalue()
            fname = voice_file.name or "upload.wav"
            mime = voice_file.type or "audio/wav"
        if not audio_bytes:
            st.sidebar.warning("Record audio or upload a file first.")
        else:
            with st.sidebar:
                with st.spinner("Transcribing & generating reply…"):
                    vr = call_agent_voice(
                        audio_bytes,
                        fname,
                        mime,
                        st.session_state.get("session_id"),
                        role_key,
                    )
            if vr["ok"]:
                tr = vr["transcript"] or "(no transcript)"
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"**Voice:** {tr}",
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": vr["reply_text"] or "(empty reply)",
                    "meta": {
                        "specialist_used": "none",
                        "route_taken": "voice",
                        "iterations": 0,
                        "guardrail_blocked": "flagged as a potential prompt injection"
                        in (vr["reply_text"] or "").lower(),
                    },
                })
                if vr.get("session_id"):
                    st.session_state.session_id = vr["session_id"]
                st.session_state["itip_last_voice_mp3"] = vr["mp3"]
                st.rerun()
            else:
                st.sidebar.error(
                    f"Voice failed ({vr['status_code']}): {vr['detail'][:800] or 'Unknown error'}"
                )

    if st.session_state.get("itip_last_voice_mp3"):
        st.sidebar.audio(st.session_state["itip_last_voice_mp3"], format="audio/mp3")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Prompts")
    for label, prompt in role["quick_prompts"]:
        if st.sidebar.button(label, key=f"qp_{label}", use_container_width=True):
            st.session_state["_pending_prompt"] = prompt

    st.sidebar.markdown("---")
    if st.sidebar.button("Sign Out", use_container_width=True, type="primary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    # --- Hero ---
    st.markdown(
        f"""
<div class="itip-hero">
  <div class="itip-badge"><span class="material-symbols-rounded" style="font-size:14px;">{role['icon']}</span> {role['label']} &middot; Multi-Agent &middot; RAG</div>
  <h1 class="itip-title">InMind Talent Intelligence</h1>
  <p class="itip-sub">{role['description']}</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.pop("_clear_chat_input", False):
        st.session_state.itip_chat_box = ""
    pending = st.session_state.pop("_pending_prompt", None)
    if pending:
        st.session_state["_input_default"] = pending

    # --- Chat area ---
    if not st.session_state.get("messages"):
        st.markdown(
            f"""
<div class="itip-chat-shell"><div class="itip-chat-inner">
<div class="itip-empty">
  <div class="itip-empty-icon"><span class="material-symbols-rounded" style="font-size:2.2rem;color:{role['accent']};">{role['icon']}</span></div>
  <p style="color:#f1f5f9;font-weight:600;">Welcome, {role['label']}</p>
  <p style="font-size:0.88rem;opacity:0.8;">{role['description']}<br>Use the quick prompts or type below to begin.</p>
</div>
</div></div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="itip-chat-shell"><div class="itip-chat-inner">', unsafe_allow_html=True)
        render_chat_messages(st.session_state.messages)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # --- Input ---
    st.markdown('<div class="itip-input-wrap">', unsafe_allow_html=True)
    default = st.session_state.pop("_input_default", None)
    if default is not None:
        st.session_state.itip_chat_box = default
    if "itip_chat_box" not in st.session_state:
        st.session_state.itip_chat_box = ""

    with st.form("chat_form", clear_on_submit=False):
        c1, c2 = st.columns([6, 1])
        with c1:
            st.text_area(
                "Message", key="itip_chat_box",
                placeholder=f"Ask anything as {role['label']}...",
                height=90, label_visibility="collapsed",
            )
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            send = st.form_submit_button("Send", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if send:
        raw = (st.session_state.get("itip_chat_box") or "").strip()
        if raw:
            st.session_state.messages.append({"role": "user", "content": raw})
            status_area = st.empty()
            status_area.caption("Connecting to agent...")

            data = call_agent_stream(raw, st.session_state.get("session_id"), role_key, status_area)

            st.session_state.session_id = data.get("session_id")
            st.session_state.messages.append({
                "role": "assistant",
                "content": data.get("reply") or "(empty response)",
                "meta": {
                    "specialist_used": data.get("specialist_used"),
                    "route_taken": data.get("route_taken"),
                    "iterations": data.get("iterations"),
                    "guardrail_blocked": data.get("guardrail_blocked"),
                },
            })
            st.session_state._clear_chat_input = True
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="InMind ITIP",
        page_icon=":material/auto_awesome:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not st.session_state.get("logged_in"):
        render_login()
    else:
        render_chat()


if __name__ == "__main__":
    main()

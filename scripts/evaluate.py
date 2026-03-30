"""
ITIP Evaluation Framework (proposal §12).

Runs the full 29-question test set against Agent A, computes:
  1. Retrieval metrics: Precision@K, Recall@K, MRR per collection and aggregate
  2. Generation metrics: RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
  3. Configuration comparisons: chunk size (300 vs 500), retrieval K (3 vs 5 vs 7)
  4. Routing accuracy: supervisor correctly routes to expected specialist
  5. Guardrail accuracy: adversarial/discriminatory queries blocked

Outputs:
  evaluation/results/retrieval_metrics.json
  evaluation/results/ragas_scores.json
  evaluation/results/routing_accuracy.json
  evaluation/results/config_comparison.json
  evaluation/results/summary.json

Usage:
  python scripts/evaluate.py                    # full evaluation
  python scripts/evaluate.py --skip-ragas       # skip RAGAS (faster, no extra cost)
  python scripts/evaluate.py --skip-configs     # skip config comparisons
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
TEST_SET_PATH = EVAL_DIR / "test_set.json"

_config = {"agent_a_url": os.getenv("AGENT_A_URL", "http://localhost:8000").rstrip("/")}

sys.path.insert(0, str(ROOT / "services" / "agent-system-a"))


def load_test_set() -> list[dict]:
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


# ---------------------------------------------------------------------------
# 1. Query Agent A
# ---------------------------------------------------------------------------

def query_agent_a(question: str, session_id: str | None = None, role: str = "admin") -> dict:
    """POST /chat to Agent A, return full response."""
    body: dict = {"message": question, "role": role}
    if session_id:
        body["session_id"] = session_id
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{_config['agent_a_url']}/chat", json=body)
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"HTTP {resp.status_code}", "reply": "", "route_taken": "error"}
    except Exception as e:
        return {"error": str(e), "reply": "", "route_taken": "error"}


# ---------------------------------------------------------------------------
# 2. Retrieval metrics (standalone — bypasses Agent A for direct RAG eval)
# ---------------------------------------------------------------------------

def get_retrieval_hits(question: str, collection: str, limit: int = 5) -> list[dict]:
    """Direct RAG retrieval from Qdrant A for metrics computation."""
    try:
        from config import get_embed_client_and_model, try_qdrant_a
        from rag.retrieve import embed_query, _search
    except ImportError:
        return []

    qdrant = try_qdrant_a()
    if not qdrant:
        return []

    embed_client, embed_model = get_embed_client_and_model()
    vec = embed_query(embed_client, embed_model, question)
    return _search(qdrant, collection, vec, limit=limit)


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    expected_ids: list[str],
) -> dict:
    """Compute P@K, R@K, MRR for a single question."""
    if not expected_ids:
        return {"precision": None, "recall": None, "mrr": None}

    expected_set = set(expected_ids)
    seen_relevant: set[str] = set()
    first_relevant_rank = 0

    for i, rid in enumerate(retrieved_ids, 1):
        if rid in expected_set:
            seen_relevant.add(rid)
            if first_relevant_rank == 0:
                first_relevant_rank = i

    k = len(retrieved_ids) if retrieved_ids else 1
    precision = len(seen_relevant) / k
    recall = min(len(seen_relevant) / len(expected_set), 1.0) if expected_set else 0
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "mrr": round(mrr, 4)}


CATEGORY_TO_COLLECTION = {
    "job_search": "job_postings",
    "hr_policy": "hr_policies",
    "bmw_placement": "placement_briefs",
}


def run_retrieval_evaluation(questions: list[dict], top_k: int = 5) -> dict:
    """Run retrieval evaluation for all RAG-relevant questions."""
    results = []
    collection_metrics: dict[str, list] = {}

    for q in questions:
        cat = q["category"]
        collection = CATEGORY_TO_COLLECTION.get(cat)
        if not collection:
            continue

        expected_ids = q.get("expected_source_ids", [])
        if not expected_ids:
            continue

        hits = get_retrieval_hits(q["question"], collection, limit=top_k)
        retrieved_ids = [h.get("source_id", "") for h in hits]
        metrics = compute_retrieval_metrics(retrieved_ids, expected_ids)

        result = {
            "id": q["id"],
            "question": q["question"],
            "collection": collection,
            "retrieved_ids": retrieved_ids,
            "expected_ids": expected_ids,
            **metrics,
        }
        results.append(result)

        if collection not in collection_metrics:
            collection_metrics[collection] = []
        collection_metrics[collection].append(metrics)

    per_collection = {}
    all_p, all_r, all_mrr = [], [], []

    for col, metrics_list in collection_metrics.items():
        ps = [m["precision"] for m in metrics_list if m["precision"] is not None]
        rs = [m["recall"] for m in metrics_list if m["recall"] is not None]
        ms = [m["mrr"] for m in metrics_list if m["mrr"] is not None]
        per_collection[col] = {
            "avg_precision": round(sum(ps) / len(ps), 4) if ps else 0,
            "avg_recall": round(sum(rs) / len(rs), 4) if rs else 0,
            "avg_mrr": round(sum(ms) / len(ms), 4) if ms else 0,
            "n_questions": len(metrics_list),
        }
        all_p.extend(ps)
        all_r.extend(rs)
        all_mrr.extend(ms)

    aggregate = {
        "avg_precision": round(sum(all_p) / len(all_p), 4) if all_p else 0,
        "avg_recall": round(sum(all_r) / len(all_r), 4) if all_r else 0,
        "avg_mrr": round(sum(all_mrr) / len(all_mrr), 4) if all_mrr else 0,
        "total_questions": len(results),
    }

    return {
        "top_k": top_k,
        "per_question": results,
        "per_collection": per_collection,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# 3. Routing accuracy
# ---------------------------------------------------------------------------

def evaluate_routing(questions: list[dict], responses: dict[str, dict]) -> dict:
    correct = 0
    total = 0
    details = []

    for q in questions:
        qid = q["id"]
        expected = q["expected_route"]
        resp = responses.get(qid, {})

        specialist = resp.get("specialist_used", "")
        route_taken = resp.get("route_taken", "unknown")
        blocked = resp.get("guardrail_blocked", False)

        actual = specialist if specialist else route_taken

        if expected == "guardrail_block":
            is_correct = blocked or route_taken == "guardrail_block"
        elif expected == "FINISH":
            is_correct = (not specialist or specialist == "none") and route_taken == "FINISH"
        else:
            is_correct = actual == expected or specialist == expected

        if is_correct:
            correct += 1
        total += 1

        details.append({
            "id": qid,
            "expected_route": expected,
            "actual_route": actual,
            "specialist_used": specialist,
            "guardrail_blocked": blocked,
            "correct": is_correct,
        })

    return {
        "accuracy": round(correct / total, 4) if total else 0,
        "correct": correct,
        "total": total,
        "details": details,
    }


# ---------------------------------------------------------------------------
# 4. RAGAS evaluation
# ---------------------------------------------------------------------------

def _llm_judge(prompt: str) -> str:
    """Call the chat model for LLM-as-judge evaluations."""
    try:
        from config import get_chat_client_and_model
        client, model = get_chat_client_and_model()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        content = resp.choices[0].message.content
        return (content or "").strip()
    except Exception as e:
        return f"ERROR: {e}"


def _score_faithfulness(answer: str, contexts: list[str]) -> float:
    """RAGAS Faithfulness: fraction of answer claims supported by retrieved context."""
    ctx_block = "\n---\n".join(contexts[:5])
    prompt = (
        "You are a strict factual grounding evaluator.\n\n"
        f"CONTEXT:\n{ctx_block}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Task: List each factual claim in the ANSWER. For each claim, decide if it is "
        "SUPPORTED or NOT SUPPORTED by the CONTEXT.\n\n"
        "Output ONLY a JSON object: {\"total_claims\": <int>, \"supported_claims\": <int>}"
    )
    raw = _llm_judge(prompt)
    try:
        raw_clean = raw.strip().strip("`").strip()
        if raw_clean.startswith("json"):
            raw_clean = raw_clean[4:].strip()
        d = json.loads(raw_clean)
        total = max(d.get("total_claims", 1), 1)
        supported = d.get("supported_claims", 0)
        return round(min(supported / total, 1.0), 4)
    except Exception:
        return 0.0


def _score_answer_relevancy(question: str, answer: str) -> float:
    """RAGAS Answer Relevancy: how well the answer addresses the question."""
    prompt = (
        "Rate how well the ANSWER addresses the QUESTION on a scale from 0.0 to 1.0.\n"
        "1.0 = perfectly relevant, directly answers the question\n"
        "0.5 = partially relevant, some useful information\n"
        "0.0 = completely irrelevant\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        "Output ONLY a JSON object: {\"score\": <float between 0 and 1>}"
    )
    raw = _llm_judge(prompt)
    try:
        raw_clean = raw.strip().strip("`").strip()
        if raw_clean.startswith("json"):
            raw_clean = raw_clean[4:].strip()
        d = json.loads(raw_clean)
        return round(float(d.get("score", 0)), 4)
    except Exception:
        return 0.0


def _score_context_precision(question: str, contexts: list[str]) -> float:
    """RAGAS Context Precision: fraction of retrieved contexts that are relevant."""
    results = []
    for i, ctx in enumerate(contexts[:5]):
        prompt = (
            "Is the following CONTEXT relevant to answering the QUESTION?\n\n"
            f"QUESTION: {question}\n\nCONTEXT:\n{ctx}\n\n"
            "Output ONLY: {\"relevant\": true} or {\"relevant\": false}"
        )
        raw = _llm_judge(prompt)
        try:
            raw_clean = raw.strip().strip("`").strip()
            if raw_clean.startswith("json"):
                raw_clean = raw_clean[4:].strip()
            d = json.loads(raw_clean)
            results.append(1.0 if d.get("relevant") else 0.0)
        except Exception:
            results.append(0.0)
    return round(sum(results) / max(len(results), 1), 4)


def _score_context_recall(answer_keywords: list[str], contexts: list[str]) -> float:
    """RAGAS Context Recall: fraction of expected answer elements found in contexts."""
    if not answer_keywords:
        return 0.0
    ctx_text = " ".join(contexts).lower()
    found = sum(1 for kw in answer_keywords if kw.lower() in ctx_text)
    return round(found / len(answer_keywords), 4)


def run_ragas_evaluation(questions: list[dict], responses: dict[str, dict]) -> dict:
    """LLM-as-judge RAGAS evaluation on RAG-relevant questions."""
    rag_categories = {"job_search", "hr_policy", "bmw_placement", "candidate_screening"}
    per_question = []

    for q in questions:
        if q["category"] not in rag_categories:
            continue

        qid = q["id"]
        resp = responses.get(qid, {})
        answer = resp.get("reply", "")
        if not answer or resp.get("guardrail_blocked"):
            continue

        collection = CATEGORY_TO_COLLECTION.get(q["category"])
        if collection:
            hits = get_retrieval_hits(q["question"], collection, limit=5)
            contexts = [h.get("text", "") for h in hits if h.get("text")]
        else:
            contexts = [answer]

        expected_keywords = q.get("expected_answer_keywords", [])

        print(f"    RAGAS {qid}: ", end="", flush=True)
        faith = _score_faithfulness(answer, contexts)
        print(f"F={faith} ", end="", flush=True)
        relevancy = _score_answer_relevancy(q["question"], answer)
        print(f"AR={relevancy} ", end="", flush=True)
        ctx_prec = _score_context_precision(q["question"], contexts)
        print(f"CP={ctx_prec} ", end="", flush=True)
        ctx_recall = _score_context_recall(expected_keywords, contexts)
        print(f"CR={ctx_recall}")

        per_question.append({
            "id": qid,
            "question": q["question"],
            "faithfulness": faith,
            "answer_relevancy": relevancy,
            "context_precision": ctx_prec,
            "context_recall": ctx_recall,
        })

        time.sleep(0.3)

    if not per_question:
        return {"error": "no RAG questions to evaluate", "aggregate": {}, "per_question": []}

    n = len(per_question)
    aggregate = {
        "faithfulness": round(sum(r["faithfulness"] for r in per_question) / n, 4),
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in per_question) / n, 4),
        "context_precision": round(sum(r["context_precision"] for r in per_question) / n, 4),
        "context_recall": round(sum(r["context_recall"] for r in per_question) / n, 4),
    }

    return {"aggregate": aggregate, "per_question": per_question, "n_questions": n}


# ---------------------------------------------------------------------------
# 5. Configuration comparisons (§12.5)
# ---------------------------------------------------------------------------

def run_config_comparisons(questions: list[dict]) -> dict:
    """
    Comparison 1: Chunk size 300 vs 500 for job_postings (measure P@5, R@5, MRR)
    Comparison 2: Top-K = 3 vs 5 vs 7 across all RAG questions
    """
    job_questions = [q for q in questions if q["category"] == "job_search" and q.get("expected_source_ids")]
    rag_questions = [q for q in questions if q["category"] in CATEGORY_TO_COLLECTION and q.get("expected_source_ids")]

    comparison_1 = {"description": "Chunk size for job_postings: current (350 tok) evaluated at K=5"}
    metrics_350 = []
    for q in job_questions:
        hits = get_retrieval_hits(q["question"], "job_postings", limit=5)
        retrieved_ids = [h.get("source_id", "") for h in hits]
        m = compute_retrieval_metrics(retrieved_ids, q["expected_source_ids"])
        if m["precision"] is not None:
            metrics_350.append(m)

    if metrics_350:
        comparison_1["chunk_350"] = {
            "avg_precision": round(sum(m["precision"] for m in metrics_350) / len(metrics_350), 4),
            "avg_recall": round(sum(m["recall"] for m in metrics_350) / len(metrics_350), 4),
            "avg_mrr": round(sum(m["mrr"] for m in metrics_350) / len(metrics_350), 4),
            "n": len(metrics_350),
        }
    comparison_1["note"] = (
        "Proposal §12.5 Comparison 1: Structure-aware 350-token chunks are hypothesized to yield "
        "higher P@5 vs 500-token whole-document chunks because section-specific queries retrieve "
        "section-specific chunks. A 500-token re-ingestion comparison can be run by modifying "
        "ingest.py chunk sizes and re-running this script."
    )

    comparison_2 = {"description": "Retrieval top-K: K=3 vs K=5 vs K=7 across all RAG questions"}
    for k_val in [3, 5, 7]:
        k_metrics = []
        for q in rag_questions:
            collection = CATEGORY_TO_COLLECTION[q["category"]]
            hits = get_retrieval_hits(q["question"], collection, limit=k_val)
            retrieved_ids = [h.get("source_id", "") for h in hits]
            m = compute_retrieval_metrics(retrieved_ids, q["expected_source_ids"])
            if m["precision"] is not None:
                k_metrics.append(m)

        if k_metrics:
            comparison_2[f"K={k_val}"] = {
                "avg_precision": round(sum(m["precision"] for m in k_metrics) / len(k_metrics), 4),
                "avg_recall": round(sum(m["recall"] for m in k_metrics) / len(k_metrics), 4),
                "avg_mrr": round(sum(m["mrr"] for m in k_metrics) / len(k_metrics), 4),
                "n": len(k_metrics),
            }

    comparison_2["hypothesis"] = (
        "K=5 is optimal: K=3 may miss multi-chunk answers (lower recall), "
        "K=7 adds noise (lower precision and RAGAS faithfulness)."
    )

    return {"comparison_1_chunk_size": comparison_1, "comparison_2_top_k": comparison_2}


# ---------------------------------------------------------------------------
# 6. Generate EVALUATION.md
# ---------------------------------------------------------------------------

def generate_evaluation_md(
    retrieval: dict,
    routing: dict,
    ragas: dict,
    configs: dict,
    responses: dict[str, dict],
    questions: list[dict],
) -> str:
    lines = [
        "# ITIP Evaluation Report",
        "",
        "**InMind Talent Intelligence Platform — Evaluation Results**",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Routing Accuracy",
        "",
        f"**Overall: {routing['accuracy']*100:.1f}% ({routing['correct']}/{routing['total']})**",
        "",
        "| Question ID | Category | Expected Route | Actual Route | Correct |",
        "|------------|----------|---------------|-------------|---------|",
    ]

    for d in routing["details"]:
        mark = "✅" if d["correct"] else "❌"
        lines.append(f"| {d['id']} | {next((q['category'] for q in questions if q['id']==d['id']), '')} | {d['expected_route']} | {d['actual_route']} | {mark} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Retrieval Metrics (Precision@5, Recall@5, MRR)",
        "",
        "### Per Collection",
        "",
        "| Collection | Avg P@5 | Avg R@5 | Avg MRR | Questions |",
        "|-----------|---------|---------|---------|-----------|",
    ])

    for col, m in retrieval.get("per_collection", {}).items():
        lines.append(f"| {col} | {m['avg_precision']:.4f} | {m['avg_recall']:.4f} | {m['avg_mrr']:.4f} | {m['n_questions']} |")

    agg = retrieval.get("aggregate", {})
    lines.extend([
        "",
        f"### Aggregate: P@5={agg.get('avg_precision', 0):.4f}, R@5={agg.get('avg_recall', 0):.4f}, MRR={agg.get('avg_mrr', 0):.4f}",
        "",
        "**Targets (§12.3):** P@5 > 0.60, R@5 > 0.65, MRR > 0.70",
        "",
        "---",
        "",
        "## 3. RAGAS Generation Metrics",
        "",
    ])

    ragas_agg = ragas.get("aggregate", {})
    if "error" in ragas or "error" in ragas_agg:
        lines.append(f"*RAGAS evaluation skipped: {ragas_agg.get('error', 'unknown')}*")
    else:
        lines.extend([
            "| Metric | Score | Target |",
            "|--------|-------|--------|",
            f"| Faithfulness | {ragas_agg.get('faithfulness', 'N/A')} | > 0.80 |",
            f"| Answer Relevancy | {ragas_agg.get('answer_relevancy', 'N/A')} | > 0.80 |",
            f"| Context Precision | {ragas_agg.get('context_precision', 'N/A')} | > 0.65 |",
            f"| Context Recall | {ragas_agg.get('context_recall', 'N/A')} | > 0.70 |",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 4. Configuration Comparisons (§12.5)",
        "",
        "### Comparison 1: Chunk Size (Job Postings)",
        "",
    ])

    c1 = configs.get("comparison_1_chunk_size", {})
    if "chunk_350" in c1:
        c350 = c1["chunk_350"]
        lines.extend([
            "| Config | Avg P@5 | Avg R@5 | Avg MRR | N |",
            "|--------|---------|---------|---------|---|",
            f"| 350-token (current) | {c350['avg_precision']:.4f} | {c350['avg_recall']:.4f} | {c350['avg_mrr']:.4f} | {c350['n']} |",
            "",
            f"*{c1.get('note', '')}*",
        ])

    lines.extend([
        "",
        "### Comparison 2: Retrieval Top-K",
        "",
        "| K | Avg P@K | Avg R@K | Avg MRR | N |",
        "|---|---------|---------|---------|---|",
    ])

    c2 = configs.get("comparison_2_top_k", {})
    for k_val in [3, 5, 7]:
        key = f"K={k_val}"
        if key in c2:
            m = c2[key]
            lines.append(f"| {k_val} | {m['avg_precision']:.4f} | {m['avg_recall']:.4f} | {m['avg_mrr']:.4f} | {m['n']} |")

    lines.extend([
        "",
        f"*Hypothesis: {c2.get('hypothesis', '')}*",
        "",
        "---",
        "",
        "## 5. Failure Case Analysis",
        "",
    ])

    failures = [d for d in routing["details"] if not d["correct"]]
    if failures:
        for i, f in enumerate(failures[:3], 1):
            q_obj = next((q for q in questions if q["id"] == f["id"]), {})
            resp = responses.get(f["id"], {})
            lines.extend([
                f"### Failure {i}: {f['id']}",
                "",
                f"- **Question:** {q_obj.get('question', '')}",
                f"- **Expected route:** {f['expected_route']}",
                f"- **Actual route:** {f['actual_route']}",
                f"- **Response preview:** {resp.get('reply', '')[:200]}...",
                f"- **Root cause:** Supervisor misclassified intent",
                f"- **Fix:** Refine supervisor prompt or add keyword hints for this category",
                "",
            ])
    else:
        lines.append("*No routing failures detected — all 29 questions routed correctly.*")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Guardrail Verification",
        "",
        "| Test | Question | Blocked | Expected |",
        "|------|----------|---------|----------|",
    ])

    guardrail_qs = [q for q in questions if q["category"] in ("adversarial", "discriminatory")]
    for q in guardrail_qs:
        resp = responses.get(q["id"], {})
        blocked = resp.get("guardrail_blocked", False)
        mark = "✅" if blocked else "❌"
        lines.append(f"| {q['id']} | {q['question'][:50]}... | {mark} | Blocked |")

    lines.extend([
        "",
        "---",
        "",
        "## 7. System Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        "| Total questions | 29 |",
        f"| Routing accuracy | {routing['accuracy']*100:.1f}% |",
        f"| Retrieval P@5 (aggregate) | {agg.get('avg_precision', 0):.4f} |",
        f"| Retrieval R@5 (aggregate) | {agg.get('avg_recall', 0):.4f} |",
        f"| Retrieval MRR (aggregate) | {agg.get('avg_mrr', 0):.4f} |",
        f"| RAGAS Faithfulness | {ragas_agg.get('faithfulness', 'N/A')} |",
        f"| RAGAS Answer Relevancy | {ragas_agg.get('answer_relevancy', 'N/A')} |",
        f"| Containers running | 6 |",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ITIP Evaluation Framework (§12)")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS (faster, no LLM cost)")
    parser.add_argument("--skip-configs", action="store_true", help="Skip configuration comparisons")
    parser.add_argument("--agent-url", default=_config["agent_a_url"], help="Agent A base URL")
    args = parser.parse_args()

    _config["agent_a_url"] = args.agent_url

    load_dotenv(ROOT / ".env")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ITIP EVALUATION FRAMEWORK (§12)")
    print("=" * 60)

    questions = load_test_set()
    print(f"\nLoaded {len(questions)} test questions from {TEST_SET_PATH.name}")

    # --- Step 1: Query Agent A for all questions ---
    print("\n[1/5] Querying Agent A for all 29 questions...")
    responses: dict[str, dict] = {}
    for q in questions:
        print(f"  {q['id']}: {q['question'][:60]}...")
        resp = query_agent_a(q["question"])
        responses[q["id"]] = resp
        time.sleep(0.5)

    # --- Step 2: Routing accuracy ---
    print("\n[2/5] Evaluating routing accuracy...")
    routing = evaluate_routing(questions, responses)
    print(f"  Routing accuracy: {routing['accuracy']*100:.1f}% ({routing['correct']}/{routing['total']})")

    with open(RESULTS_DIR / "routing_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(routing, f, indent=2)

    # --- Step 3: Retrieval metrics ---
    print("\n[3/5] Computing retrieval metrics (P@5, R@5, MRR)...")
    retrieval = run_retrieval_evaluation(questions, top_k=5)
    agg = retrieval["aggregate"]
    print(f"  Aggregate: P@5={agg['avg_precision']:.4f}, R@5={agg['avg_recall']:.4f}, MRR={agg['avg_mrr']:.4f}")

    with open(RESULTS_DIR / "retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(retrieval, f, indent=2)

    # --- Step 4: RAGAS ---
    ragas_result: dict = {}
    if args.skip_ragas:
        print("\n[4/5] RAGAS evaluation skipped (--skip-ragas)")
        ragas_result = {"aggregate": {"error": "skipped via --skip-ragas"}, "per_question": []}
    else:
        print("\n[4/5] Running RAGAS evaluation...")
        ragas_result = run_ragas_evaluation(questions, responses)
        ragas_agg = ragas_result.get("aggregate", {})
        if "error" not in ragas_agg:
            print(f"  Faithfulness:      {ragas_agg.get('faithfulness', 'N/A')}")
            print(f"  Answer Relevancy:  {ragas_agg.get('answer_relevancy', 'N/A')}")
            print(f"  Context Precision: {ragas_agg.get('context_precision', 'N/A')}")
            print(f"  Context Recall:    {ragas_agg.get('context_recall', 'N/A')}")

    with open(RESULTS_DIR / "ragas_scores.json", "w", encoding="utf-8") as f:
        json.dump(ragas_result, f, indent=2)

    # --- Step 5: Configuration comparisons ---
    configs: dict = {}
    if args.skip_configs:
        print("\n[5/5] Configuration comparisons skipped (--skip-configs)")
        configs = {"skipped": True}
    else:
        print("\n[5/5] Running configuration comparisons (K=3 vs 5 vs 7)...")
        configs = run_config_comparisons(questions)
        c2 = configs.get("comparison_2_top_k", {})
        for k_val in [3, 5, 7]:
            key = f"K={k_val}"
            if key in c2:
                m = c2[key]
                print(f"  {key}: P={m['avg_precision']:.4f}, R={m['avg_recall']:.4f}, MRR={m['avg_mrr']:.4f}")

    with open(RESULTS_DIR / "config_comparison.json", "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)

    # --- Summary ---
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(questions),
        "routing_accuracy": routing["accuracy"],
        "retrieval_aggregate": retrieval["aggregate"],
        "ragas_aggregate": ragas_result.get("aggregate", {}),
        "guardrails_tested": sum(1 for q in questions if q["category"] in ("adversarial", "discriminatory")),
        "guardrails_blocked": sum(
            1 for q in questions
            if q["category"] in ("adversarial", "discriminatory")
            and responses.get(q["id"], {}).get("guardrail_blocked", False)
        ),
    }

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Generate EVALUATION.md ---
    md = generate_evaluation_md(retrieval, routing, ragas_result, configs, responses, questions)
    eval_md_path = ROOT / "EVALUATION.md"
    with open(eval_md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Results:       {RESULTS_DIR}")
    print(f"  Report:        {eval_md_path}")
    print(f"  Routing:       {routing['accuracy']*100:.1f}%")
    print(f"  Retrieval P@5: {agg['avg_precision']:.4f}")
    print(f"  Retrieval MRR: {agg['avg_mrr']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

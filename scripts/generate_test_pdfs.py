"""Generate test PDFs that are NOT in the database, for testing the live upload feature."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "test_uploads"
TEST_DIR.mkdir(parents=True, exist_ok=True)


def sanitize(text) -> str:
    if text is None:
        return ""
    text = str(text)
    text = (
        text.replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2013", "-").replace("\u2014", "-")
        .replace("\u2026", "...").replace("\u2022", "-")
    )
    text = re.sub(r"#{1,6}\s*", "", text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ---------------------------------------------------------------------------
# 1. Test CV: A unique candidate NOT in the database
# ---------------------------------------------------------------------------
def make_test_cv():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 9, "Zara Nakamura", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 60, 120)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, "zara.nakamura@example.com | Languages: English, Japanese, French | Track: AI | Experience: 3 years",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 5, "PROFESSIONAL SUMMARY", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 4, sanitize(
        "Innovative AI researcher specializing in reinforcement learning and autonomous systems. "
        "Published 4 papers on multi-agent coordination at top-tier conferences including NeurIPS and ICML. "
        "Passionate about applying AI to real-world robotics and logistics optimization."
    ))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 5, "EDUCATION", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 4, "MSc in Machine Learning, ETH Zurich (2023)")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 5, "SKILLS", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 3.5, sanitize(
        "Python | PyTorch | TensorFlow | Reinforcement Learning | Multi-Agent Systems | "
        "Computer Vision | ROS2 | C++ | Docker | Kubernetes | MLOps | "
        "Research Writing | Public Speaking | Japanese (Native) | French (B2)"
    ))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 5, "EXPERIENCE", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 3.5, sanitize(
        "- Developed a multi-agent reinforcement learning framework for warehouse robot coordination, "
        "reducing path conflicts by 40% in simulation. "
        "- Published 'Cooperative Navigation via Decentralized Actor-Critic' at NeurIPS 2023 workshop. "
        "- Built real-time object detection pipeline using YOLOv8 for autonomous forklift navigation, "
        "achieving 94% mAP on custom industrial dataset. "
        "- Designed and deployed ML inference microservice on Kubernetes handling 500 req/s with <50ms latency. "
        "- Mentored 3 junior researchers on RL fundamentals and experimental methodology. "
        "- Contributed to open-source ROS2 navigation stack with collision avoidance improvements."
    ))

    out = TEST_DIR / "test_cv_zara_nakamura.pdf"
    pdf.output(str(out))
    print(f"  Created: {out.name}")
    return out


# ---------------------------------------------------------------------------
# 2. Test HR Policy: A completely new policy NOT in the database
# ---------------------------------------------------------------------------
def make_test_policy():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 7, "inmind.ai", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 4, "Human Resources Department", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_draw_color(30, 60, 120)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 8, "Parental Leave Policy")
    pdf.ln(2)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Category: Leave | Effective: March 2025 | Version 1.0", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    body = (
        "The Parental Leave Policy at inmind.ai provides guidelines for maternity and paternity leave "
        "in accordance with Lebanese Labour Law and company values. "
        "\n\n"
        "Maternity Leave: Female employees are entitled to 10 weeks (70 calendar days) of paid maternity leave. "
        "This consists of a mandatory pre-delivery period of at least 3 weeks before the expected due date "
        "and the remaining weeks post-delivery. The employee must notify HR at least 8 weeks before the expected "
        "delivery date and provide a medical certificate confirming the pregnancy and expected due date. "
        "During maternity leave, the employee receives 100% of their base salary. "
        "\n\n"
        "Paternity Leave: Male employees are entitled to 5 working days of paid paternity leave, to be taken "
        "within 30 days of the child's birth. An additional 5 days of unpaid leave may be requested. "
        "The employee must notify their supervisor at least 2 weeks in advance when possible. "
        "\n\n"
        "Adoption Leave: Employees who legally adopt a child under the age of 5 are entitled to 6 weeks of "
        "paid leave from the date the child is placed in their custody. Both adoptive parents are eligible. "
        "\n\n"
        "Return to Work: Employees returning from parental leave are guaranteed the same position or an "
        "equivalent role with the same compensation and benefits. A gradual return schedule may be arranged "
        "with management approval. Nursing mothers are entitled to two 30-minute breaks per day during "
        "working hours for the first 12 months after returning to work. "
        "\n\n"
        "This policy reflects inmind.ai's commitment to supporting employees through major life events "
        "while maintaining compliance with Lebanese Labour Law. Any questions should be directed to "
        "the HR department."
    )
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, sanitize(body))

    out = TEST_DIR / "test_policy_parental_leave.pdf"
    pdf.output(str(out))
    print(f"  Created: {out.name}")
    return out


# ---------------------------------------------------------------------------
# 3. Test Job Posting: A brand new position NOT in the database
# ---------------------------------------------------------------------------
def make_test_job():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 7, "inmind.ai", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 60, 120)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 9, "Prompt Engineer", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, "Location: Beirut, Lebanon | Type: Full-time | Level: Mid-level | Salary: 45,000 - 65,000 USD/year",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 6, "About the Role", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 4.5, sanitize(
        "inmind.ai is seeking a Prompt Engineer to design, test, and optimize prompts for our "
        "multi-agent AI systems. You will work closely with the AI team to craft system prompts, "
        "few-shot examples, and evaluation frameworks that maximize the accuracy and reliability "
        "of our LLM-powered talent intelligence platform. This is a unique opportunity to shape "
        "how AI agents communicate and reason in a production HR tech environment."
    ))
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 6, "Requirements", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    reqs = [
        "Strong understanding of LLM capabilities (GPT-4, Claude, Gemini)",
        "Experience designing system prompts and multi-turn conversation flows",
        "Proficiency in Python for prompt testing and evaluation automation",
        "Knowledge of RAG pipelines and retrieval-augmented prompting strategies",
    ]
    pdf.multi_cell(0, 4.5, "\n".join(f"- {r}" for r in reqs))
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 6, "Nice to Have", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    nice = [
        "Experience with LangChain, LangGraph, or Google ADK",
        "Background in HR tech or recruitment platforms",
        "Familiarity with evaluation frameworks like RAGAS",
    ]
    pdf.multi_cell(0, 4.5, "\n".join(f"- {r}" for r in nice))

    out = TEST_DIR / "test_job_prompt_engineer.pdf"
    pdf.output(str(out))
    print(f"  Created: {out.name}")
    return out


def main():
    print(f"Generating test PDFs in {TEST_DIR}...\n")
    make_test_cv()
    make_test_policy()
    make_test_job()
    print(f"\nDone! 3 test PDFs ready in: {TEST_DIR}")
    print("Upload these through the Streamlit UI to test the live upload feature.")


if __name__ == "__main__":
    main()

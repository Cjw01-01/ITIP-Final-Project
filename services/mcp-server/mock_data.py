"""
Mock scheduling data for the MCP server (proposal §8.3).

In production, this would be replaced by a real HRIS / calendar API.
For project scope: simple in-memory + JSON-file store.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

INTERVIEWERS = [
    {
        "id": "int_001",
        "name": "Dr. Sarah Khoury",
        "specialisation": "AI / Machine Learning",
        "tracks": ["AI"],
        "available_slots": {
            "2026-03-24": ["09:00", "11:00", "14:00"],
            "2026-03-25": ["10:00", "13:00", "15:00"],
            "2026-03-26": ["09:00", "11:00"],
        },
    },
    {
        "id": "int_002",
        "name": "Eng. Marc Haddad",
        "specialisation": "Backend / Systems",
        "tracks": ["Backend"],
        "available_slots": {
            "2026-03-24": ["10:00", "14:00", "16:00"],
            "2026-03-25": ["09:00", "11:00", "14:00"],
            "2026-03-26": ["10:00", "13:00", "15:00"],
        },
    },
    {
        "id": "int_003",
        "name": "Dr. Lina Nassar",
        "specialisation": "Robotics / ROS2",
        "tracks": ["Robotics", "Simulation"],
        "available_slots": {
            "2026-03-24": ["09:00", "13:00"],
            "2026-03-25": ["10:00", "14:00", "16:00"],
            "2026-03-26": ["09:00", "11:00", "14:00"],
        },
    },
    {
        "id": "int_004",
        "name": "Eng. Karim Fares",
        "specialisation": "Frontend / UI Engineering",
        "tracks": ["Frontend"],
        "available_slots": {
            "2026-03-24": ["11:00", "14:00"],
            "2026-03-25": ["09:00", "11:00", "13:00"],
            "2026-03-26": ["10:00", "14:00", "16:00"],
        },
    },
    {
        "id": "int_005",
        "name": "Dr. Rami Tabbara",
        "specialisation": "Simulation / Digital Twin",
        "tracks": ["Simulation", "AI"],
        "available_slots": {
            "2026-03-24": ["10:00", "15:00"],
            "2026-03-25": ["09:00", "11:00", "14:00", "16:00"],
            "2026-03-26": ["09:00", "13:00"],
        },
    },
    {
        "id": "int_006",
        "name": "Eng. Nadia El-Amine",
        "specialisation": "General Technical / DevOps",
        "tracks": ["AI", "Backend", "Frontend", "Robotics", "Simulation"],
        "available_slots": {
            "2026-03-24": ["09:00", "10:00", "11:00", "14:00", "15:00"],
            "2026-03-25": ["09:00", "10:00", "14:00"],
            "2026-03-26": ["10:00", "11:00", "13:00", "14:00"],
        },
    },
]

# In-memory stores (reset on container restart; fine for demo)
_booked_interviews: list[dict] = []
_dispatched_assessments: list[dict] = []


def find_interviewers(role: str, date: str, track: str | None = None) -> list[dict]:
    results = []
    for iv in INTERVIEWERS:
        if track and track not in iv["tracks"]:
            continue
        slots = iv["available_slots"].get(date, [])
        if not slots:
            continue
        results.append({
            "interviewer_id": iv["id"],
            "name": iv["name"],
            "specialisation": iv["specialisation"],
            "available_slots": slots,
        })
    return results


def book_interview(
    candidate_id: str,
    interviewer_id: str,
    dt: str,
    interview_type: str,
) -> dict:
    confirmation_id = f"CONF-{uuid.uuid4().hex[:8].upper()}"
    interviewer = next((iv for iv in INTERVIEWERS if iv["id"] == interviewer_id), None)
    name = interviewer["name"] if interviewer else "Unknown"
    record = {
        "confirmation_id": confirmation_id,
        "candidate": candidate_id,
        "interviewer": name,
        "interviewer_id": interviewer_id,
        "datetime": dt,
        "interview_type": interview_type,
        "zoom_link": f"https://zoom.us/j/{uuid.uuid4().hex[:10]}",
        "status": "confirmed",
    }
    _booked_interviews.append(record)
    return record


def dispatch_assessment(candidate_id: str, assessment_type: str) -> dict:
    tracking_id = f"ASSESS-{uuid.uuid4().hex[:8].upper()}"
    deadline = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    record = {
        "tracking_id": tracking_id,
        "candidate": candidate_id,
        "assessment_type": assessment_type,
        "deadline": deadline,
        "link": f"https://assessments.inmind.ai/{tracking_id.lower()}",
        "status": "dispatched",
    }
    _dispatched_assessments.append(record)
    return record

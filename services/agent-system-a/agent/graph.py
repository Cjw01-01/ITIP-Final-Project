"""
Build the LangGraph supervisor-specialist graph (proposal §Agent System A architecture).

Flow:
  Entry → supervisor → (conditional) → specialist → supervisor → ... → FINISH
  Max 8 routing iterations.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.state import AgentState
from agent.supervisor import supervisor_node
from agent.specialists.job_search import job_search_node
from agent.specialists.policy import policy_node
from agent.specialists.candidate_screener import candidate_screener_node
from agent.specialists.bmw_placement import bmw_placement_node


def _route_after_supervisor(state: AgentState) -> str:
    route = state.get("next_route", "FINISH")
    if route == "FINISH":
        return "end"
    return route


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("job_search", job_search_node)
    graph.add_node("policy", policy_node)
    graph.add_node("candidate_screener", candidate_screener_node)
    graph.add_node("bmw_placement", bmw_placement_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "job_search": "job_search",
            "policy": "policy",
            "candidate_screener": "candidate_screener",
            "bmw_placement": "bmw_placement",
            "end": END,
        },
    )

    graph.add_edge("job_search", "supervisor")
    graph.add_edge("policy", "supervisor")
    graph.add_edge("candidate_screener", "supervisor")
    graph.add_edge("bmw_placement", "supervisor")

    return graph


def compile_graph():
    """Return a compiled runnable graph."""
    return build_graph().compile()

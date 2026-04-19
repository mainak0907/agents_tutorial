"""
agents/state.py
────────────────
All LangGraph state definitions live here.

LangGraph concept:
  - Each graph node receives the ENTIRE state dict and returns a PARTIAL dict
    containing only the keys it wants to update.
  - State is immutable between nodes; LangGraph merges updates automatically.

State hierarchy in this project:
  ┌──────────────────────────────────────────────────────────────┐
  │                     MasterState                              │
  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
  │  │  HRAgentState   │  │FinanceAgentState│  │PlannerState │ │
  │  └─────────────────┘  └─────────────────┘  └─────────────┘ │
  └──────────────────────────────────────────────────────────────┘
  MasterState embeds all sub-state fields and adds synthesis outputs.
"""

from typing import Any
from typing_extensions import TypedDict


# ──────────────────────────────────────────────────────────────
# Sub-states  (one per specialised agent)
# ──────────────────────────────────────────────────────────────

class PlannerState(TypedDict, total=False):
    """
    The planner's job: understand the user query and decompose it into
    sub-tasks for the specialised agents.
    """
    user_query: str                    # raw input from the user
    run_id: str                        # unique ID for this pipeline run
    planned_tasks: list[str]           # e.g. ["hr_analysis", "finance_analysis"]
    planner_reasoning: str             # LLM's chain-of-thought


class HRAgentState(TypedDict, total=False):
    """
    HR agent: fetches employee headcount, role distribution, avg salary per dept.
    """
    hr_raw_data: list[dict[str, Any]]  # rows from Oracle
    hr_analysis: str                   # LLM narrative on the HR data
    hr_status: str                     # "pending" | "complete" | "error"
    hr_error: str                      # error message if status == "error"


class FinanceAgentState(TypedDict, total=False):
    """
    Finance agent: fetches salary totals, per-department payroll, anomalies.
    """
    finance_raw_data: list[dict[str, Any]]
    finance_analysis: str
    finance_status: str
    finance_error: str


# ──────────────────────────────────────────────────────────────
# Master state  (union of all sub-states + synthesis)
# ──────────────────────────────────────────────────────────────

class MasterState(
    PlannerState,
    HRAgentState,
    FinanceAgentState,
    total=False,
):
    """
    MasterState is the single graph state object passed through every node.

    LangGraph passes this to every node; each node reads what it needs and
    returns only the keys it modifies.  No node needs to copy through
    fields it doesn't touch.

    Extra fields added by the synthesiser node:
    """
    recommendations: list[str]         # actionable bullet points
    executive_summary: str             # final LLM-generated summary
    pipeline_status: str               # "running" | "complete" | "failed"
    persisted: bool                    # True once written to Oracle

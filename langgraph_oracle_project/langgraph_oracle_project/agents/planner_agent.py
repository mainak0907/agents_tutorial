"""
agents/planner_agent.py
────────────────────────
Node 1 – Planner Agent
=======================
Responsibility:
  - Receive the user's natural-language query.
  - Use an LLM call to decide which sub-agents to invoke.
  - Populate `planned_tasks` so the graph router knows what to run next.

LangGraph concept demonstrated:
  - A node is just a Python async function that accepts the full state
    and returns a **partial** state dict (only the keys it changes).
  - The graph handles merging.
"""

import json
import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import MasterState
from db import OracleClient
from utils import get_llm, get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a planning assistant for an enterprise analytics pipeline.
Your job is to read the user's query and decide which analysis modules to activate.

Available modules:
  - hr_analysis      → headcount, role distribution, average salaries
  - finance_analysis → payroll totals, department spend, salary anomalies

Respond ONLY with valid JSON in this exact format:
{
  "planned_tasks": ["hr_analysis", "finance_analysis"],
  "reasoning": "Brief explanation of why you chose these tasks."
}

Include only the tasks that are relevant to the user's query.
"""


async def planner_node(state: MasterState, db: OracleClient) -> MasterState:
    """
    LangGraph node: analyse the user query and decide which agents to run.

    Parameters
    ----------
    state : MasterState
        The shared graph state.  We read `user_query` and write
        `run_id`, `planned_tasks`, `planner_reasoning`.
    db : OracleClient
        Injected Oracle client (used to log the run start).

    Returns
    -------
    Partial MasterState with keys this node owns.
    """
    user_query = state.get("user_query", "")
    run_id = str(uuid.uuid4())

    logger.info("[planner] run_id=%s  query=%r", run_id, user_query)

    # ── LLM call ──────────────────────────────────────────────
    llm = get_llm(temperature=0.1)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"User query: {user_query}"),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    # Parse JSON safely
    try:
        parsed = json.loads(raw)
        planned_tasks: list[str] = parsed.get("planned_tasks", ["hr_analysis", "finance_analysis"])
        reasoning: str = parsed.get("reasoning", "")
    except json.JSONDecodeError:
        logger.warning("[planner] LLM returned non-JSON; defaulting to all tasks.")
        planned_tasks = ["hr_analysis", "finance_analysis"]
        reasoning = raw

    logger.info("[planner] planned_tasks=%s", planned_tasks)

    # ── Audit log to Oracle ────────────────────────────────────
    await db.execute(
        """INSERT INTO agent_run_logs (run_id, agent_name, status, message)
           VALUES (:run_id, :agent, :status, :msg)""",
        {
            "run_id": run_id,
            "agent": "planner",
            "status": "complete",
            "msg": f"Planned tasks: {planned_tasks}. Reasoning: {reasoning}",
        },
    )

    # ── Return partial state ───────────────────────────────────
    # Only include keys this node is responsible for.
    return {
        "run_id": run_id,
        "planned_tasks": planned_tasks,
        "planner_reasoning": reasoning,
        "pipeline_status": "running",
    }

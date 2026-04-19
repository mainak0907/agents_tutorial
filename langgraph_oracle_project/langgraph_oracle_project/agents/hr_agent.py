"""
agents/hr_agent.py
───────────────────
Node 2a – HR Agent
==================
Responsibility:
  - Query Oracle for headcount, role breakdown, and average salary per department.
  - Feed those rows to an LLM to produce a human-readable HR narrative.
  - Write results back into the shared MasterState.

LangGraph concept demonstrated:
  - Nodes that make BOTH a database call AND an LLM call.
  - Error handling within a node: the node sets `hr_status = "error"` and
    continues rather than raising, so the graph can still run other nodes.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import MasterState
from db import OracleClient
from utils import get_llm, get_logger

logger = get_logger(__name__)

HR_SYSTEM_PROMPT = """You are an expert HR analyst.
You will be given a JSON dataset of employee information from an Oracle database.
Write a concise, professional HR analysis covering:
1. Total headcount per department
2. Role distribution
3. Average salary observations
4. Any notable patterns or concerns

Be analytical and specific. Use numbers from the data. Keep it under 300 words.
"""


async def hr_agent_node(state: MasterState, db: OracleClient) -> MasterState:
    """
    LangGraph node: fetch HR data from Oracle, analyse with LLM.

    Reads  : run_id, planned_tasks
    Writes : hr_raw_data, hr_analysis, hr_status, (hr_error on failure)
    """
    run_id = state.get("run_id", "unknown")
    logger.info("[hr_agent] Starting HR analysis for run_id=%s", run_id)

    # ── Log start ──────────────────────────────────────────────
    await db.execute(
        "INSERT INTO agent_run_logs (run_id, agent_name, status) VALUES (:r, :a, :s)",
        {"r": run_id, "a": "hr_agent", "s": "started"},
    )

    try:
        # ── 1. Fetch data from Oracle ──────────────────────────
        hr_data = await db.fetch_all(
            """
            SELECT
                e.emp_id,
                e.name,
                d.dept_name,
                e.role,
                e.salary,
                TO_CHAR(e.hire_date, 'YYYY-MM-DD') AS hire_date
            FROM employees e
            JOIN departments d ON e.dept_id = d.dept_id
            ORDER BY d.dept_name, e.salary DESC
            """
        )

        if not hr_data:
            logger.warning("[hr_agent] No employee data found in Oracle.")
            return {
                "hr_raw_data": [],
                "hr_analysis": "No employee data available.",
                "hr_status": "complete",
            }

        logger.info("[hr_agent] Fetched %d employee rows from Oracle.", len(hr_data))

        # ── 2. LLM analysis ────────────────────────────────────
        import json
        llm = get_llm(temperature=0.3)
        messages = [
            SystemMessage(content=HR_SYSTEM_PROMPT),
            HumanMessage(content=f"Employee data:\n{json.dumps(hr_data, indent=2)}"),
        ]
        response = llm.invoke(messages)
        analysis = response.content.strip()

        logger.info("[hr_agent] LLM analysis complete (%d chars).", len(analysis))

        # ── 3. Audit log ───────────────────────────────────────
        await db.execute(
            """UPDATE agent_run_logs
               SET status = 'complete', message = :msg
               WHERE run_id = :r AND agent_name = 'hr_agent'
                 AND status = 'started'""",
            {"msg": f"Analysed {len(hr_data)} employees.", "r": run_id},
        )

        return {
            "hr_raw_data": hr_data,
            "hr_analysis": analysis,
            "hr_status": "complete",
        }

    except Exception as exc:
        logger.exception("[hr_agent] Error during HR analysis: %s", exc)
        await db.execute(
            """UPDATE agent_run_logs
               SET status = 'error', message = :msg
               WHERE run_id = :r AND agent_name = 'hr_agent'""",
            {"msg": str(exc), "r": run_id},
        )
        return {
            "hr_raw_data": [],
            "hr_analysis": "",
            "hr_status": "error",
            "hr_error": str(exc),
        }

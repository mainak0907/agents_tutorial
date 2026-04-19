"""
agents/synthesiser_agent.py
────────────────────────────
Node 3 – Synthesiser Agent  (master-state aggregator)
======================================================
Responsibility:
  - This node runs AFTER all specialised agents have completed.
  - It reads the full MasterState (all hr_* and finance_* keys).
  - Makes one final LLM call to produce:
      • An executive summary combining both analyses.
      • A prioritised list of recommendations.
  - Persists the final report to the `analysis_reports` Oracle table.
  - Sets `pipeline_status = "complete"`.

LangGraph concept demonstrated:
  - A "fan-in" node: it waits for both parallel branches to finish.
  - Reading multiple sub-state slices and producing a synthesised output.
  - Writing the final state to a database (persistence layer).
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import MasterState
from db import OracleClient
from utils import get_llm, get_logger

logger = get_logger(__name__)

SYNTHESISER_SYSTEM_PROMPT = """You are the Chief Analytics Officer producing a final board-level report.
You have received two specialist analyses:
  1. HR Analysis  – headcount, roles, and salary observations
  2. Finance Analysis – payroll totals, anomalies, and financial patterns

Your task:
A) Write a 150-200 word EXECUTIVE SUMMARY that integrates both analyses into
   a coherent narrative.  Highlight the most important findings.
B) List 3-5 RECOMMENDATIONS (bullet points) that leadership should act on.

Respond ONLY with valid JSON in this exact format:
{
  "executive_summary": "...",
  "recommendations": ["recommendation 1", "recommendation 2", ...]
}
"""


async def synthesiser_node(state: MasterState, db: OracleClient) -> MasterState:
    """
    LangGraph node: combine all agent outputs into an executive report.

    Reads  : run_id, hr_analysis, finance_analysis, hr_status, finance_status
    Writes : executive_summary, recommendations, pipeline_status, persisted
    """
    run_id = state.get("run_id", "unknown")
    logger.info("[synthesiser] Starting synthesis for run_id=%s", run_id)

    await db.execute(
        "INSERT INTO agent_run_logs (run_id, agent_name, status) VALUES (:r, :a, :s)",
        {"r": run_id, "a": "synthesiser", "s": "started"},
    )

    hr_analysis = state.get("hr_analysis", "HR analysis not available.")
    finance_analysis = state.get("finance_analysis", "Finance analysis not available.")
    hr_status = state.get("hr_status", "unknown")
    finance_status = state.get("finance_status", "unknown")

    # ── 1. LLM synthesis call ──────────────────────────────────
    llm = get_llm(temperature=0.4)
    user_content = f"""
HR Analysis (status: {hr_status}):
{hr_analysis}

Finance Analysis (status: {finance_status}):
{finance_analysis}

Original user query: {state.get("user_query", "")}
Planner reasoning:   {state.get("planner_reasoning", "")}
"""

    messages = [
        SystemMessage(content=SYNTHESISER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Parse JSON response
    try:
        parsed = json.loads(raw)
        executive_summary: str = parsed.get("executive_summary", raw)
        recommendations: list[str] = parsed.get("recommendations", [])
    except json.JSONDecodeError:
        logger.warning("[synthesiser] Non-JSON response from LLM; using raw text.")
        executive_summary = raw
        recommendations = []

    logger.info(
        "[synthesiser] Summary: %d chars, Recommendations: %d items",
        len(executive_summary), len(recommendations),
    )

    # ── 2. Persist final report to Oracle ─────────────────────
    await db.execute(
        """
        INSERT INTO analysis_reports
            (run_id, executive_summary, hr_analysis, finance_analysis, recommendations)
        VALUES
            (:run_id, :exec_sum, :hr, :fin, :recs)
        """,
        {
            "run_id": run_id,
            "exec_sum": executive_summary,
            "hr": hr_analysis,
            "fin": finance_analysis,
            "recs": json.dumps(recommendations),
        },
    )

    logger.info("[synthesiser] Report persisted to analysis_reports table.")

    await db.execute(
        """UPDATE agent_run_logs
           SET status = 'complete', message = :msg
           WHERE run_id = :r AND agent_name = 'synthesiser'""",
        {"msg": "Executive report generated and persisted.", "r": run_id},
    )

    return {
        "executive_summary": executive_summary,
        "recommendations": recommendations,
        "pipeline_status": "complete",
        "persisted": True,
    }

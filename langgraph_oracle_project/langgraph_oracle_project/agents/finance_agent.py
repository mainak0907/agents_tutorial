"""
agents/finance_agent.py
────────────────────────
Node 2b – Finance Agent
=======================
Responsibility:
  - Query Oracle for payroll aggregates (total salary spend per department,
    min/max/avg salaries, outlier detection).
  - Use an LLM to produce a finance-focused narrative and flag anomalies.

LangGraph concept demonstrated:
  - How a node that runs IN PARALLEL with hr_agent still writes to
    a different slice of MasterState (finance_* keys vs hr_* keys).
  - Oracle aggregate SQL with GROUP BY / HAVING.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import MasterState
from db import OracleClient
from utils import get_llm, get_logger

logger = get_logger(__name__)

FINANCE_SYSTEM_PROMPT = """You are a senior financial analyst specialising in corporate payroll.
You will receive JSON payroll data from an Oracle database.
Write a concise finance analysis covering:
1. Total payroll by department
2. Highest / lowest earners and their roles
3. Salary spread (max - min) as a risk indicator
4. Any departments where average salary is significantly above or below company average
5. One or two actionable recommendations

Be data-driven, specific, and professional. Keep it under 300 words.
"""

# Salary more than 2x company average is flagged as an outlier
OUTLIER_MULTIPLIER = 2.0


async def finance_agent_node(state: MasterState, db: OracleClient) -> MasterState:
    """
    LangGraph node: fetch payroll data from Oracle, analyse with LLM.

    Reads  : run_id
    Writes : finance_raw_data, finance_analysis, finance_status, (finance_error)
    """
    run_id = state.get("run_id", "unknown")
    logger.info("[finance_agent] Starting finance analysis for run_id=%s", run_id)

    await db.execute(
        "INSERT INTO agent_run_logs (run_id, agent_name, status) VALUES (:r, :a, :s)",
        {"r": run_id, "a": "finance_agent", "s": "started"},
    )

    try:
        # ── 1. Department payroll summary ──────────────────────
        dept_payroll = await db.fetch_all(
            """
            SELECT
                d.dept_name,
                COUNT(e.emp_id)       AS headcount,
                SUM(e.salary)         AS total_payroll,
                ROUND(AVG(e.salary))  AS avg_salary,
                MIN(e.salary)         AS min_salary,
                MAX(e.salary)         AS max_salary
            FROM employees e
            JOIN departments d ON e.dept_id = d.dept_id
            GROUP BY d.dept_name
            ORDER BY total_payroll DESC
            """
        )

        # ── 2. Company-wide average for outlier detection ──────
        company_avg_row = await db.fetch_one(
            "SELECT ROUND(AVG(salary)) AS company_avg FROM employees"
        )
        company_avg: float = float(company_avg_row["company_avg"]) if company_avg_row else 0.0

        # ── 3. Outlier employees ───────────────────────────────
        outliers = await db.fetch_all(
            """
            SELECT e.name, d.dept_name, e.role, e.salary
            FROM employees e
            JOIN departments d ON e.dept_id = d.dept_id
            WHERE e.salary > :threshold
            ORDER BY e.salary DESC
            """,
            {"threshold": company_avg * OUTLIER_MULTIPLIER},
        )

        logger.info(
            "[finance_agent] dept_payroll=%d rows, company_avg=%.0f, outliers=%d",
            len(dept_payroll), company_avg, len(outliers),
        )

        finance_data = {
            "company_avg_salary": company_avg,
            "department_payroll": dept_payroll,
            "salary_outliers": outliers,
        }

        # ── 4. LLM analysis ────────────────────────────────────
        llm = get_llm(temperature=0.3)
        messages = [
            SystemMessage(content=FINANCE_SYSTEM_PROMPT),
            HumanMessage(content=f"Payroll data:\n{json.dumps(finance_data, indent=2)}"),
        ]
        response = llm.invoke(messages)
        analysis = response.content.strip()

        logger.info("[finance_agent] LLM analysis complete (%d chars).", len(analysis))

        await db.execute(
            """UPDATE agent_run_logs
               SET status = 'complete', message = :msg
               WHERE run_id = :r AND agent_name = 'finance_agent'""",
            {"msg": f"Payroll for {len(dept_payroll)} depts. Outliers: {len(outliers)}.", "r": run_id},
        )

        return {
            "finance_raw_data": [finance_data],   # wrap in list to match TypedDict type
            "finance_analysis": analysis,
            "finance_status": "complete",
        }

    except Exception as exc:
        logger.exception("[finance_agent] Error: %s", exc)
        await db.execute(
            """UPDATE agent_run_logs
               SET status = 'error', message = :msg
               WHERE run_id = :r AND agent_name = 'finance_agent'""",
            {"msg": str(exc), "r": run_id},
        )
        return {
            "finance_raw_data": [],
            "finance_analysis": "",
            "finance_status": "error",
            "finance_error": str(exc),
        }

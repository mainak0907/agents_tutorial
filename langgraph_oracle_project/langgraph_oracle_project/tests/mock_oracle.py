"""
tests/mock_oracle.py
─────────────────────
A mock OracleClient that returns hard-coded data so you can run
the full LangGraph pipeline WITHOUT a real Oracle database.

Usage:
    python tests/mock_oracle.py
"""

from typing import Any
from db import OracleClient


MOCK_EMPLOYEES = [
    {"emp_id": 1, "name": "Arjun Mehta",   "dept_name": "Engineering", "role": "Senior Engineer", "salary": 120000, "hire_date": "2021-03-15"},
    {"emp_id": 2, "name": "Priya Sharma",  "dept_name": "Engineering", "role": "Engineer",        "salary": 90000,  "hire_date": "2022-07-01"},
    {"emp_id": 6, "name": "Ananya Rao",    "dept_name": "Engineering", "role": "Tech Lead",       "salary": 140000, "hire_date": "2020-01-10"},
    {"emp_id": 3, "name": "Rahul Gupta",   "dept_name": "Finance",     "role": "Finance Analyst", "salary": 85000,  "hire_date": "2021-11-20"},
    {"emp_id": 4, "name": "Sneha Iyer",    "dept_name": "Finance",     "role": "Finance Manager", "salary": 110000, "hire_date": "2019-06-05"},
    {"emp_id": 8, "name": "Kavya Reddy",   "dept_name": "Finance",     "role": "Junior Analyst",  "salary": 65000,  "hire_date": "2023-02-14"},
    {"emp_id": 5, "name": "Vikram Nair",   "dept_name": "HR",          "role": "HR Specialist",   "salary": 75000,  "hire_date": "2022-03-28"},
    {"emp_id": 7, "name": "Deepak Joshi",  "dept_name": "HR",          "role": "HR Manager",      "salary": 95000,  "hire_date": "2020-09-17"},
]

MOCK_DEPT_PAYROLL = [
    {"dept_name": "Engineering", "headcount": 3, "total_payroll": 350000, "avg_salary": 116667, "min_salary": 90000,  "max_salary": 140000},
    {"dept_name": "Finance",     "headcount": 3, "total_payroll": 260000, "avg_salary": 86667,  "min_salary": 65000,  "max_salary": 110000},
    {"dept_name": "HR",          "headcount": 2, "total_payroll": 170000, "avg_salary": 85000,  "min_salary": 75000,  "max_salary": 95000},
]

MOCK_COMPANY_AVG = {"company_avg": 97500}


class MockOracleClient(OracleClient):
    """
    Drop-in replacement for OracleClient that never touches a real database.
    Overrides fetch_all, fetch_one, and execute to return static data.
    """

    def init_pool(self) -> None:
        print("[MockOracle] Pool initialised (no real DB).")

    def close_pool(self) -> None:
        print("[MockOracle] Pool closed.")

    async def fetch_all(self, sql: str, params: dict | None = None) -> list[dict[str, Any]]:
        sql_upper = sql.upper()
        if "HIRE_DATE" in sql_upper:
            # HR employee query
            return MOCK_EMPLOYEES
        if "TOTAL_PAYROLL" in sql_upper:
            # Finance dept payroll query
            return MOCK_DEPT_PAYROLL
        if "THRESHOLD" in sql_upper:
            # Outlier query (anyone > 2x avg = 195000; none in mock)
            return []
        return []

    async def fetch_one(self, sql: str, params: dict | None = None) -> dict[str, Any] | None:
        if "AVG(SALARY)" in sql.upper() or "AVG(salary)" in sql:
            return MOCK_COMPANY_AVG
        return None

    async def execute(self, sql: str, params: dict | None = None, commit: bool = True) -> int:
        # Silently absorb all DML (INSERT/UPDATE for audit logs)
        return 1

    async def execute_many(self, sql: str, params_list: list, commit: bool = True) -> None:
        pass


# ──────────────────────────────────────────────────────────────
# Quick runner using the mock client
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import json
    from tools.pipeline import build_pipeline
    from agents.state import MasterState

    async def run_mock():
        db = MockOracleClient()
        db.init_pool()

        pipeline = build_pipeline(db)
        initial: MasterState = {
            "user_query": "Give me a full HR and finance analysis of our workforce."
        }

        print("\n🚀 Running pipeline with MockOracleClient...\n")
        final: MasterState = await pipeline.ainvoke(initial)

        print("\n══════════════════════════════════════════════════════")
        print("EXECUTIVE SUMMARY")
        print("══════════════════════════════════════════════════════")
        print(final.get("executive_summary", ""))

        print("\nRECOMMENDATIONS")
        for i, r in enumerate(final.get("recommendations", []), 1):
            print(f"  {i}. {r}")

        print("\nPIPELINE STATUS:", final.get("pipeline_status"))
        db.close_pool()

    asyncio.run(run_mock())

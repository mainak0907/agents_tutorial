"""
db/schema.py
─────────────
DDL helpers to create the tables this demo project uses.

Run once:
    python -m db.schema

Tables created:
  - employees          (source data)
  - departments        (source data)
  - agent_run_logs     (audit trail written by the LangGraph agents)
  - analysis_reports   (final master-state outputs persisted after each run)
"""

import logging

import oracledb

from config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# DDL statements
# ──────────────────────────────────────────────────────────────

DDL_STATEMENTS = [
    # departments
    """
    CREATE TABLE departments (
        dept_id   NUMBER        PRIMARY KEY,
        dept_name VARCHAR2(100) NOT NULL,
        location  VARCHAR2(100)
    )
    """,
    # employees
    """
    CREATE TABLE employees (
        emp_id    NUMBER         PRIMARY KEY,
        name      VARCHAR2(100)  NOT NULL,
        dept_id   NUMBER         REFERENCES departments(dept_id),
        role      VARCHAR2(100),
        salary    NUMBER(10, 2),
        hire_date DATE           DEFAULT SYSDATE
    )
    """,
    # agent run logs (one row per agent invocation)
    """
    CREATE TABLE agent_run_logs (
        log_id     NUMBER         GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        run_id     VARCHAR2(64)   NOT NULL,
        agent_name VARCHAR2(100)  NOT NULL,
        status     VARCHAR2(20)   DEFAULT 'started',
        message    CLOB,
        created_at TIMESTAMP      DEFAULT SYSTIMESTAMP
    )
    """,
    # final analysis reports (master state persisted)
    """
    CREATE TABLE analysis_reports (
        report_id          NUMBER        GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        run_id             VARCHAR2(64)  NOT NULL,
        executive_summary  CLOB,
        hr_analysis        CLOB,
        finance_analysis   CLOB,
        recommendations    CLOB,
        created_at         TIMESTAMP     DEFAULT SYSTIMESTAMP
    )
    """,
]

SEED_DATA = [
    "INSERT INTO departments (dept_id, dept_name, location) VALUES (10, 'Engineering', 'Bengaluru')",
    "INSERT INTO departments (dept_id, dept_name, location) VALUES (20, 'Finance',     'Mumbai')",
    "INSERT INTO departments (dept_id, dept_name, location) VALUES (30, 'HR',          'Delhi')",

    "INSERT INTO employees (emp_id, name, dept_id, role, salary) VALUES (1, 'Arjun Mehta',   10, 'Senior Engineer', 120000)",
    "INSERT INTO employees (emp_id, name, dept_id, role, salary) VALUES (2, 'Priya Sharma',  10, 'Engineer',        90000)",
    "INSERT INTO employees (emp_id, name, dept_id, role, salary) VALUES (3, 'Rahul Gupta',   20, 'Finance Analyst', 85000)",
    "INSERT INTO employees (emp_id, name, dept_id, role, salary) VALUES (4, 'Sneha Iyer',    20, 'Finance Manager', 110000)",
    "INSERT INTO employees (emp_id, name, dept_id, role, salary) VALUES (5, 'Vikram Nair',   30, 'HR Specialist',   75000)",
    "INSERT INTO employees (emp_id, name, dept_id, role, salary) VALUES (6, 'Ananya Rao',    10, 'Tech Lead',       140000)",
    "INSERT INTO employees (emp_id, name, dept_id, role, salary) VALUES (7, 'Deepak Joshi',  30, 'HR Manager',      95000)",
    "INSERT INTO employees (emp_id, name, dept_id, role, salary) VALUES (8, 'Kavya Reddy',   20, 'Junior Analyst',  65000)",
]


def _table_exists(conn: oracledb.Connection, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM user_tables WHERE table_name = :t",
            {"t": table_name.upper()},
        )
        return cur.fetchone()[0] > 0  # type: ignore[index]


def create_schema(seed: bool = True) -> None:
    """Connect, create tables if they don't exist, optionally seed data."""
    conn = oracledb.connect(
        user=settings.ORACLE_USER,
        password=settings.ORACLE_PASSWORD,
        dsn=settings.ORACLE_DSN,
    )
    try:
        for ddl in DDL_STATEMENTS:
            table_name = ddl.strip().split()[2]  # "CREATE TABLE <name>"
            if _table_exists(conn, table_name):
                logger.info("Table %s already exists – skipping.", table_name)
                continue
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
            logger.info("Created table: %s", table_name)

        if seed:
            # Only seed if employees table is empty
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM employees")
                count = cur.fetchone()[0]  # type: ignore[index]
            if count == 0:
                with conn.cursor() as cur:
                    for stmt in SEED_DATA:
                        cur.execute(stmt)
                conn.commit()
                logger.info("Seed data inserted.")
            else:
                logger.info("Seed data already present – skipping.")
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_schema()
    print("Schema setup complete.")

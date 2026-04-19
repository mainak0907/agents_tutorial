"""
db/oracle_client.py
────────────────────
Manages a connection pool to Oracle DB using the oracledb (python-oracledb) driver.

Key concepts:
  - oracledb.create_pool()  → creates a session pool (reuse connections efficiently)
  - pool.acquire()          → checkout a connection from the pool
  - conn.cursor()           → create a cursor to execute SQL
  - Always release/close connections back to the pool after use

Usage:
    from db.oracle_client import OracleClient

    client = OracleClient()
    await client.init_pool()

    rows = await client.fetch_all("SELECT * FROM employees WHERE dept_id = :dept", {"dept": 10})
    await client.execute("INSERT INTO logs (msg) VALUES (:msg)", {"msg": "hello"})

    await client.close_pool()
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

import oracledb

from config import settings

logger = logging.getLogger(__name__)


class OracleClient:
    """
    Thin async-friendly wrapper around python-oracledb's synchronous pool.

    Why synchronous?
        oracledb's async support requires the C extension ("thick mode").
        In many CI/cloud environments the Oracle Instant Client isn't present,
        so we stay in "thin mode" (pure Python) and wrap blocking calls in a
        thread executor when needed.  For a learning project this sync approach
        is the safest and most portable choice.
    """

    def __init__(self) -> None:
        self._pool: oracledb.ConnectionPool | None = None

    # ──────────────────────────────────────────────
    # Pool lifecycle
    # ──────────────────────────────────────────────

    def init_pool(self) -> None:
        """Create the connection pool.  Call once at application startup."""
        pool_params: dict[str, Any] = {
            "user": settings.ORACLE_USER,
            "password": settings.ORACLE_PASSWORD,
            "dsn": settings.ORACLE_DSN,
            "min": settings.POOL_MIN,
            "max": settings.POOL_MAX,
            "increment": settings.POOL_INCREMENT,
        }

        # Optional mTLS / wallet support (Oracle Autonomous DB / TLS)
        if settings.ORACLE_WALLET_LOCATION:
            pool_params["wallet_location"] = settings.ORACLE_WALLET_LOCATION
            pool_params["wallet_password"] = settings.ORACLE_WALLET_PASSWORD

        logger.info("Initialising Oracle connection pool → %s", settings.ORACLE_DSN)
        self._pool = oracledb.create_pool(**pool_params)
        logger.info("Oracle pool ready (min=%d, max=%d)", settings.POOL_MIN, settings.POOL_MAX)

    def close_pool(self) -> None:
        """Drain and close the pool.  Call at application shutdown."""
        if self._pool:
            self._pool.close()
            logger.info("Oracle connection pool closed.")

    # ──────────────────────────────────────────────
    # Context manager: auto-return connection to pool
    # ──────────────────────────────────────────────

    @asynccontextmanager
    async def _connection(self):
        """Yield a pooled connection; always return it to the pool."""
        if self._pool is None:
            raise RuntimeError("Call init_pool() before executing queries.")
        conn = self._pool.acquire()
        try:
            yield conn
        finally:
            self._pool.release(conn)

    # ──────────────────────────────────────────────
    # Query helpers
    # ──────────────────────────────────────────────

    async def fetch_all(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a SELECT and return all rows as a list of dicts.

        Example:
            rows = await client.fetch_all(
                "SELECT id, name, salary FROM employees WHERE dept_id = :dept",
                {"dept": 10},
            )
        """
        async with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or {})
                # Build column-name → value dicts from the cursor description
                columns = [col[0].lower() for col in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    async def fetch_one(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Execute a SELECT and return the first row as a dict, or None."""
        async with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or {})
                columns = [col[0].lower() for col in cur.description]
                row = cur.fetchone()
                return dict(zip(columns, row)) if row else None

    async def execute(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> int:
        """
        Execute a DML statement (INSERT / UPDATE / DELETE).
        Returns the number of rows affected.

        Example:
            affected = await client.execute(
                "UPDATE employees SET salary = :sal WHERE id = :id",
                {"sal": 75000, "id": 42},
            )
        """
        async with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or {})
                if commit:
                    conn.commit()
                return cur.rowcount

    async def execute_many(
        self,
        sql: str,
        params_list: list[dict[str, Any]],
        commit: bool = True,
    ) -> None:
        """
        Batch-execute a DML statement.

        Example:
            await client.execute_many(
                "INSERT INTO logs (agent_name, message, created_at) VALUES (:agent, :msg, SYSDATE)",
                [{"agent": "planner", "msg": "started"}, {"agent": "analyst", "msg": "done"}],
            )
        """
        async with self._connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, params_list)
                if commit:
                    conn.commit()

    async def call_procedure(
        self,
        proc_name: str,
        params: list[Any] | None = None,
    ) -> list[Any]:
        """
        Call a stored procedure and return output bind variables.

        Example:
            out = await client.call_procedure("pkg_reports.calc_bonus", [10, cursor_var])
        """
        async with self._connection() as conn:
            with conn.cursor() as cur:
                out_params = params or []
                cur.callproc(proc_name, out_params)
                conn.commit()
                return out_params

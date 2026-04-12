# Oracle SQL AI Agent

A conversational AI agent that translates natural language into Oracle SQL queries, built with **LangChain**, **LangGraph**, and **ChatOpenAI**.

---

## Project Structure

```
oracle_sql_agent/
├── main.py          # Entry point — interactive REPL loop
├── agent.py         # Agent creation using create_agent
├── tools.py         # LangChain tools bound to the agent
├── db.py            # Oracle DB connection manager
├── requirements.txt # Python dependencies
└── .env             # Credentials (not committed — see setup below)
```

---

## Architecture

```
User (natural language)
        │
        ▼
LangChain Agent  (create_agent → CompiledStateGraph)
  ├── ChatOpenAI (your model, base_url, api_key)
  └── Tool-calling loop
        ├── run_sql_query     — execute SELECT queries
        ├── list_tables       — list all accessible tables
        ├── describe_table    — get columns, types, nullability
        ├── validate_query    — block non-SELECT / dangerous SQL
        └── get_row_count     — count rows in a table
        │
        ▼
Oracle Database  (oracledb — thin mode, no Oracle Client needed)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create your `.env` file

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://your-custom-base-url/v1
OPENAI_MODEL=gpt-4o

ORACLE_USER=your_db_user
ORACLE_PASSWORD=your_db_password
ORACLE_DSN=your_host:1521/your_service_name
```

### 3. Run the agent

```bash
python main.py
```

---

## Usage

```
============================================================
  Oracle SQL AI Agent — powered by LangChain + ChatOpenAI
  Type your question in plain English. Type 'exit' to quit.
============================================================

You: Show me the top 5 customers by total order value
You: How many orders were placed last month?
You: What columns does the EMPLOYEES table have?
You: exit
```

---

## File Reference

### `main.py`

Entry point. Starts an interactive REPL loop, invokes the agent graph for each user message, and prints the final AI response.

```python
from agent import build_agent


def main():
    print("=" * 60)
    print("  Oracle SQL AI Agent — powered by LangChain + ChatOpenAI")
    print("  Type your question in plain English. Type 'exit' to quit.")
    print("=" * 60)

    graph = build_agent()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            # create_agent returns a CompiledStateGraph.
            # Input must be {"messages": [...]}, output is the full messages list.
            result = graph.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            # The last message in the list is the final AI response
            final_message = result["messages"][-1]
            print(f"\nAgent: {final_message.content}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
```

---

### `agent.py`

Builds the LangChain agent using `create_agent`. Returns a `CompiledStateGraph` that manages the tool-calling loop internally — no `AgentExecutor` or Hub prompt needed.

```python
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools import ALL_TOOLS

load_dotenv()


SYSTEM_PROMPT = """You are an expert Oracle SQL assistant. Your job is to help users
query an Oracle database by translating their natural language questions into correct
Oracle SQL SELECT statements and returning clear, readable answers.

Guidelines:
- Always use list_tables first if you are unsure which tables exist.
- Use describe_table to understand a table's columns before writing a query.
- Use validate_query before run_sql_query when constructing complex queries.
- Only run SELECT queries — never modify data.
- If a user's question is ambiguous, ask for clarification before querying.
- Present results in a clean, easy-to-read format.
- Explain what query you ran and why.
"""


def build_agent():
    """
    Build and return a compiled LangChain agent graph using create_agent.

    create_agent returns a CompiledStateGraph that runs the tool-calling
    loop internally — no AgentExecutor or Hub prompt needed.
    """
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )

    # create_agent wires the LLM, tools, and system prompt into a
    # compiled LangGraph state machine that loops until no tool calls remain.
    graph = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        debug=True,   # prints per-node execution trace (equivalent to verbose=True)
    )

    return graph
```

---

### `tools.py`

Five `@tool`-decorated functions that the agent can call. All tools are read-only — `validate_query` ensures no mutating SQL ever reaches the database.

```python
import re
from typing import Any

from langchain_core.tools import tool

from db import get_connection


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _execute_query(sql: str, params: dict | None = None) -> list[dict[str, Any]]:
    """Execute a SQL statement and return rows as a list of dicts."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params or {})
        columns = [col[0].lower() for col in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return rows
    finally:
        conn.close()


# ──────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────

@tool
def run_sql_query(query: str) -> str:
    """
    Execute a read-only SQL SELECT query against the Oracle database and return
    the results as a formatted string. Only SELECT statements are allowed.

    Args:
        query: A valid Oracle SQL SELECT statement.

    Returns:
        Query results as a readable string, or an error message.
    """
    validation_result = validate_query(query)
    if validation_result != "valid":
        return f"Query blocked: {validation_result}"

    try:
        rows = _execute_query(query)
        if not rows:
            return "Query executed successfully. No rows returned."
        # Format as readable table-like text
        headers = list(rows[0].keys())
        lines = [" | ".join(str(row.get(h, "")) for h in headers) for row in rows]
        header_line = " | ".join(headers)
        separator = "-" * len(header_line)
        result = "\n".join([header_line, separator] + lines)
        return f"Results ({len(rows)} rows):\n{result}"
    except Exception as e:
        return f"Error executing query: {str(e)}"


@tool
def list_tables() -> str:
    """
    List all accessible tables in the Oracle database for the current user.

    Returns:
        A newline-separated list of table names.
    """
    try:
        rows = _execute_query(
            "SELECT table_name FROM user_tables ORDER BY table_name"
        )
        if not rows:
            return "No tables found for the current user."
        tables = [row["table_name"] for row in rows]
        return "Available tables:\n" + "\n".join(f"  - {t}" for t in tables)
    except Exception as e:
        return f"Error listing tables: {str(e)}"


@tool
def describe_table(table_name: str) -> str:
    """
    Return the column names, data types, and nullability for a given Oracle table.

    Args:
        table_name: The name of the table to describe (case-insensitive).

    Returns:
        A formatted description of the table's columns.
    """
    try:
        rows = _execute_query(
            """
            SELECT column_name, data_type, data_length, nullable
            FROM   user_tab_columns
            WHERE  UPPER(table_name) = UPPER(:tname)
            ORDER  BY column_id
            """,
            {"tname": table_name.upper()},
        )
        if not rows:
            return f"Table '{table_name}' not found or has no columns."
        lines = [f"  {r['column_name']} | {r['data_type']}({r['data_length']}) | nullable={r['nullable']}" for r in rows]
        return f"Schema for {table_name.upper()}:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error describing table: {str(e)}"


@tool
def validate_query(query: str) -> str:
    """
    Check that a SQL query is safe to execute — only SELECT statements allowed.
    Blocks INSERT, UPDATE, DELETE, DROP, TRUNCATE, and similar mutating operations.

    Args:
        query: The SQL query string to validate.

    Returns:
        'valid' if the query is safe, otherwise a rejection reason string.
    """
    stripped = query.strip().upper()

    # Must start with SELECT
    if not stripped.startswith("SELECT"):
        return "Only SELECT queries are permitted."

    # Block dangerous keywords anywhere in the query
    forbidden = [
        r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bDROP\b",
        r"\bTRUNCATE\b", r"\bALTER\b", r"\bCREATE\b", r"\bGRANT\b",
        r"\bREVOKE\b", r"\bEXEC\b", r"\bEXECUTE\b",
    ]
    for pattern in forbidden:
        if re.search(pattern, stripped):
            keyword = pattern.replace(r"\b", "")
            return f"Forbidden keyword detected: {keyword}"

    return "valid"


@tool
def get_row_count(table_name: str) -> str:
    """
    Return the total number of rows in a given Oracle table.

    Args:
        table_name: The name of the table to count rows in.

    Returns:
        A string stating the row count, or an error message.
    """
    try:
        rows = _execute_query(
            f"SELECT COUNT(*) AS row_count FROM {table_name}"  # noqa: S608
        )
        count = rows[0]["row_count"] if rows else 0
        return f"Table '{table_name}' has {count:,} rows."
    except Exception as e:
        return f"Error counting rows: {str(e)}"


# ──────────────────────────────────────────────
# Export list for easy import in agent.py
# ──────────────────────────────────────────────

ALL_TOOLS = [
    run_sql_query,
    list_tables,
    describe_table,
    validate_query,
    get_row_count,
]
```

---

### `db.py`

Oracle connection manager using the `oracledb` driver. Runs in **thin mode** by default — no Oracle Client installation required.

```python
import os

import oracledb
from dotenv import load_dotenv

load_dotenv()


def get_connection() -> oracledb.Connection:
    """Create and return an Oracle DB connection using env credentials."""
    user = os.getenv("ORACLE_USER")
    password = os.getenv("ORACLE_PASSWORD")
    dsn = os.getenv("ORACLE_DSN")

    if not all([user, password, dsn]):
        raise ValueError(
            "Missing Oracle credentials. Set ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN in .env"
        )

    connection = oracledb.connect(user=user, password=password, dsn=dsn)
    return connection
```

> **Thick mode:** If you need Oracle advanced features (Advanced Queuing, Sharding, etc.), add `oracledb.init_oracle_client()` before the `connect()` call.

---

### `requirements.txt`

```
langchain>=1.0
langchain-openai
langgraph
oracledb
python-dotenv
```

---

## Notes

- **Read-only by design** — `validate_query` blocks any non-SELECT SQL before it reaches the database.
- **No Oracle Client needed** — `oracledb` thin mode works out of the box.
- **Extend the system prompt** in `agent.py` with your specific table names or domain context to improve query accuracy.
- **Add tools** by defining new `@tool` functions in `tools.py` and appending them to `ALL_TOOLS`.

# LangGraph + Oracle DB — Multi-Agent Analytics Pipeline

A hands-on learning project that wires together **LangGraph**, **Anthropic Claude**, and **Oracle Database** (`python-oracledb`) into a realistic multi-agent pipeline.

---

## What You Will Learn

| Concept | Where it appears |
|---|---|
| LangGraph `StateGraph` | `tools/pipeline.py` |
| Typed sub-states + master state | `agents/state.py` |
| Parallel fan-out with `Send` | `tools/pipeline.py → route_to_agents` |
| Fan-in (multiple branches → one node) | synthesiser edges |
| Conditional routing | `add_conditional_edges` |
| LLM calls inside nodes | every `agents/*.py` |
| Oracle connection pool | `db/oracle_client.py` |
| Async-safe DB wrapper | `db/oracle_client.py` |
| Audit logging to Oracle | every agent node |
| Report persistence to Oracle | `agents/synthesiser_agent.py` |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   LangGraph Pipeline                    │
│                                                         │
│   ┌─────────────┐                                       │
│   │  planner    │  ← LLM decides which agents to run   │
│   └──────┬──────┘                                       │
│          │  conditional fan-out (Send)                  │
│    ┌─────┴──────┐                                       │
│    ▼            ▼                                       │
│ ┌──────┐   ┌─────────┐   ← run in parallel             │
│ │  HR  │   │ Finance │   ← each queries Oracle + LLM   │
│ └──┬───┘   └────┬────┘                                  │
│    └──────┬─────┘  fan-in                               │
│           ▼                                             │
│   ┌──────────────┐                                      │
│   │ Synthesiser  │  ← LLM combines, persists to Oracle │
│   └──────────────┘                                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Executive Report (stdout + Oracle analysis_reports table)
```

---

## Project Structure

```
langgraph_oracle_project/
│
├── main.py                      # Entry point
├── requirements.txt
├── .env.example                 # Copy to .env and fill in
│
├── config/
│   ├── __init__.py
│   └── settings.py              # All config from .env
│
├── db/
│   ├── __init__.py
│   ├── oracle_client.py         # Connection pool + query helpers
│   └── schema.py                # DDL + seed data (run once)
│
├── agents/
│   ├── __init__.py
│   ├── state.py                 # ★ All TypedDict state definitions
│   ├── planner_agent.py         # Node 1: decomposes the user query
│   ├── hr_agent.py              # Node 2a: HR data + LLM analysis
│   ├── finance_agent.py         # Node 2b: payroll data + LLM analysis
│   └── synthesiser_agent.py     # Node 3: master-state aggregator
│
├── tools/
│   ├── __init__.py
│   └── pipeline.py              # ★ StateGraph wiring + conditional routing
│
├── utils/
│   ├── __init__.py
│   ├── llm.py                   # ChatAnthropic factory
│   └── logger.py                # Consistent logging
│
└── tests/
    ├── __init__.py
    └── mock_oracle.py           # Run the full pipeline WITHOUT a real DB
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd langgraph_oracle_project
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Oracle credentials and Anthropic API key
```

### 3. Create Oracle schema (first time only)

```bash
python main.py --setup-schema
```

This creates four tables and seeds eight demo employees across Engineering, Finance, and HR departments.

### 4. Run the pipeline

```bash
# Default query (full analysis)
python main.py

# Custom query
python main.py --query "How is our Engineering headcount vs Finance?"

# Print full final state for debugging
python main.py --pretty
```

### 5. Try without Oracle (mock mode)

No Oracle database? No problem:

```bash
python tests/mock_oracle.py
```

This uses `MockOracleClient` — a drop-in replacement that returns hard-coded data — so you can observe the full LangGraph flow with real LLM calls.

---

## Oracle Tables

| Table | Purpose |
|---|---|
| `departments` | 3 demo departments (Engineering, Finance, HR) |
| `employees` | 8 demo employees with salaries and roles |
| `agent_run_logs` | One row per agent invocation — audit trail |
| `analysis_reports` | Final synthesised report persisted after each run |

Query your results after a run:

```sql
-- See which agents ran
SELECT run_id, agent_name, status, created_at
FROM agent_run_logs
ORDER BY created_at DESC;

-- Read the latest executive report
SELECT executive_summary, recommendations
FROM analysis_reports
ORDER BY created_at DESC
FETCH FIRST 1 ROW ONLY;
```

---

## LangGraph Concepts Explained

### State

```python
# agents/state.py

class MasterState(PlannerState, HRAgentState, FinanceAgentState, total=False):
    recommendations: list[str]
    executive_summary: str
    pipeline_status: str
    persisted: bool
```

`MasterState` inherits from three sub-states. Every node receives the whole `MasterState` but **only returns the keys it owns**. LangGraph merges all partial returns automatically.

### Nodes

Every node is a plain async function:

```python
async def hr_agent_node(state: MasterState, db: OracleClient) -> MasterState:
    rows = await db.fetch_all("SELECT ... FROM employees ...")
    analysis = llm.invoke([...])          # LLM call
    return {                               # partial state — only HR keys
        "hr_raw_data": rows,
        "hr_analysis": analysis.content,
        "hr_status": "complete",
    }
```

### Parallel Fan-out

```python
# tools/pipeline.py

def route_to_agents(state: MasterState) -> list[Send]:
    sends = []
    if "hr_analysis" in state["planned_tasks"]:
        sends.append(Send("hr_agent", state))
    if "finance_analysis" in state["planned_tasks"]:
        sends.append(Send("finance_agent", state))
    return sends                           # LangGraph runs these in parallel
```

`Send(node_name, state)` creates a parallel branch. Returning a list means all of them fire concurrently. Both branches write to different keys of `MasterState`, so there are no conflicts.

### Fan-in

```python
builder.add_edge("hr_agent",      "synthesiser")
builder.add_edge("finance_agent", "synthesiser")
```

LangGraph waits for **both** `hr_agent` and `finance_agent` to complete before calling `synthesiser`. No extra synchronisation code needed.

### Dependency Injection

Nodes need the Oracle client, but LangGraph node signatures must be `(state) -> state`. We use `functools.partial` to bind `db` at graph construction time:

```python
# tools/pipeline.py
hr = functools.partial(hr_agent_node, db=db)
builder.add_node("hr_agent", hr)
```

---

## Configuration Reference

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Your Anthropic API key |
| `ORACLE_USER` | ✅ | Oracle schema username |
| `ORACLE_PASSWORD` | ✅ | Oracle password |
| `ORACLE_DSN` | ✅ | `host:port/service` e.g. `localhost:1521/XEPDB1` |
| `ORACLE_WALLET_LOCATION` | ❌ | Path to wallet dir (Oracle Autonomous DB / TLS) |
| `ORACLE_WALLET_PASSWORD` | ❌ | Wallet password |
| `LOG_LEVEL` | ❌ | `DEBUG`, `INFO` (default), `WARNING` |

---

## Extending the Project

### Add a new agent

1. Add new keys to `MasterState` in `agents/state.py`.
2. Create `agents/my_agent.py` with an async node function.
3. Register in `tools/pipeline.py`:
   ```python
   builder.add_node("my_agent", functools.partial(my_agent_node, db=db))
   ```
4. Add it to the router's `Send` list.
5. Add an edge to `synthesiser`.

### Add Oracle stored procedure calls

```python
# db/oracle_client.py already has call_procedure()
result = await db.call_procedure("pkg_hr.get_attrition_risk", [dept_id])
```

### Add LangGraph checkpointing (resumable runs)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
pipeline = builder.compile(checkpointer=checkpointer)

# Resume a run by thread_id
config = {"configurable": {"thread_id": "run-123"}}
await pipeline.ainvoke(state, config=config)
```

---

## Requirements

- Python 3.11+
- Oracle Database (XE, Standard, or Autonomous) **or** use mock mode
- Anthropic API key

```
langgraph>=0.2.0
langchain>=0.3.0
langchain-anthropic>=0.2.0
oracledb>=2.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

---

## License

MIT

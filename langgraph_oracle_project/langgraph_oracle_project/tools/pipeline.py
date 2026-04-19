"""
tools/pipeline.py
──────────────────
LangGraph Pipeline Builder
===========================
This is the heart of the project. It wires all nodes into a directed graph.

Graph topology:
                         ┌─────────────────┐
                         │   START (entry)  │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  planner_node   │  LLM decides which tasks to run
                         └────────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │  (conditional fan-out)      │
                    ▼                             ▼
          ┌──────────────────┐        ┌────────────────────┐
          │   hr_agent_node  │        │ finance_agent_node │  (run in parallel via Send)
          └────────┬─────────┘        └──────────┬─────────┘
                    │                             │
                    └─────────────┬───────────────┘
                                  │  (fan-in)
                         ┌────────▼────────┐
                         │ synthesiser_node│  LLM combines all analyses
                         └────────┬────────┘
                                  │
                              ┌───▼───┐
                              │  END  │
                              └───────┘

Key LangGraph APIs used:
  - StateGraph(MasterState)     → create a graph typed to our state
  - graph.add_node(name, fn)    → register a node function
  - graph.add_edge(a, b)        → unconditional edge
  - graph.add_conditional_edges → route based on state values
  - Send(node, state)           → fan-out: dispatch parallel branches
  - graph.compile()             → compile to a runnable
"""

import functools
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agents.state import MasterState
from agents.planner_agent import planner_node
from agents.hr_agent import hr_agent_node
from agents.finance_agent import finance_agent_node
from agents.synthesiser_agent import synthesiser_node
from db import OracleClient
from utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Conditional router: fan-out based on planner's task list
# ──────────────────────────────────────────────────────────────

def route_to_agents(state: MasterState) -> list[Send]:
    """
    Called after planner_node.  Returns a list of `Send` objects – one
    per planned task.  LangGraph executes them in parallel.

    `Send(node_name, state_slice)` dispatches a branch to `node_name`
    with the given state.  Since we share one MasterState, we pass the
    whole state; each agent reads only its own keys.

    LangGraph concept:
        Returning a list of Send objects triggers PARALLEL execution.
        All branches share the same MasterState; their returned partial
        dicts are merged by LangGraph before the next node runs.
    """
    planned_tasks: list[str] = state.get("planned_tasks", [])
    logger.info("[router] Routing to tasks: %s", planned_tasks)

    sends: list[Send] = []
    if "hr_analysis" in planned_tasks:
        sends.append(Send("hr_agent", state))
    if "finance_analysis" in planned_tasks:
        sends.append(Send("finance_agent", state))

    if not sends:
        # Safety: if planner returned no tasks, run both
        logger.warning("[router] No tasks planned; defaulting to all agents.")
        sends = [Send("hr_agent", state), Send("finance_agent", state)]

    return sends


# ──────────────────────────────────────────────────────────────
# Pipeline factory
# ──────────────────────────────────────────────────────────────

def build_pipeline(db: OracleClient):
    """
    Build and compile the LangGraph pipeline.

    We use functools.partial to inject the `db` dependency into each
    node function.  LangGraph nodes must have the signature:
        async def node(state: MasterState) -> MasterState
    so we bind `db` at construction time.

    Returns
    -------
    CompiledStateGraph
        Call `.ainvoke(initial_state)` to run the pipeline.
    """
    # ── Bind db to each node ───────────────────────────────────
    planner   = functools.partial(planner_node,   db=db)
    hr        = functools.partial(hr_agent_node,  db=db)
    finance   = functools.partial(finance_agent_node, db=db)
    synthesiser = functools.partial(synthesiser_node, db=db)

    # ── Build graph ────────────────────────────────────────────
    builder = StateGraph(MasterState)

    # Register nodes
    builder.add_node("planner",     planner)
    builder.add_node("hr_agent",    hr)
    builder.add_node("finance_agent", finance)
    builder.add_node("synthesiser", synthesiser)

    # Edges
    builder.add_edge(START, "planner")

    # After planner: conditional fan-out via Send
    builder.add_conditional_edges(
        "planner",
        route_to_agents,        # returns list[Send]
        # We must enumerate all possible target nodes for LangGraph's type checker
        ["hr_agent", "finance_agent"],
    )

    # Both parallel branches converge at synthesiser (fan-in)
    builder.add_edge("hr_agent",      "synthesiser")
    builder.add_edge("finance_agent", "synthesiser")

    # Synthesiser → END
    builder.add_edge("synthesiser", END)

    # ── Compile ────────────────────────────────────────────────
    pipeline = builder.compile()
    logger.info("[pipeline] Graph compiled successfully.")
    return pipeline

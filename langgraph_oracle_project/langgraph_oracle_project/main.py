"""
main.py
────────
Application entry point.

Run:
    python main.py
    python main.py --query "Show me HR headcount only"
    python main.py --query "Analyse our payroll spend" --pretty

What happens:
  1. Oracle connection pool is initialised.
  2. LangGraph pipeline is compiled (planner → [hr, finance] → synthesiser).
  3. The pipeline is invoked with the user query as initial state.
  4. Results are printed to the terminal.
  5. The pool is cleanly shut down.
"""

import argparse
import asyncio
import json
import sys

from db import OracleClient
from db.schema import create_schema
from tools.pipeline import build_pipeline
from agents.state import MasterState
from utils import get_logger

logger = get_logger(__name__)

DEFAULT_QUERY = (
    "Give me a complete analysis of our workforce: "
    "headcount by department, role breakdown, salary insights, "
    "and overall payroll health."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangGraph + Oracle DB multi-agent analytics pipeline"
    )
    parser.add_argument(
        "--query", "-q",
        default=DEFAULT_QUERY,
        help="Natural-language question to analyse (default: full workforce analysis)",
    )
    parser.add_argument(
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print the full final state JSON at the end",
    )
    parser.add_argument(
        "--setup-schema",
        action="store_true",
        help="Create Oracle tables and seed demo data, then exit",
    )
    return parser.parse_args()


def print_banner(title: str) -> None:
    width = 70
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_final_report(final_state: MasterState) -> None:
    """Print the synthesised executive report to stdout."""
    print_banner("📊  EXECUTIVE SUMMARY")
    print(final_state.get("executive_summary", "No summary generated."))

    recs = final_state.get("recommendations", [])
    if recs:
        print_banner("✅  RECOMMENDATIONS")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec}")

    print_banner("🤖  HR ANALYSIS  (agent sub-state)")
    print(final_state.get("hr_analysis", "Not run."))

    print_banner("💰  FINANCE ANALYSIS  (agent sub-state)")
    print(final_state.get("finance_analysis", "Not run."))

    print_banner("🗺️   PIPELINE METADATA")
    print(f"  run_id         : {final_state.get('run_id')}")
    print(f"  planned_tasks  : {final_state.get('planned_tasks')}")
    print(f"  hr_status      : {final_state.get('hr_status')}")
    print(f"  finance_status : {final_state.get('finance_status')}")
    print(f"  pipeline_status: {final_state.get('pipeline_status')}")
    print(f"  persisted      : {final_state.get('persisted')}")
    print()


async def run_pipeline(query: str, pretty: bool) -> None:
    """Initialise resources, run graph, clean up."""
    # ── 1. Oracle pool ─────────────────────────────────────────
    db = OracleClient()
    db.init_pool()

    # ── 2. Build LangGraph pipeline ────────────────────────────
    pipeline = build_pipeline(db)

    # ── 3. Define initial state ────────────────────────────────
    #    Only `user_query` is set; all other keys are filled by nodes.
    initial_state: MasterState = {
        "user_query": query,
    }

    logger.info("Starting pipeline with query: %r", query)

    try:
        # ── 4. Run the graph ───────────────────────────────────
        #    ainvoke() runs the full graph asynchronously and returns
        #    the FINAL merged MasterState after all nodes have executed.
        final_state: MasterState = await pipeline.ainvoke(initial_state)

        # ── 5. Display results ─────────────────────────────────
        print_final_report(final_state)

        if pretty:
            print_banner("🔍  FULL FINAL STATE  (debug)")
            # Exclude raw Oracle rows for readability
            debug = {
                k: v for k, v in final_state.items()
                if k not in ("hr_raw_data", "finance_raw_data")
            }
            print(json.dumps(debug, indent=2, default=str))

    except Exception:
        logger.exception("Pipeline failed.")
        sys.exit(1)

    finally:
        # ── 6. Always close the pool ───────────────────────────
        db.close_pool()


def main() -> None:
    args = parse_args()

    if args.setup_schema:
        print("Setting up Oracle schema and seeding demo data...")
        create_schema(seed=True)
        print("Done. Run without --setup-schema to start the pipeline.")
        return

    asyncio.run(run_pipeline(query=args.query, pretty=args.pretty))


if __name__ == "__main__":
    main()

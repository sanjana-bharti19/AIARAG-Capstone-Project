"""Inspect persisted runs and search the answers.

Usage:
    python -m src.pipeline.query_results --runs       # list recent runs
    python -m src.pipeline.query_results              # all answers
    python -m src.pipeline.query_results RAG          # answers whose question contains "RAG"
"""
from __future__ import annotations
import sqlite3
import sys


def show_runs(con: sqlite3.Connection) -> None:
    """Print a fixed-width table of all runs, most-recent first."""
    print(f"{'id':>4}  {'started':>12}  {'questions':>10}  "
          f"{'retries':>8}  {'cost_usd':>10}  {'fail':>6}  {'fake':>4}")
    for row in con.execute(
        "SELECT id, started_at, n_questions, n_retries_total, total_cost_usd, "
        "       fail_rate, use_fake "
        "FROM runs ORDER BY id DESC"
    ):
        print(f"{row[0]:>4}  {row[1]:>12.1f}  {row[2]:>10}  {row[3]:>8}  "
              f"{row[4]:>10.4f}  {row[5]:>6.2f}  {row[6]:>4}")


def search_answers(con: sqlite3.Connection, pattern: str) -> None:
    """Print all answers whose question matches `%pattern%` (empty pattern = all)."""
    rows = con.execute(
        "SELECT id, run_id, retries, question, answer FROM answers "
        "WHERE question LIKE ? ORDER BY id",
        (f"%{pattern}%",),
    ).fetchall()
    for _id, run_id, retries, q, a in rows:
        print(f"[#{_id} run={run_id} retries={retries}] {q}")
        print(f"   → {a[:140]}")
        print()


def main() -> None:
    con = sqlite3.connect("results.db")
    if len(sys.argv) > 1 and sys.argv[1] == "--runs":
        show_runs(con)
    else:
        pattern = sys.argv[1] if len(sys.argv) > 1 else ""
        search_answers(con, pattern)


if __name__ == "__main__":
    main()

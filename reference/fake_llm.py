"""Fake LLM stand-in for Week 2 development.

Has the same shape as the real AsyncOpenAI client wrapper so the rest of the
pipeline doesn't know which path it's on — flip Settings.use_fake to swap.
"""
from __future__ import annotations
import asyncio
import random

from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Domain models — reused across the package
# ─────────────────────────────────────────────────────────────────────────────
class Question(BaseModel):
    text: str


class Answer(BaseModel):
    question: str
    text:     str
    cost_usd: float
    retries:  int = 0


class FakeLLMError(Exception):
    """Simulated transient API failure."""


# ─────────────────────────────────────────────────────────────────────────────
# Canned responses keyed by question keywords
# ─────────────────────────────────────────────────────────────────────────────
_CANNED = {
    "rag":         "RAG combines retrieval over a document corpus with an LLM, so answers are grounded in real sources rather than the model's training data alone.",
    "vector":      "Vector databases power semantic search, RAG context retrieval, and similarity-based recommendation systems.",
    "hallucinate": "LLMs hallucinate when they produce confident text that isn't grounded in their training data or any provided context.",
    "pydantic":    "A Pydantic BaseModel validates types and constraints at construction — invalid data is caught at the door, not three function calls later.",
    "async":       "async and await are Python keywords for cooperative concurrency. `async def` creates a coroutine; `await` pauses it until another coroutine completes.",
    "gather":      "asyncio.gather schedules every coroutine on the event loop concurrently and returns when all of them are done, preserving input order.",
    "backoff":     "Exponential backoff waits 1, 2, 4, 8... seconds between retries — far gentler on a struggling API than constant-interval retry.",
    "json log":    "JSON-formatted logs are machine-readable: every field is queryable, no regex parsing needed.",
    "fastapi":     "The @app.post('/path') decorator registers an async function as a POST endpoint handler at the given path.",
    "foreign key": "A foreign key declares that a column references the primary key of another table, enforcing referential integrity between rows.",
    "api key":     "Loading API keys from environment variables keeps them out of source control and makes per-environment rotation trivial.",
    "422":         "HTTP 422 Unprocessable Entity means the request was well-formed JSON but failed semantic validation — usually a field type or constraint violation.",
    "parameter":   "Parameterised SQL queries (`?` placeholders) prevent injection by sending the query template and values as separate channels.",
    "select":      "SELECT reads rows from a table; INSERT writes new rows into one. SELECT is idempotent; INSERT changes state.",
    "dependency":  "Dependency injection passes a component's dependencies in from outside rather than constructing them internally — makes testing and substitution easier.",
    "unit test":   "A unit test exercises one function or method in isolation, asserting that for a given input it produces the expected output or side effect.",
    "virtual":     "A Python virtual environment is an isolated installation directory — packages installed in one venv don't affect another or the system Python.",
    "rate":        "Rate limiting caps how many requests a client can make per time window, protecting the API from overload and unfair use.",
}


def _canned_for(question_text: str) -> str:
    text_lower = question_text.lower()
    for keyword, answer in _CANNED.items():
        if keyword in text_lower:
            return answer
    return f"(simulated answer for: {question_text[:60]}...)"


# ─────────────────────────────────────────────────────────────────────────────
# The fake call — same signature as a real async LLM client
# ─────────────────────────────────────────────────────────────────────────────
async def fake_ask_llm(q: Question, fail_rate: float = 0.0) -> Answer:
    """Pretend to call an LLM. Sleeps briefly. May raise FakeLLMError.

    Args:
        q: The Question to answer.
        fail_rate: Probability of raising FakeLLMError (0.0 → never, 1.0 → always).

    Returns:
        An Answer with a canned response and a placeholder cost.

    Raises:
        FakeLLMError: With probability `fail_rate`, before any sleep.
    """
    # Simulate variable latency (300-1500 ms)
    await asyncio.sleep(random.uniform(0.3, 1.5))

    # Simulate transient failure
    if random.random() < fail_rate:
        raise FakeLLMError(f"simulated transient failure for: {q.text[:40]}")

    return Answer(
        question=q.text,
        text=_canned_for(q.text),
        cost_usd=0.0001,
        retries=0,
    )

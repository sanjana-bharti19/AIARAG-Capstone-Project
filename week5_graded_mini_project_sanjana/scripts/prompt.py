import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = settings.DATA_DIR

client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL
)

MODEL = settings.DEFAULT_MODEL
JUDGE_MODEL = settings.JUDGE_MODEL
TEMPERATURE = settings.TEMPERATURE


def load_data() -> tuple[list[dict], dict[str, dict]]:
    snippets = [
        json.loads(line)
        for line in (DATA_DIR / "job_snippets.jsonl").read_text().splitlines()
        if line.strip()
    ]
    golden = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (DATA_DIR / "golden_set.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    return snippets, golden


def prompt_zero_shot(snippet_text: str) -> list[dict]:
    """Strategy 1 — zero-shot."""
    return [
        {
            "role": "user",
            "content": (
                "Extract the following fields from the job snippet and return JSON only: "
                "company, role, years_experience_required.\n\n"
                f"Snippet:\n{snippet_text}"
            ),
        }
    ]


def prompt_few_shot(snippet_text: str) -> list[dict]:
    """Strategy 2 — few-shot."""
    examples = """
Example 1:
Input: "Acme Corp is hiring a Senior Software Engineer with 5+ years..."
Output: {"company": "Acme Corp", "role": "Senior Software Engineer", "years_experience_required": 5}

Example 2:
Input: "Northwind Ltd seeks a Data Analyst with 2 years of SQL experience."
Output: {"company": "Northwind Ltd", "role": "Data Analyst", "years_experience_required": 2}

Example 3:
Input: "Globex is looking for a Product Manager with at least 4 years in B2C."
Output: {"company": "Globex", "role": "Product Manager", "years_experience_required": 4}
"""
    return [
        {"role": "user", "content": examples + "\n\nNow extract from this snippet:\n" + snippet_text},
    ]


def prompt_structured(snippet_text: str) -> list[dict]:
    """Strategy 3 — structured / role-based."""
    system_prompt = (
        "You are an expert recruiter and data extraction specialist. "
        "Extract information exactly and conservatively from the job snippet. "
        "Return valid JSON with exactly these keys: company, role, years_experience_required. "
        "Use integers for years_experience_required and do not invent missing data."
    )
    user_prompt = (
        "Return JSON only.\n\n"
        "Schema: {\n"
        '  "company": "string",\n'
        '  "role": "string",\n'
        '  "years_experience_required": integer\n'
        "}\n\n"
        f"Snippet:\n{snippet_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def prompt_cot(snippet_text: str) -> list[dict]:
    """Strategy 4 — chain-of-thought."""
    return [
        {
            "role": "user",
            "content": (
                "Think step by step about the job posting. Identify the company, the job role, and the minimum years of experience required. "
                "Then return only valid JSON with keys: company, role, years_experience_required.\n\n"
                f"Snippet:\n{snippet_text}"
            ),
        }
    ]


STRATEGIES = {
    "zero_shot": prompt_zero_shot,
    "few_shot": prompt_few_shot,
    "structured": prompt_structured,
    "cot": prompt_cot,
}


def parse_response(text: str) -> dict | None:
    """Simple JSON parser for the model output."""
    try:
        clean_text = text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.replace("```json", "").replace("```", "").strip()

        return json.loads(clean_text)
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON.")
        return {}
    except Exception as e:
        print(f"An API error occurred: {e}")
        return {}


async def run_one(strategy_name: str, snippet: dict) -> dict:
    """Run one prompt strategy against one job snippet."""
    strategy_fn = STRATEGIES[strategy_name]
    messages = strategy_fn(snippet["snippet"])

    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )
    latency_s = time.perf_counter() - start

    raw_text = response.choices[0].message.content or ""
    parsed = parse_response(raw_text)

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    model_rates = settings.OPENAI_COST_RATES.get(MODEL, {"in": 0.0, "out": 0.0})
    cost_usd = (
        prompt_tokens * model_rates["in"]
        + completion_tokens * model_rates["out"]
    )

    return {
        "strategy": strategy_name,
        "snippet_id": snippet["id"],
        "raw_response": raw_text,
        "parsed_extraction": parsed,
        "cost_usd": cost_usd,
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def run_all() -> list[dict]:
    """Run all snippet x strategy combinations in parallel."""
    snippets, _ = load_data()
    tasks = [
        asyncio.create_task(run_one(strategy_name, snippet))
        for strategy_name in STRATEGIES
        for snippet in snippets
    ]
    return await asyncio.gather(*tasks)


def normalize_value(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(int(value))
    return str(value).strip().lower()


def score_accuracy(extracted: dict | None, gold: dict) -> int:
    """Compare the 3 target fields and count matches."""
    if not isinstance(extracted, dict):
        return 0

    fields = ["company", "role", "years_experience_required"]
    matches = 0
    for field in fields:
        if normalize_value(extracted.get(field)) == normalize_value(gold.get(field)):
            matches += 1
    return matches


async def score_llm_judge(snippet_text: str, extracted: dict | None, gold: dict) -> int:
    """Judge the extraction quality on a 1-4 rubric."""
    if not isinstance(extracted, dict):
        return 1

    judge_prompt = (
        "You are evaluating a structured extraction against a gold answer.\n\n"
        "Snippet:\n"
        f"{snippet_text}\n\n"
        "Gold answer:\n"
        f"{json.dumps(gold, ensure_ascii=False)}\n\n"
        "Extracted answer:\n"
        f"{json.dumps(extracted, ensure_ascii=False)}\n\n"
        "Return only an integer from 1 to 4.\n"
        "4 = all three fields correct; "
        "3 = two correct and no fabricated data; "
        "2 = one correct or fabricated data; "
        "1 = none correct or unparsable."
    )

    response = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
    )

    text = response.choices[0].message.content or ""
    match = re.search(r"(\d)", text)
    if match:
        score = int(match.group(1))
        return max(1, min(4, score))
    return 1


async def score_results(results: list[dict], golden: dict[str, dict]) -> list[dict]:
    """Attach accuracy, parse_success, and llm judge scores to each result."""
    scored = []
    for item in results:
        snippet_id = item["snippet_id"]
        gold = golden.get(snippet_id, {})
        snippet_text = next(
            (s["snippet"] for s in load_data()[0] if s["id"] == snippet_id),
            "",
        )

        extracted = item.get("parsed_extraction")
        item["accuracy"] = score_accuracy(extracted, gold)
        item["parse_success"] = int(extracted is not None)
        item["llm_judge_score"] = await score_llm_judge(snippet_text, extracted, gold)
        scored.append(item)
    return scored


def build_summary(scored: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(scored)
    summary = (
        df.groupby("strategy")
        .agg(
            {
                "accuracy": "mean",
                "parse_success": "mean",
                "llm_judge_score": "mean",
                "cost_usd": "sum",
                "latency_s": "median",
            }
        )
        .round(3)
    )
    summary.columns = [
        "Accuracy (mean of 3)",
        "Parse rate",
        "Judge score",
        "Total cost ($)",
        "Latency p50 (s)",
    ]
    return summary


async def main() -> None:
    snippets, golden = load_data()
    results = await run_all()
    scored = await score_results(results, golden)

    summary = build_summary(scored)
    print(f"Scored {len(scored)} results.")
    print(summary)

    output_path = settings.RESULTS_JSON_PATH
    output_path.write_text(json.dumps(scored, indent=2, ensure_ascii=False))
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

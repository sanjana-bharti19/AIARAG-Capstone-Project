# MP1 Prompt Lab

This project compares four prompting strategies for structured extraction from job-posting snippets:

- zero-shot
- few-shot
- structured
- chain-of-thought

The task is to extract:

- company
- role
- years_experience_required

and compare each strategy using:

- parse success
- extraction accuracy
- LLM judge score
- total cost
- latency

---

## Project files

- `prompt.py` — main script that runs the full workflow
- `data/job_snippets.jsonl` — 10 input job snippets
- `data/golden_set.jsonl` — gold-standard outputs for scoring
- `config/settings.py` — project settings and OpenAI cost rates
- `scripts/run_cost_calculation.py` — simple cost calculator
- `learner/MP1_Starter_Template.ipynb` — notebook version of the same project
- `results.json` — saved outputs from the script run

---

## Setup

1. Open a terminal in the project folder.
2. Create or activate a Python environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set OpenAI API key:

```bash
export OPENAI_API_KEY="api_key_here"
```

or create a `.env` file in the project root with:

```env
OPENAI_API_KEY=api_key_here
```

---

## Run the project

From the project root:

```bash
python scripts/prompt.py
```

This script will:

- load the snippets and golden set
- run all four strategies
- parse each model response
- calculate cost and latency
- score results against the gold set
- print a summary table
- save the raw results to `results.json`

---

## Interpret the results

The final summary compares strategies across these metrics:

- Accuracy (mean of the 3 extraction fields)
- Parse rate
- Judge score
- Total cost
- Latency p50

### How to read it

- Higher accuracy = better extraction quality
- Higher parse rate = more reliable JSON output
- Higher judge score = stronger overall answer quality
- Lower cost = cheaper execution
- Lower latency = faster execution

A good strategy is usually the one that balances all of these, not just the highest accuracy.

---


## Troubleshooting

### Module import error

Run the script from the project root, not from inside a nested folder.

### Missing API key

Check that `OPENAI_API_KEY` is exported or present in the `.env` file.

### JSON parsing problems

Some models return markdown code fences. The parser handles that automatically, but if the output is malformed, it may return `None` for parse success.

---

## Expected output

After running the script, we should have:

- a printed comparison table in the terminal
- a `results.json` file in the project root
- a summary write-up in `mp1_comparison.md`

This gives  both the raw results and a concise comparison of the strategies.

Git repo link:

https://github.com/sanjana-bharti19/AIARAG-Capstone-Project
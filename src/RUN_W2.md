# W2 — How to run and verify on Vocareum

This package is a fully laid-out W2 capstone repo at the **end of Lab Step 4**
(SQLite persistence). It contains only the existing shipped reference code —
nothing modified, nothing missing.

---

## What's inside

```
W2_package/
├── src/
│   ├── __init__.py
│   └── pipeline/
│       ├── __init__.py
│       ├── settings.py            ← W2 reference (Settings + RunSummary)
│       ├── logging_config.py      ← W2 reference (JsonFormatter + get_logger)
│       ├── fake_llm.py            ← W2 starter (copied as-is per Step 2a)
│       ├── pipeline.py            ← W2 reference (the completed pipeline)
│       ├── store.py               ← W2 reference (SQLite persistence)
│       └── query_results.py       ← W2 reference (query CLI)
├── data/
│   └── questions.csv              ← W2 lab CSV (20 questions)
├── reference/                      ← extra copies from cohort-repo/week2/
│   ├── pipeline_starter.py        ← starter with TODOs
│   ├── pipeline_reference.py      ← same content as src/pipeline/pipeline.py
│   ├── pydantic_demo.py           ← optional reading
│   └── fake_llm.py
├── requirements.txt                ← openai, pydantic, python-dotenv, httpx
├── logs/                           ← the script fills this with per-step logs
├── RUN_W2.sh                       ← run this (single script)
└── RUN_W2.md                       ← this file
```

**md5sum-verified:** every `.py` file in `src/pipeline/` is byte-identical to
its published counterpart in `/mnt/user-data/outputs/AI-RAG_W2_*.py`. No
modifications, no patches.

---

## How to run on Vocareum

```bash
unzip AI-RAG_W2_Package.zip
cd W2_package
bash RUN_W2.sh
```

That's it. The script does everything:

1. Confirms Python + `OPENAI_API_KEY`.
2. Runs `pip install -r requirements.txt`.
3. Runs **every "Run it" command from the W2 lab guide** against the shipped
   reference code (21 steps in total — environment, Settings validation,
   logger, `ask_llm`, retry logic, `run_batch`, CSV loading, `RunSummary`,
   end-to-end pipeline, SQLite persistence, query CLI).
4. Captures each command's output to a dedicated file under `logs/`.
5. Writes `logs/_SUMMARY.log` with a one-line PASS/FAIL per step.

Expected wall-clock: about 30-60 seconds (most of it is `pip install`).

---

## What you should see

At the end of the run, the console prints a table like:

```
STEP                                       VERDICT    DESCRIPTION
----                                       -------    -----------
step0a_python_version                      PASS       Python version
step0a_openai_key                          PASS       OPENAI_API_KEY present
step1b_pip_install                         PASS       pip install -r requirements.txt
step1b_imports_ok                          PASS       Required packages importable
step1c_settings_validates                  PASS       Settings(batch_size=0) raises as expected
step1c_settings_default                    PASS       Settings() with defaults constructs cleanly
step1d_logger_writes                       PASS       logger writes one JSON line to logs/pipeline.log
step1d_log_line_present                    PASS       logs/pipeline.log contains the smoke line
step2b_ask_llm_single                      PASS       ask_llm returns Answer with text/cost/retries
step2c_retry_clean                         PASS       ask_llm_with_retry succeeds without retries
step2c_retry_lossy_raises                  PASS       ask_llm_with_retry raises after 3 attempts
step2d_run_batch                           PASS       run_batch processes 3 questions concurrently
step3b_load_questions                      PASS       load_questions reads CSV into Question objects
step3d_run_summary_validates               PASS       RunSummary(elapsed_seconds=-1) raises as expected
step3d_run_summary_ok                      PASS       RunSummary constructs with valid data
step3e_end_to_end_fake                     PASS       python -m src.pipeline.pipeline (end-to-end fake)
step3e_results_json_present                PASS       results.json was written with expected keys
step4a_store_connect                       PASS       store.connect creates the schema
step4b_results_db_has_rows                 PASS       results.db has a runs row + answers rows
step4c_query_runs                          PASS       query_results --runs lists the run row
step4c_query_search                        PASS       query_results 'RAG' returns matching answers
```

**All 21 steps PASS** = the instructor's W2 lab session will run without errors.

A FAIL on any step is captured raw in the corresponding `logs/<step>.log`.
Nothing is auto-fixed — you (and the SME) see exactly what failed.

---

## What to send back to me

Either of these is enough:

- **Just the summary** — `cat logs/_SUMMARY.log` and paste the output.
- **Full logs** — `tar czf w2_logs.tgz logs/` and attach the tarball.

If any step shows FAIL, also include the contents of that step's log:

```bash
cat logs/<failed_step_id>.log
```

I'll read what you send back and confirm whether W2 is good to ship to the
instructor, or which exact file needs a real fix (which **you** would decide
whether to make, not me).

---

## Negative tests (steps that are *supposed* to "raise")

Three steps deliberately exercise validation logic that **must raise an
error**. They're labelled `*_validates` or `*_raises`. The Python snippet
catches the expected exception and exits 0 if (and only if) the exception
fires correctly:

- `step1c_settings_validates` — `Settings(batch_size=0)` must raise
  `ValidationError` (because the field has `gt=0`).
- `step3d_run_summary_validates` — `RunSummary(elapsed_seconds=-1)` must raise
  `ValidationError` (because the field has `ge=0`).
- `step2c_retry_lossy_raises` — `ask_llm_with_retry` with `fail_rate=1.0`
  must raise after 3 attempts.

A PASS on these means the validation logic is doing its job. A FAIL means
the constraint isn't enforced — which would be a real bug.

---

## What this script does NOT do

- Does **not** modify any file in `src/`, `data/`, `reference/`, or
  `requirements.txt`.
- Does **not** auto-fix any failure. Every command's output is captured raw.
- Does **not** require an interactive prompt — runs fully unattended.

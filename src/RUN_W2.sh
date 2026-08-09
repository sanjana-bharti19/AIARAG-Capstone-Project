#!/usr/bin/env bash
# ============================================================================
# RUN_W2.sh — Verifies W2 reference code on Vocareum.
#
# What this script does:
#   1. Confirms Python + OPENAI_API_KEY are present.
#   2. Installs requirements.txt.
#   3. Runs every "Run it" command from the W2 lab guide against the shipped
#      reference code (already laid out in src/pipeline/ + data/).
#   4. Captures each command's output to a dedicated file under logs/.
#   5. Writes logs/_SUMMARY.log with a one-line PASS/FAIL per step.
#
# What this script does NOT do:
#   - Modify any code in src/, reference/, or data/.
#   - Auto-fix or hide any failure. Every step's output is captured raw.
#
# When done, zip up logs/ and share — that's enough for the curriculum SME
# to verify whether the instructor's lab session will run without errors.
# ============================================================================

set -u   # error on undefined vars; do NOT use -e (we keep going past failures)

# Move to the package root regardless of where the user invokes the script
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 2

LOG_DIR="$ROOT/logs"
SUMMARY="$LOG_DIR/_SUMMARY.log"

mkdir -p "$LOG_DIR"
# Clean any state from a previous run so the verification starts fresh
rm -f "$LOG_DIR"/*.log
rm -f "$ROOT/results.json" "$ROOT/results.db" "$ROOT/test.db"
rm -rf "$ROOT/src/pipeline/__pycache__" "$ROOT/src/__pycache__"

# Master summary header
{
    echo "W2 Verification Run"
    echo "==================="
    echo "Timestamp:  $(date -Iseconds)"
    echo "PWD:        $ROOT"
    echo "Python:     $(python --version 2>&1)"
    echo "Has API key: $(python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))" 2>&1)"
    echo
    printf "%-44s  %-30s  %s\n" "STEP" "VERDICT" "DESCRIPTION"
    printf "%-44s  %-30s  %s\n" "----" "-------" "-----------"
} > "$SUMMARY"

# ---------------------------------------------------------------------------
# Helper: run_step <step_id> <description> <expected_rc> <command-string>
#
# - command-string is run via `bash -c "$cmd"` so it supports pipes/redirects.
# - The full output (stdout+stderr) is captured to logs/<step_id>.log.
# - expected_rc=0 by convention; pass "any" to treat any exit code as DONE
#   (used for housekeeping steps like the install).
# ---------------------------------------------------------------------------
run_step () {
    local step_id="$1"
    local description="$2"
    local expected_rc="$3"
    local cmd="$4"
    local log_file="$LOG_DIR/${step_id}.log"

    {
        echo "STEP:        $step_id"
        echo "DESCRIPTION: $description"
        echo "EXPECTED RC: $expected_rc"
        echo "TIMESTAMP:   $(date -Iseconds)"
        echo "COMMAND:"
        echo "$cmd" | sed 's/^/    /'
        echo "--- OUTPUT ---"
    } > "$log_file"

    bash -c "$cmd" >> "$log_file" 2>&1
    local rc=$?

    local verdict
    if [ "$expected_rc" = "any" ]; then
        verdict="DONE  (rc=$rc)"
    elif [ "$rc" = "$expected_rc" ]; then
        verdict="PASS"
    else
        verdict="FAIL  (rc=$rc, expected $expected_rc)"
    fi

    echo "  [$verdict]  $step_id  —  $description"
    printf "%-44s  %-30s  %s\n" "$step_id" "$verdict" "$description" >> "$SUMMARY"

    {
        echo "--- END ---"
        echo "EXIT_CODE: $rc"
        echo "VERDICT:   $verdict"
    } >> "$log_file"
}

# ===========================================================================
# Step 0 — Environment checks (Lab Guide Step 0a)
# ===========================================================================
echo ""
echo "=========================================="
echo " Step 0 — Environment"
echo "=========================================="

run_step "step0a_python_version"   "Python version"           0 \
    "python --version"

run_step "step0a_openai_key"       "OPENAI_API_KEY present"   0 \
    "python -c \"import os; ok=bool(os.getenv('OPENAI_API_KEY')); print('OPENAI_API_KEY present:', ok); raise SystemExit(0 if ok else 1)\""

# ===========================================================================
# Step 1 — Skeleton (Lab Guide Steps 1a-1d)
# ===========================================================================
echo ""
echo "=========================================="
echo " Step 1 — Skeleton"
echo "=========================================="

# Lab guide says: pip install -r requirements.txt
# Vocareum's Python 3.10 is externally-managed, so --break-system-packages is needed
# (same flag W1 used; on systems that don't need it, it's a harmless no-op).
run_step "step1b_pip_install"      "pip install -r requirements.txt"  0 \
    "pip install --break-system-packages -q -r requirements.txt"

run_step "step1b_imports_ok"       "Required packages importable"     0 \
    "python -c \"import openai, pydantic, dotenv, httpx; print('OK · pydantic', pydantic.VERSION, '· openai', openai.__version__)\""

# Lab Guide Step 1c run-it: Settings(batch_size=0) should raise ValidationError
run_step "step1c_settings_validates"  "Settings(batch_size=0) raises ValidationError as expected"  0 \
    "python -c \"
from src.pipeline.settings import Settings
from pydantic import ValidationError
try:
    Settings(batch_size=0)
    print('FAIL — no validation error raised')
    raise SystemExit(1)
except ValidationError as e:
    print('PASS — ValidationError raised as expected:', str(e).splitlines()[0])
\""

run_step "step1c_settings_default"    "Settings() with defaults constructs cleanly"  0 \
    "python -c \"
from src.pipeline.settings import Settings
import json
print(json.dumps(Settings().model_dump(mode='json'), indent=2))
\""

# Lab Guide Step 1d run-it: settings + logger smoke
run_step "step1d_logger_writes"       "logger writes one JSON line to logs/pipeline.log"  0 \
    "python -c \"
from src.pipeline.settings import Settings
from src.pipeline.logging_config import get_logger
s = Settings()
log = get_logger()
log.info(f'smoke test — settings: {s.model_dump(mode=\\\"json\\\")}')
print('OK — settings constructed, log line written')
\""

run_step "step1d_log_line_present"    "logs/pipeline.log contains the smoke line"   0 \
    "test -f logs/pipeline.log && tail -1 logs/pipeline.log"

# ===========================================================================
# Step 2 — Pipeline basics (Lab Guide Steps 2b-2e)
# ===========================================================================
echo ""
echo "=========================================="
echo " Step 2 — Pipeline basics"
echo "=========================================="

run_step "step2b_ask_llm_single"      "ask_llm returns Answer with text/cost/retries"  0 \
    "python -c \"
import asyncio
from src.pipeline.pipeline import ask_llm, Question
ans = asyncio.run(ask_llm(Question(text='What is RAG in one sentence?')))
print('type:', type(ans).__name__)
print('text:', ans.text[:80])
print('cost:', ans.cost_usd)
print('retries:', ans.retries)
\""

run_step "step2c_retry_clean"         "ask_llm_with_retry succeeds without retries on clean run"  0 \
    "python -c \"
import asyncio
from src.pipeline.pipeline import ask_llm_with_retry, Question
ans = asyncio.run(ask_llm_with_retry(Question(text='What is RAG?')))
print('clean:    retries =', ans.retries)
\""

run_step "step2c_retry_lossy_raises"  "ask_llm_with_retry raises after 3 attempts on always-fail"  0 \
    "python -c \"
import asyncio
from src.pipeline.pipeline import ask_llm_with_retry, Question
try:
    asyncio.run(ask_llm_with_retry(Question(text='What is RAG?'), fail_rate=1.0))
    print('FAIL — exception was expected but none raised')
    raise SystemExit(1)
except Exception as e:
    print('lossy:    raised after 3 attempts:', type(e).__name__)
\""

run_step "step2d_run_batch"           "run_batch processes 3 questions concurrently"  0 \
    "python -c \"
import asyncio, time
from src.pipeline.pipeline import run_batch, Question
questions = [
    Question(text='What is RAG?'),
    Question(text='Name three uses of vector databases.'),
    Question(text='Why might an LLM hallucinate?'),
]
t0 = time.time()
answers = asyncio.run(run_batch(questions))
elapsed = time.time() - t0
print(f'wall-clock: {elapsed:.2f}s')
for a in answers:
    print(f'- {a.text[:60]}')
\""

# ===========================================================================
# Step 3 — Extended pipeline (Lab Guide Steps 3a-3e)
# ===========================================================================
echo ""
echo "=========================================="
echo " Step 3 — Extended pipeline"
echo "=========================================="

run_step "step3b_load_questions"      "load_questions reads CSV into Question objects"  0 \
    "python -c \"
from src.pipeline.pipeline import load_questions
qs = load_questions()
print('count:', len(qs))
print('first:', qs[0])
print('type:', type(qs[0]).__name__)
\""

run_step "step3d_run_summary_validates"  "RunSummary(elapsed_seconds=-1) raises as expected"  0 \
    "python -c \"
from src.pipeline.settings import RunSummary
from pydantic import ValidationError
try:
    RunSummary(started_at=0, elapsed_seconds=-1, n_questions=0, n_succeeded=0, n_retries_total=0, total_cost_usd=0, fail_rate=0, use_fake=True)
    print('FAIL — no validation error raised')
    raise SystemExit(1)
except ValidationError as e:
    print('PASS — ValidationError raised as expected:', str(e).splitlines()[0])
\""

run_step "step3d_run_summary_ok"      "RunSummary constructs with valid data"  0 \
    "python -c \"
from src.pipeline.settings import RunSummary
import json
s = RunSummary(started_at=1717.0, elapsed_seconds=4.42, n_questions=20, n_succeeded=20, n_retries_total=0, total_cost_usd=0.002, fail_rate=0.0, use_fake=True)
print(json.dumps(s.model_dump(), indent=2))
\""

run_step "step3e_end_to_end_fake"     "python -m src.pipeline.pipeline (end-to-end fake)"  0 \
    "python -m src.pipeline.pipeline"

run_step "step3e_results_json_present"   "results.json was written and has the expected keys"  0 \
    "python -c \"
import json
d = json.load(open('results.json'))
print('top-level keys:', sorted(d.keys()))
print('summary keys:', sorted(d['summary'].keys()))
print('n answers:', len(d['answers']))
print('first answer text:', d['answers'][0]['text'][:80])
\""

# ===========================================================================
# Step 4 — SQLite persistence (Lab Guide Steps 4a-4d)
# ===========================================================================
echo ""
echo "=========================================="
echo " Step 4 — SQLite persistence"
echo "=========================================="

run_step "step4a_store_connect"       "store.connect creates the schema"   0 \
    "python -c \"
from src.pipeline.store import connect
con = connect('test.db')
print(con.execute('SELECT name FROM sqlite_master WHERE type=\\\"table\\\"').fetchall())
\" && rm -f test.db"

# Step 3e already ran the pipeline and persisted to results.db, so check the rows
run_step "step4b_results_db_has_rows"  "results.db has a runs row + answers rows"   0 \
    "python -c \"
import sqlite3
con = sqlite3.connect('results.db')
print('runs:')
for r in con.execute('SELECT id, n_questions, n_retries_total, total_cost_usd, fail_rate, use_fake FROM runs'):
    print(' ', r)
print('answers per run:')
for r in con.execute('SELECT run_id, COUNT(*) FROM answers GROUP BY run_id'):
    print(' ', r)
\""

run_step "step4c_query_runs"          "query_results --runs lists the run row"   0 \
    "python -m src.pipeline.query_results --runs"

run_step "step4c_query_search"        "query_results 'RAG' returns matching answers"   0 \
    "python -m src.pipeline.query_results RAG"

# ===========================================================================
# Done — final summary
# ===========================================================================
echo ""
echo "=========================================="
echo " Summary"
echo "=========================================="
cat "$SUMMARY"
echo ""
echo "All step logs are in: $LOG_DIR/"
echo "Top-line summary is in: $SUMMARY"
echo ""
echo "To share, either zip the logs/ directory or just paste the contents of:"
echo "  $SUMMARY"
echo "plus any individual log file where you see FAIL."

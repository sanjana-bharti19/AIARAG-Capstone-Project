"""MP3 · Grounded Q&A — Starter
================================

The load → chunk → embed → store → retrieve → answer pipeline is PROVIDED and
working (it's the MP2 loop). Your job is the three things that make it
*trustworthy*. Fill in the FOUR functions marked TODO:

    format_context     (citations)   validate_citations (citations)
    should_abstain     (abstention)  evaluate           (evaluation)

Everything else — the pipeline, the prompt, `answer()`, the table renderer, the
CLI — is done for you. Once your four functions are in, `answer()` and `eval`
will work end to end.

Check your work offline (no API key needed):
    python mp3_grounded_qa.py selftest      # goes green once your 4 functions are correct

Under the hood this is the MP2 Mini-RAG pipeline extended into a *trustworthy*
assistant:

  1. CITATIONS  — the answer cites each claim inline with [n], and every [n]
                  is validated against an actually-retrieved chunk.
  2. ABSTENTION — when retrieval finds no supporting chunk, the assistant says
                  "I don't know based on the provided documents" instead of
                  guessing.
  3. EVALUATION — a small harness measures retrieval hit rate, grounded rate,
                  and abstention accuracy over a question set, and prints a
                  table (this is the graded artifact).

The load → chunk → embed → store → retrieve loop is carried from MP2. The three
capabilities above are the MP3 additions.

Run:
    pip install -r requirements.txt
    source .env
    python mp3_grounded_qa.py ingest      # build the collection (once)
    python mp3_grounded_qa.py ask         # interactive Q&A with citations + abstention
    python mp3_grounded_qa.py eval        # the retrieval-quality evaluation table
    python mp3_grounded_qa.py selftest    # OFFLINE checks of the new logic (no keys)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR        = settings.CORPUS_DIR
DATA_DIR          = settings.DATA_DIR
COLLECTION_NAME   = settings.COLLECTION_NAME
EMBEDDING_MODEL   = settings.EMBEDDING_MODEL
EMBEDDING_DIM     = settings.EMBEDDING_DIM
CHAT_MODEL        = settings.CHAT_MODEL
TARGET_CHUNK_SIZE = settings.TARGET_CHUNK_SIZE
CHUNK_OVERLAP     = settings.CHUNK_OVERLAP

openai = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
oai = openai
qdrant = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY or None,
)

RETRIEVAL_THRESHOLD = float(os.environ.get("RETRIEVAL_THRESHOLD", "0.30"))  # abstain if top chunk scores below this (cosine); tune via .env

ABSTAIN_MESSAGE = "I don't know based on the provided documents."


def _require_clients() -> None:
    if not settings.OPENAI_API_KEY:
        sys.exit("Missing OPENAI_API_KEY in .env/settings.")
    if not settings.QDRANT_URL:
        sys.exit("Missing QDRANT_URL in .env/settings.")



# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:

    docs: list[dict[str, Any]] = []

    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        title = ""
        for line in lines:
            cleaned = line.strip()
            if cleaned:
                title = cleaned
                break

        docs.append(
            {
                "source": path.name,
                "title": title,
                "text": text,
            }
        )

    return docs


# ─── Step 2: Chunk each document ────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Chunk by paragraph/section with overlap to preserve cross-boundary facts."""
    text = (doc.get("text") or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return []

    overlap = max(0, int(getattr(settings, "CHUNK_OVERLAP", 0) or 0))
    target = max(200, int(getattr(settings, "TARGET_CHUNK_SIZE", 500) or 500))

    chunks: list[dict[str, Any]] = []
    section_name = "Overview"
    section_index = 1
    buffer: list[str] = []
    buffer_len = 0
    previous_tail: list[str] = []

    def flush_chunk(chunk_paras: list[str], chunk_section: str) -> None:
        nonlocal section_name, section_index
        if not chunk_paras:
            return
        chunk_text = "\n\n".join(chunk_paras).strip()
        chunks.append(
            {
                "source": doc["source"],
                "title": doc["title"],
                "section": chunk_section,
                "text": chunk_text,
            }
        )
        section_index += 1
        section_name = f"Part {section_index}"

    for para in paragraphs:
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        first_line = lines[0] if lines else ""

        is_heading = (
            bool(first_line)
            and len(first_line) < 80
            and len(first_line.split()) <= 12
            and not re.search(r"[.!?]$", first_line)
        )

        if is_heading:
            if buffer:
                flush_chunk(buffer, section_name)
                previous_tail = buffer[-max(1, overlap // 120):]
                buffer = []
                buffer_len = 0
            section_name = first_line
            continue

        para_len = len(para)
        if buffer and buffer_len + 1 + para_len > target:
            flush_chunk(buffer, section_name)
            if overlap > 0:
                overlap_count = max(1, min(len(buffer), max(1, overlap // 120)))
                previous_tail = buffer[-overlap_count:]
            else:
                previous_tail = []
            buffer = list(previous_tail)
            buffer_len = sum(len(p) for p in buffer)
            previous_tail = []

        if buffer and overlap > 0 and buffer_len + para_len <= target:
            buffer.append(para)
            buffer_len += para_len
            continue

        if buffer and overlap > 0 and buffer_len >= target // 2:
            # Preserve continuity at the chunk boundary without exploding chunk size.
            pass

        buffer.append(para)
        buffer_len += para_len

    if buffer:
        flush_chunk(buffer, section_name)

    return chunks


# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]


# ─── Step 4: Set up the Qdrant collection ───────────────────────────────

def setup_collection() -> None:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )


# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    
    if not chunks:
        return

    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=chunk,
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    query_vector = embed_texts([query])[0]
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )

    hits = getattr(response, "points", []) or []
    results: list[dict[str, Any]] = []
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        payload["score"] = getattr(hit, "score", None)
        results.append(payload)

    return results


# ─── NEW 1 · Citations ──────────────────────────────────────────────────

def format_context(chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Number the retrieved chunks [1..n] for the prompt and return a parallel
    source map the answer's [n] markers can be validated against."""
    if not chunks:
        return "", []

    sections: list[str] = []
    sources: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, start=1):
        text = (chunk.get("text") or "").strip()
        title = chunk.get("title") or "Untitled"
        section = chunk.get("section") or "Overview"
        source = chunk.get("source") or ""
        score = chunk.get("score", 0.0)

        sections.append(f"[{idx}] (from \"{title}\" — {section})\n{text}")
        sources.append({
            "n": idx,
            "source": source,
            "title": title,
            "section": section,
            "score": score,
        })

    context = "\n\n".join(sections)
    return context, sources


def validate_citations(answer_text: str, n_sources: int) -> dict[str, Any]:
    """Parse [n] markers from the answer and check each maps to a real source.
    A fabricated citation (a number with no matching source) is the tell we
    care about."""
    matches = re.findall(r"\[(\d+)\]", answer_text or "")
    numbers = [int(m) for m in matches]

    used = [n for n in numbers if 1 <= n <= n_sources]
    invalid = [n for n in numbers if n < 1 or n > n_sources]

    return {
        "used": used,
        "invalid": invalid,
        "has_citation": bool(used),
    }


# ─── NEW 2 · Abstention ─────────────────────────────────────────────────

def should_abstain(chunks: list[dict[str, Any]], threshold: float = RETRIEVAL_THRESHOLD) -> bool:
    """Abstain when there is no sufficiently-relevant chunk to ground an answer."""
    if not chunks:
        return True

    best_score = max((float(chunk.get("score", 0.0)) for chunk in chunks), default=0.0)
    return best_score < threshold


SYSTEM_PROMPT = """You answer questions about a small collection of Sherlock Holmes stories.
Use ONLY the numbered excerpts provided below. After each claim, cite the excerpt it
came from using its number in square brackets, e.g. [1]. If the excerpts do not contain
the answer, reply with exactly: "I don't know based on the provided documents." Do not
use any outside knowledge."""


def split_compound_questions(question: str) -> list[str]:
    """Break a compound question into sub-questions for better retrieval coverage."""
    if not question:
        return []

    q = question.strip().rstrip()
    if not q:
        return []

    base = q[:-1] if q.endswith("?") else q
    patterns = [
        r"^(.*?)(?:,\s*and\s+|\s+and\s+)(.*)$",
        r"^(.*?)(?:;\s*and\s+|\s+and\s+)(.*)$",
    ]
    for pattern in patterns:
        m = re.match(pattern, base, flags=re.IGNORECASE)
        if not m:
            continue
        left = (m.group(1) or "").strip().rstrip(",;:")
        right = (m.group(2) or "").strip().rstrip(",;:")
        if left and right:
            return [
                *split_compound_questions(left + "?"),
                *split_compound_questions(right + "?"),
            ]
    return [q if q.endswith("?") else q + "?"]


def answer(question: str, k: int = 3, threshold: float = RETRIEVAL_THRESHOLD) -> dict[str, Any]:
    """Retrieve → (abstain?) → cite-formatted generation → validate citations."""
    t0 = time.time()
    subquestions = split_compound_questions(question)

    if len(subquestions) > 1:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for subq in subquestions:
            for hit in retrieve(subq, k=k):
                key = (
                    str(hit.get("source", "")),
                    str(hit.get("section", "")),
                    str(hit.get("text", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(hit)
        chunks = sorted(merged, key=lambda c: float(c.get("score", 0.0)), reverse=True)[: max(k, len(merged))]
    else:
        chunks = retrieve(question, k=k)

    retrieved_sources = [{"source": c["source"], "score": c.get("score", 0.0)} for c in chunks]

    if should_abstain(chunks, threshold):
        return {"question": question, "answer": ABSTAIN_MESSAGE, "citations": [],
                "retrieved_sources": retrieved_sources, "abstained": True,
                "citation_check": {"used": [], "invalid": [], "has_citation": False},
                "latency_ms": int((time.time() - t0) * 1000)}

    context, sources = format_context(chunks)
    resp = oai.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}],
    )
    text = resp.choices[0].message.content.strip()

    # A model can ignore instructions and abstain in prose — respect that.
    if ABSTAIN_MESSAGE.lower() in text.lower():
        return {"question": question, "answer": ABSTAIN_MESSAGE, "citations": [],
                "retrieved_sources": retrieved_sources, "abstained": True,
                "citation_check": {"used": [], "invalid": [], "has_citation": False},
                "latency_ms": int((time.time() - t0) * 1000)}

    check = validate_citations(text, len(sources))
    cited = [s for s in sources if s["n"] in check["used"]]
    return {"question": question, "answer": text, "citations": cited,
            "retrieved_sources": retrieved_sources, "abstained": False,
            "citation_check": check, "latency_ms": int((time.time() - t0) * 1000)}


# ─── NEW 3 · Evaluation harness ─────────────────────────────────────────

def evaluate(questions: list[dict[str, Any]],
             answer_fn: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """Score an assistant over a question set. Each question is either
    answerable (has `expected_source`) or a should-abstain question
    (`should_abstain: true` or `expected_source: null`).

    Metrics (demonstrating grounded-response and hallucination KPIs):
      - retrieval hit rate    : answerable Qs whose expected source is in top-K
      - grounded & cited rate  : answerable Qs answered, citing the right source, no bad [n]
      - abstention accuracy    : should-abstain Qs the assistant correctly refused
      - hallucination rate     : should-abstain Qs the assistant answered anyway
    """
    rows: list[dict[str, Any]] = []
    n_answerable = 0
    n_abstain = 0
    retrieval_hits = 0
    grounded_hits = 0
    abstention_ok = 0
    hallucination_count = 0

    for q in questions:
        qid = q.get("id", q.get("question", "unknown"))
        prompt = q.get("question", "")
        expected_source = q.get("expected_source")
        should_abstain_q = bool(q.get("should_abstain")) or expected_source in (None, "", "ABSTAIN")

        r = answer_fn(prompt)
        retrieved_sources = r.get("retrieved_sources", [])
        source_names = {item.get("source") for item in retrieved_sources if item.get("source")}

        if should_abstain_q:
            n_abstain += 1
            abstained = bool(r.get("abstained"))
            ok = abstained
            if abstained:
                abstention_ok += 1
            else:
                hallucination_count += 1

            rows.append({
                "id": qid,
                "type": "abstain",
                "hit": False,
                "abstained": abstained,
                "grounded": False,
                "ok": ok,
            })
            continue

        n_answerable += 1
        hit = expected_source in source_names
        if hit:
            retrieval_hits += 1

        citation_check = r.get("citation_check", {}) or {}
        has_citation = bool(citation_check.get("has_citation"))
        invalid = bool(citation_check.get("invalid"))
        cited_sources = {c.get("source") for c in r.get("citations", []) if c.get("source")}
        grounded = (
            not bool(r.get("abstained"))
            and has_citation
            and not invalid
            and expected_source in cited_sources
        )
        if grounded:
            grounded_hits += 1

        rows.append({
            "id": qid,
            "type": "answer",
            "hit": hit,
            "abstained": bool(r.get("abstained")),
            "grounded": grounded,
            "ok": hit and grounded,
        })

    retrieval_hit_rate = (retrieval_hits / n_answerable) if n_answerable else 0.0
    grounded_rate = (grounded_hits / n_answerable) if n_answerable else 0.0
    abstention_accuracy = (abstention_ok / n_abstain) if n_abstain else 0.0
    hallucination_rate = (hallucination_count / n_abstain) if n_abstain else 0.0

    for row in rows:
        row["ok"] = bool(row["ok"]) if row["type"] == "answer" else bool(row["ok"])

    return {
        "rows": rows,
        "summary": {
            "n_answerable": n_answerable,
            "n_abstain": n_abstain,
            "retrieval_hit_rate": retrieval_hit_rate,
            "grounded_rate": grounded_rate,
            "abstention_accuracy": abstention_accuracy,
            "hallucination_rate": hallucination_rate,
        },
    }


def format_eval_table(result: dict[str, Any]) -> str:
    """Render the evaluation as the submittable table."""
    def cell(v):
        return {True: "✓", False: "✗"}.get(v, str(v))

    lines = ["", "  id                              type     hit   abstained  grounded  ok",
             "  " + "-" * 74]
    for r in result["rows"]:
        lines.append(f"  {r['id']:<31} {r['type']:<8} {cell(r['hit']):<5} "
                     f"{cell(r['abstained']):<10} {cell(r['grounded']):<9} {cell(r['ok'])}")
    s = result["summary"]
    lines += [
        "  " + "-" * 74, "",
        f"  Retrieval hit rate    : {s['retrieval_hit_rate']}   "
        f"({s['n_answerable']} answerable questions)",
        f"  Grounded & cited rate : {s['grounded_rate']}   (KPI: grounded response rate)",
        f"  Abstention accuracy   : {s['abstention_accuracy']}   "
        f"({s['n_abstain']} should-abstain questions)",
        f"  Hallucination rate    : {s['hallucination_rate']}   (KPI: hallucination rate)",
        "",
    ]
    return "\n".join(lines)


# ─── Commands ───────────────────────────────────────────────────────────

def cmd_ingest() -> None:
    _require_clients()
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        cs = chunk_document(doc)
        all_chunks.extend(cs)
        print(f"  {doc['source']}: {len(cs)} chunks")
    print(f"→ {len(all_chunks)} chunks · setting up collection · ingesting…")
    setup_collection()
    ingest_chunks(all_chunks)
    print("✓ Done. Try: python mp3_grounded_qa.py ask")


def _print_result(r: dict[str, Any]) -> None:
    print(f"\n{r['answer']}\n")
    if r["abstained"]:
        print("  (abstained — no supporting excerpt)")
    else:
        print("  Sources:")
        for c in r["citations"]:
            print(f"    [{c['n']}] {c['title']} — {c['section']}  (score {c['score']})")
        if r["citation_check"]["invalid"]:
            print(f"  ⚠ fabricated citation(s): {r['citation_check']['invalid']}")
    print(f"  Latency: {r['latency_ms']}ms\n")


def cmd_ask() -> None:
    _require_clients()
    print("Grounded Q&A over the Sherlock corpus. Empty line to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not q:
            return
        _print_result(answer(q))


def cmd_eval() -> None:
    _require_clients()
    path = DATA_DIR / "predefined_questions.jsonl"
    questions = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    learner = DATA_DIR / "learner_questions.jsonl"
    if learner.exists():
        extra = [json.loads(l) for l in learner.read_text().splitlines() if l.strip()]
        if extra and not str(extra[0]["question"]).startswith("Replace this"):
            questions += extra
    print(format_eval_table(evaluate(questions, answer)))


def cmd_selftest() -> None:
    """Offline checks of the MP3 logic — no OpenAI, no Qdrant, no keys."""
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'✓' if cond else '✗'} {name}")
        ok = ok and cond

    docs = load_corpus(CORPUS_DIR)
    check(f"load_corpus reads the corpus ({len(docs)} docs)", len(docs) >= 1)
    if docs:
        cs = chunk_document(docs[0])
        check(f"chunk_document splits doc 0 into chunks ({len(cs)})",
              len(cs) >= 1 and all({"source", "title", "section", "text"} <= c.keys() for c in cs))

    fake = [{"source": "a.txt", "title": "A", "section": "S1", "text": "alpha", "score": 0.9},
            {"source": "b.txt", "title": "B", "section": "S2", "text": "beta", "score": 0.5}]
    ctx, srcs = format_context(fake)
    check("format_context numbers chunks [1]/[2]", "[1]" in ctx and "[2]" in ctx and len(srcs) == 2)

    v_ok = validate_citations("The adder [1] came via the ventilator [2].", 2)
    v_bad = validate_citations("An outside fact [3].", 2)
    check("validate_citations accepts real markers", v_ok["used"] == [1, 2] and not v_ok["invalid"])
    check("validate_citations flags a fabricated marker", v_bad["invalid"] == [3])

    check("should_abstain: low score → abstain", should_abstain([{"score": 0.1}], 0.3) is True)
    check("should_abstain: good score → answer", should_abstain([{"score": 0.8}], 0.3) is False)
    check("should_abstain: no chunks → abstain", should_abstain([], 0.3) is True)

    # evaluate() with a deterministic mock answer_fn (no network)
    qs = [
        {"id": "ans_hit", "question": "q1", "expected_source": "01.txt"},
        {"id": "ans_miss", "question": "q2", "expected_source": "02.txt"},
        {"id": "abstain_ok", "question": "q3", "should_abstain": True},
    ]
    def mock(q):
        if q == "q1":  # correct source retrieved + cited
            return {"answer": "A [1]", "abstained": False, "retrieved_sources": [{"source": "01.txt", "score": 0.9}],
                    "citations": [{"source": "01.txt"}], "citation_check": {"used": [1], "invalid": [], "has_citation": True}}
        if q == "q2":  # wrong source retrieved → miss + not grounded
            return {"answer": "B [1]", "abstained": False, "retrieved_sources": [{"source": "99.txt", "score": 0.9}],
                    "citations": [{"source": "99.txt"}], "citation_check": {"used": [1], "invalid": [], "has_citation": True}}
        return {"answer": ABSTAIN_MESSAGE, "abstained": True, "retrieved_sources": [{"source": "x", "score": 0.1}],
                "citations": [], "citation_check": {"used": [], "invalid": [], "has_citation": False}}
    res = evaluate(qs, mock)
    s = res["summary"]
    check("evaluate: hit rate 1/2 answerable", s["retrieval_hit_rate"] == 0.5)
    check("evaluate: grounded rate 1/2 answerable", s["grounded_rate"] == 0.5)
    check("evaluate: abstention accuracy 1/1", s["abstention_accuracy"] == 1.0)
    check("evaluate: hallucination rate 0/1", s["hallucination_rate"] == 0.0)
    check("format_eval_table renders a table", "Retrieval hit rate" in format_eval_table(res))

    print("\n✓ selftest passed" if ok else "\n✗ selftest FAILED")
    sys.exit(0 if ok else 1)


def main() -> None:
    cmds = {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval, "selftest": cmd_selftest}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__); sys.exit(0 if len(sys.argv) < 2 else 1)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()

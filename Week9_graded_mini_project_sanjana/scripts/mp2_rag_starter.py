"""MP2 · Mini-RAG — Starter Template
====================================

You'll build a complete RAG pipeline over the Sherlock Holmes corpus in this
file. Fill in every TODO. The reference solution is ~250 lines, but yours can
be shorter or longer — what matters is that it works end-to-end.

Pipeline you're building:
    corpus/*.txt  →  chunks  →  embeddings  →  Qdrant
                                                  ↓
                              question  →  retrieve  →  answer + citations

Run sequence (once you've filled in the TODOs):
    pip install -r requirements.txt
    source .env                 # exports your OpenAI + Qdrant credentials
    python mp2_rag.py ingest    # builds the collection (run once)
    python mp2_rag.py ask       # interactive Q&A loop
    python mp2_rag.py validate  # runs against data/predefined_questions.jsonl

Tip: get the CORE pipeline working FIRST (Steps 1-7 below), THEN come back to
polish and add your 3 questions. Don't try to perfect each step before moving
on — you'll learn more from a rough end-to-end loop than a polished half.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

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
qdrant = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY or None,
)


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
   
    text = (doc.get("text") or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_len = 0
    section_name = "Overview"
    section_index = 1

    def flush_chunk() -> None:
        nonlocal section_name, section_index
        if not current:
            return

        chunk_text = "\n\n".join(current).strip()
        chunks.append(
            {
                "source": doc["source"],
                "title": doc["title"],
                "section": section_name,
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
            if current:
                flush_chunk()
            section_name = first_line
            current = []
            current_len = 0
            continue

        para_len = len(para)

        if current and current_len + 1 + para_len > TARGET_CHUNK_SIZE:
            flush_chunk()
            current = []
            current_len = 0

        current.append(para)
        current_len += para_len

    if current:
        flush_chunk()

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
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=k,
    )

    results: list[dict[str, Any]] = []
    for hit in hits:
        payload = hit.payload or {}
        payload["score"] = hit.score
        results.append(payload)

    return results


# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Cite the source (story title
+ section) in your answer."""


def answer(question: str, k: int = 3) -> dict[str, Any]:
    start = time.perf_counter()
    hits = retrieve(question, k=k)

    context_parts: list[str] = []
    citations: list[dict[str, str]] = []
    for hit in hits:
        title = hit.get("title", "Unknown")
        section = hit.get("section", "Overview")
        text = hit.get("text", "").strip()
        if not text:
            continue

        context_parts.append(f"[Source: {title} — {section}]\n{text}")
        citations.append({
            "source": title,
            "title": title,
            "section": section,
        })

    context = "\n\n---\n\n".join(context_parts)
    user_message = (
        f"Question: {question}\n\n"
        f"Relevant excerpts:\n{context}"
    )

    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )

    answer_text = response.choices[0].message.content.strip()
    latency_ms = round((time.perf_counter() - start) * 1000)

    return {
        "question": question,
        "answer": answer_text,
        "citations": citations,
        "latency_ms": latency_ms,
    }


# ─── Validation harness (provided — do not modify) ──────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

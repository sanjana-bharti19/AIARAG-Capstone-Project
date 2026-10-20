# MP3 Grounded Q&A

This project extends the Mini-RAG pipeline into a grounded, trustworthy assistant over the Sherlock Holmes corpus. The goal is to answer using only the retrieved excerpts, cite the supporting chunk, and abstain when the corpus does not contain the answer.

---

## What this project does

- loads the Sherlock corpus from `corpus/*.txt`
- chunks documents by paragraph/section with overlap
- embeds chunks into Qdrant
- retrieves relevant chunks for each question
- grounds the LLM answer in retrieved evidence
- validates citations and abstains when needed
- evaluates retrieval, grounding, and abstention quality

---

## Project files

- `scripts/mp3_grounded_qa.py` — main MP3 pipeline and CLI
- `config/settings.py` — config for API keys, collection, chunk size, and model settings
- `corpus/` — Sherlock Holmes story text files
- `data/predefined_questions.jsonl` — built-in validation set
- `data/learner_questions.jsonl` — custom learner questions
- `mp3_eval.txt` — output from the evaluation run
- `mp3_reflection.md` — reflection summary
- `.env` — local environment variables
- `requirements.txt` — Python dependencies

---

## Setup

1. Open the project folder.
2. Create or activate a virtual environment.
3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Add your credentials in `.env`:

```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

---

## Run the project

Ingest the corpus and build the Qdrant collection:

```bash
python3 scripts/mp3_grounded_qa.py ingest
```

Run interactive Q&A:

```bash
python3 scripts/mp3_grounded_qa.py ask
```

Run the built-in evaluation set:

```bash
python3 scripts/mp3_grounded_qa.py eval
```

Save the evaluation output to a file:

```bash
python3 scripts/mp3_grounded_qa.py eval > mp3_eval.txt
```

Run offline self-tests for the MP3 logic:

```bash
python3 scripts/mp3_grounded_qa.py selftest
```

---

## MP3 logic

The key pieces are:

- `format_context()` — number retrieved chunks for grounded prompts
- `validate_citations()` — reject fabricated or out-of-range citation markers
- `should_abstain()` — refuse if retrieval quality is below threshold
- `evaluate()` — measures retrieval hit rate, grounded rate, abstention accuracy, and hallucination rate

---

## Expected behavior

The system should:

- answer questions supported by the corpus with citations
- refuse unsupported or out-of-domain questions
- return valid `source` citations from retrieved chunks
- keep hallucination rate low through grounding and abstention checks

---

## Notes

This project is the MP3 version of a trustworthy RAG system: retrieval, grounding, abstention, citation control, and evaluation are all built into the same flow.

Git repo:

https://github.com/sanjana-bharti19/AIARAG-Capstone-Project

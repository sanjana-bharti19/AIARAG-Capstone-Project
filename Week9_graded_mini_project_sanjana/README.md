# MP2 Mini-RAG

This project builds a small retrieval-augmented generation (RAG) system over the Sherlock Holmes corpus. The workflow is:

- load the story files from `corpus/*.txt`
- split each story into meaningful chunks
- embed the chunks with OpenAI embeddings
- store the chunk vectors in Qdrant
- retrieve the most relevant chunks for a question
- send the retrieved excerpts to the LLM for a grounded answer
- return the answer with source citations

The goal is to answer questions about the Sherlock Holmes stories using only the supplied corpus, rather than relying on raw model memory.

---

## Project files

- `scripts/mp2_rag.py` — main Mini-RAG pipeline and CLI
- `config/settings.py` — shared app settings and environment values
- `corpus/` — source text files for the Sherlock Holmes stories
- `data/predefined_questions.jsonl` — evaluation questions for the built-in validation set
- `data/learner_questions.jsonl` — custom questions used for additional testing
- `.env` — local API and Qdrant settings
- `requirements.txt` — Python requirements for the project

---

## Setup

1. Open a terminal in the project folder.
2. Create or activate a Python environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Make sure your environment variables are configured. A `.env` file in the project root should include values like:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://openai.vocareum.com/v1
QDRANT_URL=
QDRANT_API_KEY=
```

---

## Run the project

From the project root:

```bash
python scripts/mp2_rag.py ingest
```

This loads all corpus files, chunks them, creates the Qdrant collection, and embeds all chunks into the vector database.

Then run the interactive question loop:

```bash
python scripts/mp2_rag.py ask
```

This accepts user questions, retrieves the closest matching chunks, and generates answers with citations.

Finally, run validation against the predefined evaluation set:

```bash
python scripts/mp2_rag.py validate
```

This runs the built-in evaluation harness and prints:

- which story was cited
- whether the expected source matched
- how many expected facts were present in the answer
- latency

---

## How the pipeline works

The project follows this flow:

```text
corpus/*.txt
    ↓
load_corpus()
    ↓
chunk_document()
    ↓
embed_texts()
    ↓
setup_collection() + ingest_chunks()
    ↓
retrieve()
    ↓
answer()
```

In plain terms:

- stories are read from the corpus directory
- each document is split into chunks
- each chunk is embedded into a vector
- vectors are stored in Qdrant
- a user question is embedded and compared against the stored vectors
- the closest matches are sent to the LLM as context
- the model responds using the retrieved excerpts and cites the matching source section

---

## Expected output

After running the pipeline, you should see:

- a Qdrant collection created for the corpus
- interactive question answers in the CLI
- validation output showing source matches and fact matches
- source citations in the LLM responses

---

## Troubleshooting

### Module import error

Run the script from the project root, not from inside a nested folder.

### Missing API key or Qdrant connection

Check the `.env` file and confirm that OpenAI and Qdrant are reachable.

### Validation failures

Make sure the `source` stored in the citation matches the exact filename expected by the validator, such as `01_red_headed_league.txt`.

### Qdrant client error

If you see an error related to `search()`, make sure the installed `qdrant-client` version matches the API used by the code. The newer client expects `query_points()` instead of the older `search()` style.

---

## Notes

This project is a good example of a lightweight but complete RAG system: chunking, vector storage, retrieval, prompt construction, answer generation, and evaluation are all implemented in one script.

Git repo link:

https://github.com/sanjana-bharti19/AIARAG-Capstone-Project

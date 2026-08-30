import os
import argparse

files = {
    # --- 1. CORE RAG MODULE ---
    "src/__init__.py": '"""Source Code Root. This directory contains the entire application source code, separated into functional modules (rag, eval, pipeline)."""\n',
    "src/rag/__init__.py": '"""\nRetrieval-Augmented Generation (RAG) Module Initialization.\nThis dedicated package encapsulates the entire end-to-end RAG workflow.\nIt is organized into distinct sub-modules handling chunking (text splitting),\nembeddings (vectorization), vector store (database interfacing), retrieval (search),\nand generation (LLM synthesis). This structure allows for independent testing and\nswapping of components (e.g., changing the vector database without affecting the generator).\n"""\n',
    "src/rag/chunking.py": '"""\nText Splitting and Chunking Strategies.\nThis file contains algorithms for breaking down large documents into smaller, semantically\nmeaningful chunks before embedding. It includes implementations for:\n- Recursive Character Text Splitting: Splitting by paragraphs, sentences, and words.\n- Semantic Chunking: Grouping text based on embedding similarity to maintain context.\n- Token-based Splitting: Ensuring chunks fit strictly within LLM context windows.\nProper chunking is critical for effective retrieval and minimizing noise in the context.\n\nFunctions:\n- chunk_by_characters(text, chunk_size, overlap): Recursive splitting.\n- chunk_by_tokens(text, max_tokens): Token-aware splitting.\n- semantic_chunk(text, embedding_model): Groups sentences by semantic similarity.\n"""\n',
    "src/rag/embeddings.py": '"""\nEmbedding Model Wrappers.\nThis module acts as an abstraction layer over various embedding models (e.g., OpenAI\'s\ntext-embedding-ada-002, HuggingFace sentence-transformers, Cohere).\nIt provides a unified interface for converting text chunks into dense vector representations.\nIt also includes logic for batch processing, handling rate limits, and caching embeddings\nto avoid redundant API calls and reduce costs during indexing.\n\nClasses:\n- BaseEmbeddingModel: Abstract base class for embedding models.\n- OpenAIEmbeddings: Implementation for OpenAI API.\n- HuggingFaceEmbeddings: Local embedding generation using sentence-transformers.\n\nFunctions:\n- get_embedding(text): Returns the embedding vector for a single string.\n- get_embeddings_batch(texts): Returns a list of vectors for a batch of strings.\n"""\n',
    "src/rag/vector_store.py": '"""\nVector Database Interface.\nThis script manages the connection and interactions with the underlying vector database\n(such as ChromaDB, FAISS, Qdrant, or Pinecone).\nIt provides standard methods for initializing the database, inserting document chunks and\ntheir corresponding embeddings, deleting old records, and persisting the index to disk.\nThis abstraction ensures the core RAG logic remains decoupled from specific database vendors.\n\nClasses:\n- VectorStore: Abstract interface for vector databases.\n- ChromaDBStore: Implementation using local ChromaDB.\n- PineconeStore: Implementation using Pinecone cloud vector DB.\n\nMethods:\n- add_documents(documents, embeddings, metadata): Inserts records.\n- search(query_embedding, top_k): Returns top_k similar documents.\n- delete(document_ids): Removes documents from the index.\n"""\n',
    "src/rag/retriever.py": '"""\nCore Retrieval Logic.\nThis module is responsible for taking a user query, embedding it, and fetching the most\nrelevant document chunks from the vector store.\nIt implements advanced retrieval techniques including:\n- Top-K Similarity Search (Cosine similarity, L2 distance)\n- Hybrid Search (combining dense vector search with sparse keyword search like BM25)\n- Query Re-writing and Expansion to improve recall.\n- Re-ranking algorithms (e.g., using Cross-Encoders) to refine the fetched results.\n\nClasses:\n- BaseRetriever: Abstract interface.\n- DenseRetriever: Standard vector similarity search.\n- HybridRetriever: Combines sparse (BM25) and dense search.\n\nMethods:\n- retrieve(query, top_k=5): Executes the full retrieval pipeline including re-ranking.\n"""\n',
    "src/rag/generator.py": '"""\nPrompt Construction and LLM Answer Generation.\nThis file bridges the gap between retrieved documents and the final user answer.\nIt contains the prompt templates necessary to instruct the LLM on how to use the provided\ncontext. It handles the injection of context and the user query into the prompt, makes\nthe API call to the generative model (e.g., GPT-4, Claude), streams the response if necessary,\nand enforces constraints (like \'answer only using the provided text\').\n\nClasses:\n- LLMGenerator: Manages interactions with LLM APIs.\n\nMethods:\n- generate_answer(query, retrieved_context): Builds the prompt and calls the LLM.\n- generate_stream(query, retrieved_context): Yields the answer token by token for UI streaming.\n"""\n',
    "src/rag/prompts/system_prompt.txt": 'You are an expert financial assistant. Your goal is to answer the user\'s question accurately based ONLY on the provided context.\nIf the answer is not in the context, say "I cannot answer this based on the provided documents."\nNever hallucinate or make up information outside the provided text.\nMaintain a professional, objective tone.\n',
    "src/rag/prompts/user_prompt.txt": 'Use the following pieces of context to answer the question at the end.\n\nContext:\n{context}\n\nQuestion: {query}\n\nHelpful Answer:\n',

    # --- 2. EVALUATION MODULE ---
    "src/eval/__init__.py": '"""\nEvaluation Module Initialization.\nThis module contains the core components for evaluating LLM-generated responses.\nIt exposes standard interfaces for judging response quality, managing golden datasets,\nrunning pairwise comparisons between models, and creating AI critics.\n"""\n',
    "src/eval/judge.py": '"""\nEvaluation Judge Logic.\nThis file implements the `LLMJudge` classes responsible for scoring and grading\ngenerated answers against provided rubrics or expected outcomes.\nIt includes logic for prompt formatting for the judge LLM, parsing JSON or structured\noutputs from the judge, handling retries on malformed outputs, and normalizing scores.\n\nClasses:\n- LLMJudge: Uses an LLM to score responses based on criteria like relevance and faithfulness.\n- ExactMatchJudge: Simple string comparison for factual questions.\n\nMethods:\n- evaluate(query, context, generated_answer, expected_answer): Returns a numerical score and reasoning.\n"""\n',
    "src/eval/golden.py": '"""\nGolden Set Evaluation Logic.\nThis script provides utilities for loading, parsing, and running evaluations against\na \'Golden Set\' - a curated dataset of inputs and known-good outputs.\nIt includes functions to iterate over test cases, invoke the generation pipeline,\ncompare the generated result against the expected answer, and aggregate metrics.\n\nFunctions:\n- load_golden_set(filepath): Parses the JSONL dataset.\n- run_evaluation(golden_dataset, rag_pipeline, judge): Executes the full benchmark run.\n- aggregate_metrics(results): Computes average scores, latency, and failure rates.\n"""\n',
    "src/eval/pairwise.py": '"""\nPairwise Evaluation Logic.\nThis module is dedicated to A/B testing two different model responses or two different\nRAG pipeline configurations against each other. It handles positional bias swapping.\n\nFunctions:\n- run_ab_test(model_a_output, model_b_output, query): Uses a judge LLM to pick the better response.\n- calculate_win_rate(ab_results): Computes statistical win/loss/tie ratios.\n"""\n',
    "src/eval/critic_creator.py": '"""\nCritic Creator Logic.\nThis experimental module focuses on generating specialized \'critic\' prompts or dynamically\ncreating specialized evaluation agents to check for specific flaws (e.g., hallucination, tone).\n\nFunctions:\n- generate_critic_prompt(flaw_type): Returns a system prompt tailored for detecting a specific flaw.\n- critique_response(response, critic_prompt): Applies the critic to a generated answer.\n"""\n',

    # --- 2.5 SCHEMAS ---
    "src/schemas/__init__.py": '"""Schemas Module. Contains Pydantic models for data validation and API input/output structures."""\n',
    "src/schemas/api.py": '"""\nAPI Schemas.\nDefines the Pydantic models for FastAPI request and response bodies.\n\nClasses:\n- QueryRequest: User query and optional filters.\n- QueryResponse: Generated answer, retrieved context, and metadata.\n- UploadResponse: Status of document ingestion.\n"""\nfrom pydantic import BaseModel\nfrom typing import List, Optional\n\nclass QueryRequest(BaseModel):\n    query: str\n    top_k: Optional[int] = 5\n\nclass QueryResponse(BaseModel):\n    answer: str\n    sources: List[str]\n',
    "src/schemas/rag.py": '"""\nRAG Core Schemas.\nDefines internal data structures passed between chunker, retriever, and generator.\n\nClasses:\n- DocumentChunk: Represents a single piece of text with metadata and an optional embedding.\n- RetrievalResult: Represents a chunk fetched from the vector store with a similarity score.\n"""\nfrom pydantic import BaseModel\nfrom typing import Dict, Any, List, Optional\n\nclass DocumentChunk(BaseModel):\n    text: str\n    metadata: Dict[str, Any] = {}\n    embedding: Optional[List[float]] = None\n',

    # --- 3. PIPELINE & STORAGE ---
    "src/pipeline/__init__.py": '"""Pipeline Module. Handles orchestration, data logging, and experiment tracking."""\n',
    "src/pipeline/store.py": '"""\nSQLite Experiment Tracking.\nThis module manages the persistence layer for all evaluation runs. It establishes\na connection to a local SQLite database to log inputs, outputs, judge scores,\nmodel configurations, and latency metrics for every single evaluation.\n\nClasses:\n- ExperimentTracker: Manages DB connections and table schemas.\n\nMethods:\n- log_run(config, metrics): Inserts a new experiment run.\n- log_evaluation_case(run_id, query, response, score): Logs individual test case results.\n- export_to_csv(): Dumps DB contents for external analysis.\n"""\n',

    # --- 4. CONFIGURATION ---
    "config/__init__.py": '"""Config module for environment variables and system-wide settings."""\n',
    "config/settings.py": '"""\nConfiguration Settings.\nThis module uses Pydantic Settings to load and validate environment variables from the .env file.\nIt centralizes all configuration management, ensuring type safety for API keys,\ndatabase paths, and other system-wide constants.\n"""\nimport os\nfrom pydantic_settings import BaseSettings, SettingsConfigDict\n\n# Get the absolute path to the root of the project (going up one level from config folder)\nBASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\nENV_FILE_PATH = os.path.join(BASE_DIR, ".env")\n\nclass Settings(BaseSettings):\n    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH, env_file_encoding="utf-8", extra="ignore")\n\n    OPENAI_API_KEY: str = ""\n    HUGGINGFACE_API_KEY: str = ""\n    DB_PATH: str = "data/db/eval_runs.db"\n    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-ada-002"\n    CHUNK_SIZE: int = 500\n    CHUNK_OVERLAP: int = 50\n\nsettings = Settings()\n',
    "config/logging_config.py": '"""\nStandardized Logging Configuration.\nSets up JSON or standard formatting for logs across the entire RAG pipeline.\nEnsures that errors, warnings, and evaluation metrics are properly piped to standard out\nor log files for production monitoring.\n\nFunctions:\n- setup_logger(name): Returns a pre-configured Python logger instance.\n"""\n',

    # --- 5. UI & API (FASTAPI) ---
    "ui/__init__.py": '"""Streamlit UI Module."""\n',
    "ui/app.py": '"""\nStreamlit Web Interface.\nThis script launches a user-friendly chat interface for the RAG system using Streamlit.\nIt allows users to upload documents, ask questions, and visualize the retrieved chunks\nalongside the AI-generated answer. It connects directly to the `src.rag` backend components or the FastAPI backend.\n\nFeatures:\n- Sidebar for configuration (model selection, chunk size).\n- File uploader (PDF, TXT) for dynamic knowledge base ingestion.\n- Chat interface with streaming text generation.\n- Expandable sections showing retrieved context sources.\n\nTo run: `streamlit run ui/app.py`\n"""\n',
    "ui/components/__init__.py": '"""Streamlit UI Components. Contains reusable UI elements like chat bubbles, file uploaders, and metric displays."""\n',
    "api/__init__.py": '"""FastAPI Application Module."""\n',
    "api/main.py": '"""\nFastAPI main application.\nExposes endpoints for the RAG pipeline to be consumed by other services or the Streamlit UI.\n\nEndpoints:\n- GET /health: Basic health check.\n- POST /upload: Upload documents for indexing.\n- POST /query: Submit a question and get a generated answer with sources.\n- GET /metrics: Retrieve system performance metrics.\n\nRun with: `uvicorn api.main:app --reload`\n"""\nfrom fastapi import FastAPI\n\napp = FastAPI(title="RAG API")\n\n@app.get("/health")\ndef health_check():\n    return {"status": "ok"}\n',

    # --- 6. SCRIPTS ---
    "scripts/__init__.py": '"""Scripts Module. Contains executable Python scripts for evaluations and data processing."""\n',
    "scripts/run_eval.py": '"""\nStandard Evaluation Runner (Step 3).\nMain entry point for running standard, single-metric evaluations. It loads a dataset,\ninstantiates the specified evaluation judge, processes outputs, and logs final scores.\n\nUsage:\n`python scripts/run_eval.py --dataset data/golden_set.jsonl --judge llm_judge`\n"""\n',
    "scripts/run_rag_eval.py": '"""\nEnd-to-End RAG Pipeline Orchestrator.\nRuns the entire RAG system against a golden set to measure overall system accuracy.\nTests the synergy of chunking, retrieval, and generation.\n\nUsage:\n`python scripts/run_rag_eval.py --config config/settings.py`\n"""\n',
    "scripts/run_pairwise.py": '"""\nPairwise Evaluation Script (Step 4).\nA/B testing tool that runs two different RAG configurations (e.g., BM25 vs Dense retrieval) and uses an LLM judge to determine the winner for a set of queries.\n\nUsage:\n`python scripts/run_pairwise.py --model_a gpt-3.5-turbo --model_b gpt-4`\n"""\n',
    "scripts/run_critic_creator.py": '"""\nCritic Creator Execution Script (Step 5).\nGenerates synthetic adversarial questions and evaluates the RAG pipeline\'s robustness against them.\n"""\n',

    # --- 7. BINS (SHELL SCRIPTS) ---
    "bin/setup_env.sh": '#!/bin/bash\n# Shell script to install requirements (via pyproject.toml) using uv, and run Docker deployment.\n# Run via: bash bin/setup_env.sh\nif [ ! -f ".env" ]; then\n    echo "Creating .env from .env.example..."\n    cp .env.example .env\nfi\nif [ ! -d ".venv" ]; then\n    echo "Syncing project dependencies and creating uv.lock..."\n    uv sync\nelse\n    echo "Virtual environment already exists, skipping sync."\nfi\necho "Activating virtual environment..."\nsource .venv/bin/activate\necho "Starting Docker deployment..."\ndocker-compose up -d --build\n',
    "bin/setup_env.ps1": '# PowerShell script to install requirements (via pyproject.toml) using uv, and run Docker deployment.\n# Run via: .\\bin\\setup_env.ps1\nif (-Not (Test-Path ".env")) {\n    Write-Host "Creating .env from .env.example..."\n    Copy-Item ".env.example" -Destination ".env"\n}\nif (-Not (Test-Path ".venv")) {\n    Write-Host "Syncing project dependencies and creating uv.lock..."\n    uv sync\n} else {\n    Write-Host "Virtual environment already exists, skipping sync."\n}\nWrite-Host "Activating virtual environment..."\n.\\.venv\\Scripts\\Activate.ps1\nWrite-Host "Starting Docker deployment..."\ndocker-compose up -d --build\n',
    "bin/run_app.sh": '#!/bin/bash\n# Shell script to start the FastAPI server and Streamlit UI concurrently.\nsource .venv/bin/activate\nuvicorn api.main:app --reload &\nstreamlit run ui/app.py\n',
    "bin/run_app.ps1": '# PowerShell script to start the FastAPI server and Streamlit UI concurrently.\n.\\.venv\\Scripts\\Activate.ps1\nStart-Process "uvicorn" -ArgumentList "api.main:app --reload"\nstreamlit run ui\\app.py\n',

    # --- 8. DOCKER & DEPLOYMENT ---
    "Dockerfile": '# Production Dockerfile\nFROM python:3.11-slim\nWORKDIR /app\nCOPY pyproject.toml .\nRUN pip install .\nCOPY . .\nCMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]\n',
    "docker-compose.yml": '# Docker Compose configuration\n# Spins up the FastAPI backend, RAG UI, and an optional vector database service locally.\nversion: "3.9"\nservices:\n  rag-api:\n    build: .\n    ports:\n      - "8000:8000"\n    env_file:\n      - .env\n  rag-ui:\n    build: .\n    command: streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0\n    ports:\n      - "8501:8501"\n    env_file:\n      - .env\n    depends_on:\n      - rag-api\n',

    # --- 9. DATA & MARKDOWN STORAGE ---
    "data/golden_set.jsonl": '{"query": "What is the capital of France?", "expected_answer": "Paris", "context": "France is a country in Europe. Its capital is Paris."}\n{"_comment": "Learners fill in 20 RAG test cases (query, context, expected answer)."}\n',
    "data/golden_set_full.jsonl": '{"_comment": "Worked reference dataset containing extensive real-world examples."}\n',
    "data/raw_markdown/.gitkeep": 'This directory stores raw Markdown (.md) or text files that act as the source knowledge base for the RAG pipeline. The chunker will read files from this directory to populate the vector store. Ensure documents are clean and UTF-8 encoded.\n',
    "data/uploads/.gitkeep": 'This directory temporarily stores user-uploaded files (like PDFs or Text documents) from the Streamlit UI before they are processed by the ingestion pipeline.\n',
    "data/db/.gitkeep": 'This directory stores local database files such as SQLite files (for evaluation tracking) and ChromaDB persistent storage directories (for vector embeddings).\n',
    "data/temp/.gitkeep": 'This directory is for intermediate processing artifacts, temporary JSON dumps, and short-lived caching files during ingestion.\n',

    # --- 10. PROJECT ROOT FILES ---
    "notebooks/.gitkeep": 'This directory stores Jupyter notebooks (.ipynb) used for exploratory data analysis, pipeline prototyping, and visualization.\n',
    ".env.example": '# Copy this file to .env and fill in the values\nOPENAI_API_KEY=your_openai_key_here\nHUGGINGFACE_API_KEY=your_hf_key_here\nDB_PATH=data/db/eval_runs.db\nCHUNK_SIZE=500\nCHUNK_OVERLAP=50\n',
    ".gitignore": '.env\n.venv\n__pycache__/\n*.sqlite3\n*.db\neval_runs.db\n.DS_Store\n\n# Ignore runtime data but keep folders\ndata/uploads/*\n!data/uploads/.gitkeep\ndata/db/*\n!data/db/.gitkeep\ndata/temp/*\n!data/temp/.gitkeep\n',
    "README.md": '''# Capstone: RAG Pipeline

This repository contains the production-ready structure for an end-to-end Retrieval Augmented Generation (RAG) system.

## 📂 Detailed Folder Structure & Contents

### `src/` (Source Code)
The main codebase, containing all core logic for the RAG application.
- **`rag/`**: The core RAG engine. 
  - `chunking.py`: Logic for splitting documents (recursive, semantic, token-based).
  - `embeddings.py`: Wrappers for embedding models (OpenAI, HuggingFace).
  - `vector_store.py`: Abstraction layer for interacting with vector databases (ChromaDB, Pinecone).
  - `retriever.py`: Logic for finding relevant chunks (dense, sparse, hybrid search).
  - `generator.py`: Prompt construction and interaction with the LLM to generate the final answer.
  - `prompts/`: Contains `.txt` files with system and user prompt templates.
- **`eval/`**: Evaluation logic to score the RAG pipeline.
  - `judge.py`: LLM-as-a-judge implementation to grade answers.
  - `golden.py`: Utilities for evaluating against known golden datasets.
  - `pairwise.py`: A/B testing logic to compare two models.
  - `critic_creator.py`: Experimental agents designed to find flaws in generated text.
- **`pipeline/`**: Data persistence and orchestration.
  - `store.py`: SQLite-based experiment tracking to log queries, answers, and scores.
- **`schemas/`**: Pydantic models defining input/output structures for the API and internal data flow.
  - `api.py`: FastAPI request and response models.
  - `rag.py`: Internal structures like `DocumentChunk` and `RetrievalResult`.

### `api/` (Backend API)
- `main.py`: The FastAPI application exposing the RAG pipeline as RESTful endpoints (`/upload`, `/query`). Serves as the backend for the UI.

### `ui/` (User Interface)
- `app.py`: The Streamlit frontend providing a chat interface, document upload, and configuration sidebars.
- `components/`: Reusable Streamlit UI components (e.g., custom chat bubbles).

### `data/` (Datasets & Storage)
- **`uploads/`**: Temporary storage for files uploaded via the UI.
- **`raw_markdown/`**: Source knowledge base documents (`.md`, `.txt`) awaiting ingestion.
- **`db/`**: Persistent local storage, including SQLite databases for experiment tracking and ChromaDB vector indexes.
- **`temp/`**: Short-lived artifacts and JSON dumps used during processing.
- `golden_set.jsonl`: Benchmark datasets used for automated evaluation.

### `notebooks/`
- Jupyter notebooks (`.ipynb`) used for exploratory data analysis, testing chunking strategies, and visualization.

### `scripts/` (Execution)
Command-line Python scripts used for running evaluations and batch tasks.
- `run_eval.py`: Run standard evaluations.
- `run_rag_eval.py`: Orchestrate end-to-end pipeline evaluations.
- `run_pairwise.py`: Run A/B comparisons.

### `bin/` (Shell Utilities)
- `setup_env.sh` / `.ps1`: Scripts to install dependencies (via `pyproject.toml`) and create virtual environments.
- `run_app.sh` / `.ps1`: Scripts to concurrently launch the FastAPI server and Streamlit UI.

### `tests/`
Unit tests isolating specific components.
- `test_rag.py`: Tests for chunking, embedding, and retrieval.
- `test_judge.py`, `test_api.py`, `test_ui.py`: Asserts correctness for evaluation, API endpoints, and UI logic.

### `docs/`
Documentation, templates, and architecture decisions.
- `adr/`: Architecture Decision Records.
- `dr/`: Design Review materials.

### `config/`
- `settings.py`: Centralized configuration mapping to `.env` variables using `pydantic-settings` (API keys, DB paths, embedding models).
- `logging_config.py`: Standardized logging setup for the application.
''',
    "pyproject.toml": '''[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "rag-pipeline"
version = "0.1.0"
description = "Production RAG Pipeline"
dependencies = [
    "fastapi",
    "uvicorn",
    "streamlit",
    "openai",
    "langchain",
    "chromadb",
    "python-dotenv",
    "pydantic",
    "pydantic-settings",
    "pytest",
    "requests"
]

[tool.black]
line-length = 88

[tool.isort]
profile = "black"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
''',
    "Makefile": 'install:\n\tpip install -e .\n\ntest:\n\tpytest tests/\n\nrun-ui:\n\tstreamlit run ui/app.py\n\nrun-api:\n\tuvicorn api.main:app --reload\n\nformat:\n\tblack src tests api\n\tisort src tests api\n',

    # --- 11. DOCS ---
    "docs/stakeholder-map-template.md": "# Stakeholder Map Template\nUse this to track key stakeholders (users, developers, product managers) and their needs.\n",
    "docs/golden-set-notes-template.md": "# Golden Set Notes Template\nDocument the rationale behind each golden set question, the sources used, and why the expected answer is correct.\n",
    "docs/eval-run-001-template.md": "# Evaluation Run 001 Summary Template\nRecord the results, metrics, and insights from the first major evaluation run.\n",
    "docs/prompt-pairwise-001-template.md": "# Prompt Pairwise 001 Template\nTrack the prompts used, the models compared, and the statistical outcome of A/B testing.\n",
    "docs/critic-creator-trace-template.md": "# Critic Creator Trace Template\nLog the interactions and critiques generated by the AI critic agents.\n",
    "docs/design-review-rubric.md": "# Design Review Rubric\nCriteria for reviewing the architecture, code quality, and security of the RAG pipeline.\n",
    "docs/adr/0001-capstone-framing.md": "# ADR 0001: Capstone Framing\nContext, decision, and consequences regarding the overall scope and technology stack of the project.\n",
    "docs/dr/dr1-summary-template.md": "# DR1 (Design Review 1) Summary Template\nExecutive summary of the first design review meeting.\n",
    "docs/dr/dr1-peer-review-form.md": "# DR1 Peer Review Form\nForm for peers to provide structured feedback on the initial design.\n",
    "docs/dr/dr1-facilitator-script.md": "# DR1 Facilitator Script\nStep-by-step guide for facilitating the design review meeting.\n",

    # --- 12. TESTS ---
    "tests/__init__.py": '"""Tests Module. Contains all unit and integration tests."""\n',
    "tests/test_judge.py": '"""Unit Tests for Evaluation Judge. Verifies grading accuracy and prompt formats."""\n',
    "tests/test_pairwise.py": '"""Unit Tests for Pairwise Evaluation. Tests positional bias handling."""\n',
    "tests/test_golden_loader.py": '"""Unit Tests for Golden Dataset Loader. Verifies JSONL parsing and error handling."""\n',
    "tests/test_rag.py": '"""Unit Tests for RAG Components (Chunking, Embedding, Retrieval). Verifies algorithm correctness."""\n',
    "tests/test_api.py": '"""Unit Tests for FastAPI Application. Tests endpoint routing and status codes."""\n',
    "tests/test_ui.py": '"""Unit Tests for Streamlit UI. Verifies component rendering and state management."""\n',

    # --- 13. REFERENCE (EMPTY FOLDERS) ---
    "reference/src/eval/.gitkeep": "This directory contains reference implementations for advanced evaluation strategies. Keep this file to track the empty folder in Git.\n"
}

def create_structure():
    parser = argparse.ArgumentParser(description="Generate RAG folder structure.")
    parser.add_argument("project_name", nargs="?", default="starter", help="The name of the project directory (default: starter)")
    args = parser.parse_args()
    
    project_dir = args.project_name
    print(f"Generating complete production-ready RAG folder structure in '{project_dir}/'...")
    
    # Track created files and folders for output
    created_count = 0
    
    for path, content in files.items():
        # Get absolute path relative to current working directory + project_dir
        full_path = os.path.abspath(os.path.join(project_dir, path))
        
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write the vast descriptions to the file
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        # Make shell scripts executable
        if full_path.endswith('.sh') or full_path.endswith('.ps1'):
            import stat
            st = os.stat(full_path)
            os.chmod(full_path, st.st_mode | stat.S_IEXEC)
            
        created_count += 1

    print(f"\\n✅ Successfully generated {created_count} files across the directory tree in '{project_dir}/'.")
    print("✅ Configuration, FastAPI, UI, Docker, Data, Shell Scripts, and tests have been correctly placed.")
    print("✅ Emtpy folders (like ui/components and data/raw_markdown) are tracked via .gitkeep with descriptions.")

if __name__ == "__main__":
    create_structure()

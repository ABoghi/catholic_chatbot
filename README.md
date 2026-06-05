# Catholic Chatbot

A Python chatbot that answers questions from PDFs and text files using:
- `chromadb` for vector search
- `sentence-transformers` for embeddings
- `ollama` for a free local LLM

## Setup

1. Create and activate a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Ollama and a free model on your machine:
- Ollama: https://ollama.com/docs
- Example free model: `ollama pull llama2`

4. Create a `.env` file if you want to override defaults:

```env
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
OLLAMA_MODEL=llama2
CHROMA_PERSIST_DIR=./chromadb
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Usage

### Ingest documents

Place your `.pdf`, `.txt`, or `.md` files in a folder, for example `backend/data/`, and run:

```bash
python backend/main.py ingest --source backend/data
```

This will:
- read all supported files recursively
- split them into overlapping chunks
- embed them locally
- save vectors into a persistent Chroma database
- avoid duplicate ingestion and update changed files automatically

#### PDF metadata and tier support

For PDFs, use a sidecar metadata file in the same folder named `metadata.json`, `metadata.yml`, or `metadata.yaml`.

Example `metadata.yml`:

```yaml
documents:
  my_document.pdf:
    title: "Gospel of Matthew"
    authors:
      - "Saint Matthew"
    publisher: "Holy See"
    year: 2025
    category: "Scripture"
    tier: 1
  papal_encyclical.pdf:
    title: "Veritatis Splendor"
    authors:
      - "Pope John Paul II"
    publisher: "Vatican Press"
    year: 1993
    category: "Magisterium"
    tier: 2
```

Example `metadata.json`:

```json
{
  "documents": {
    "my_document.pdf": {
      "title": "Gospel of Matthew",
      "authors": ["Saint Matthew"],
      "publisher": "Holy See",
      "year": 2025,
      "category": "Scripture",
      "tier": 1
    }
  }
}
```

This sidecar file allows the ingestion pipeline to attach `tier`, `category`, `author`, and other metadata to PDF documents.

### Query the chatbot

```bash
python main.py query "What does the Bible say about forgiveness?"
```

### Start an interactive chat session

```bash
python main.py interactive
```

### List ingested files

```bash
python main.py list-docs
```

## How periodic updates work

The vector database is **persistent** — documents are stored permanently in `./chromadb/` and remain in memory between sessions. You **do not** need to re-add documents every time.

Simply re-run the ingestion command with the folder containing your documents:

```bash
python backend/main.py ingest --source backend/data
```

The ingestion process is smart:
- **New files**: added to the vector store
- **Modified files**: detected via file hash and refreshed (old vectors removed, new ones added)
- **Unchanged files**: skipped entirely (detected via hash comparison)
- **Deleted files**: removed from the vector store if they no longer exist in the source folder

**Example workflow:**
1. First run: `python backend/main.py ingest --source backend/data` → 10 documents added
2. Add 3 new documents to `backend/data/` folder
3. Second run: `python backend/main.py ingest --source backend/data` → adds the 3 new documents (keeps the original 10)
4. Edit one existing document
5. Third run: `python backend/main.py ingest --source backend/data` → updates the edited document, keeps all others

The database persists across restarts — you only need to run the web app or chatbot commands to use the knowledge.

## Web UI

A simple web frontend is included using FastAPI. It provides a main page where users can ask questions, a donations form, an about/contact page, a disclaimer, a documents listing, a technical page, and statistics.

- Start the web app (development):

```bash
source .venv/bin/activate
uvicorn backend.webapp:app --reload --host 127.0.0.1 --port 8000
```

If you prefer not to activate the venv first:

```bash
./.venv/bin/uvicorn backend.webapp:app --reload --host 127.0.0.1 --port 8000
```

- Open: `http://127.0.0.1:8000`

- Start the Streamlit frontend:

```bash
streamlit run streamlit_app.py
```

- Key pages/endpoints:
	- `/` — main page (ask questions + donate)
	- `/documents` — table of ingested documents with `title` and `inserted_at`
	- `/technical` — technical overview of the stack
	- `/stats` — question and donation statistics
	- `/about` and `/contact` — contact form
	- `/disclaimer` — legal disclaimer

- Data storage:
	- Web tracking (queries, donations, contacts) is stored in `web_data.db` in the project root.
	- The `/documents` page reads `document_metadata.json` produced by the ingestion process, so it reflects newly ingested documents immediately after you run the ingest command (no server restart required).

- Backend LLM independence:
	- The backend now uses LangChain wrappers for Ollama and local HuggingFace models when available.
	- This makes the LLM provider selectable by `LLM_PROVIDER` and `OLLAMA_MODEL` while keeping the same chat prompt flow.

- Document metadata:
	- Each ingested file stores `title` (extracted from front-matter or the first non-empty line) and `inserted_at` (first ingestion timestamp).
	- Re-running ingestion preserves `inserted_at` for existing files and updates the `title` if the file content changes.

- Optional: auto-ingest on filesystem changes can be added (e.g., with `watchdog`) — ask me if you want this.

## Notes

- Keep `ollama` running locally or make sure the Ollama API is available.
- The default embedding model is `all-MiniLM-L6-v2`, which is free and works offline.
- The default vector store persistence directory is `./chromadb`.

**Switching LLMs**

- You can select the LLM provider and model using environment variables or CLI flags.
- Environment variables:

```env
LLM_PROVIDER=ollama      # or 'local' or 'transformers'
OLLAMA_MODEL=llama2      # model name used by Ollama or default model name
```

- CLI example (query using a local HF model):

```bash
python main.py query "What is forgiveness?" --llm-provider local --llm-model phi-4-mini
```

- Provider options:
	- `ollama`: uses the Ollama HTTP API or CLI (default)
	- `local` / `transformers`: uses a local HuggingFace `transformers` pipeline for text generation

- Note: using `local` models requires installing `transformers` and `torch` and downloading the model weights. See `requirements.txt` for optional dependencies.


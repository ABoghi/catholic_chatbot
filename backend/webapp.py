from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sqlite3
from datetime import datetime
import json
import os
from pathlib import Path

try:
    from .document_store import DocumentStore
    from .llm_clients import create_llm_client
    from .chatbot import Chatbot
    from .config import DEFAULT_CHROMA_DIR, DEFAULT_DATA_DIR
except ImportError:
    from document_store import DocumentStore
    from llm_clients import create_llm_client
    from chatbot import Chatbot
    from config import DEFAULT_CHROMA_DIR, DEFAULT_DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DB = PROJECT_ROOT / "web_data.db"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI()
templates = Jinja2Templates(directory=str(FRONTEND_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


def init_db():
    conn = sqlite3.connect(APP_DB)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            question TEXT PRIMARY KEY,
            count INTEGER,
            last_asked TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS donations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            amount REAL,
            date TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            message TEXT,
            date TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def record_query(question: str):
    conn = sqlite3.connect(APP_DB)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute("SELECT count FROM queries WHERE question = ?", (question,))
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE queries SET count = count + 1, last_asked = ? WHERE question = ?", (now, question))
    else:
        cur.execute("INSERT INTO queries (question, count, last_asked) VALUES (?, ?, ?)", (question, 1, now))
    conn.commit()
    conn.close()


def record_donation(name: str, amount: float):
    conn = sqlite3.connect(APP_DB)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO donations (name, amount, date) VALUES (?, ?, ?)", (name, amount, now))
    conn.commit()
    conn.close()


def record_contact(name: str, email: str, message: str):
    conn = sqlite3.connect(APP_DB)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO contacts (name, email, message, date) VALUES (?, ?, ?, ?)", (name, email, message, now))
    conn.commit()
    conn.close()


def load_documents_table(source_dir: str = None):
    if source_dir is None:
        source_dir = DEFAULT_DATA_DIR
    else:
        source_dir = Path(source_dir)
    # load DocumentStore metadata
    meta_path = Path(DEFAULT_CHROMA_DIR) / "document_metadata.json"
    docs = []
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            file_index = json.load(f)
        for rel, entry in file_index.items():
            title = entry.get("title", "")
            inserted_at = entry.get("inserted_at", "")
            author = entry.get("author", "")
            publisher = entry.get("publisher", "")
            year = entry.get("year", "")
            category = entry.get("category", "")
            tier = entry.get("tier", "")
            # fallback to front-matter for text files if metadata is missing
            source_path = Path(source_dir) / rel
            if source_path.exists() and source_path.suffix.lower() != ".pdf":
                text = source_path.read_text(encoding="utf-8", errors="ignore")
                if text.startswith("---"):
                    parts = text.split("---")
                    if len(parts) >= 3:
                        fm = parts[1]
                        for line in fm.splitlines():
                            if ":" in line:
                                k, v = line.split(":", 1)
                                key = k.strip().lower()
                                val = v.strip()
                                if key in {"author", "authors"} and not author:
                                    author = val
                                if key == "publisher" and not publisher:
                                    publisher = val
                                if key in {"year", "date"} and not year:
                                    year = val
                                if key in {"category", "categories"} and not category:
                                    category = val
                                if key == "tier" and not tier:
                                    tier = val
            docs.append({
                "path": rel,
                "title": title,
                "inserted_at": inserted_at,
                "author": author,
                "publisher": publisher,
                "year": year,
                "category": category,
                "tier": tier,
            })
    return docs


init_db()

# create DocumentStore and Chatbot instances
store = DocumentStore()
llm = create_llm_client(None, None)
bot = Chatbot(store, llm)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/query", response_class=HTMLResponse)
def query(request: Request, question: str = Form(...)):
    answer = bot.ask(question)
    record_query(question)
    return templates.TemplateResponse("index.html", {"request": request, "question": question, "answer": answer})


@app.post("/donate")
def donate(name: str = Form(...), amount: float = Form(...)):
    record_donation(name, amount)
    return RedirectResponse(url="/stats", status_code=303)


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.post("/contact")
def contact(name: str = Form(...), email: str = Form(...), message: str = Form(...)):
    record_contact(name, email, message)
    return RedirectResponse(url="/", status_code=303)


@app.get("/disclaimer", response_class=HTMLResponse)
def disclaimer(request: Request):
    return templates.TemplateResponse("disclaimer.html", {"request": request})


@app.get("/documents", response_class=HTMLResponse)
def documents(request: Request):
    docs = load_documents_table()
    return templates.TemplateResponse("documents.html", {"request": request, "documents": docs})


@app.get("/technical", response_class=HTMLResponse)
def technical(request: Request):
    # basic technical info
    tech = {
        "llm_options": "Ollama or local transformers",
        "vector_store": "ChromaDB (duckdb+parquet)",
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    }
    return templates.TemplateResponse("technical.html", {"request": request, "tech": tech})


@app.get("/stats", response_class=HTMLResponse)
def stats(request: Request):
    conn = sqlite3.connect(APP_DB)
    cur = conn.cursor()
    cur.execute("SELECT question, count, last_asked FROM queries ORDER BY count DESC LIMIT 100")
    queries = cur.fetchall()
    cur.execute("SELECT name, amount, date FROM donations ORDER BY date DESC LIMIT 100")
    donations = cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM donations")
    total_donations = cur.fetchone()[0]
    cur.execute("SELECT SUM(amount) FROM donations")
    donation_sum = cur.fetchone()[0] or 0
    conn.close()
    return templates.TemplateResponse("stats.html", {"request": request, "queries": queries, "donations": donations, "donation_count": total_donations, "donation_sum": donation_sum})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp:app", host="127.0.0.1", port=8000, reload=True)

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BACKEND_DIR / "data"
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chromadb"))
DEFAULT_COLLECTION_NAME = "knowledge"
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md"}

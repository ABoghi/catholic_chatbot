import hashlib
import json
from datetime import datetime
from pathlib import Path

import chromadb
import pdfplumber
import yaml
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS,
)


class DocumentStore:
    def __init__(self, persist_dir=DEFAULT_CHROMA_DIR, collection_name=DEFAULT_COLLECTION_NAME):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(self.persist_dir))
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        self.metadata_path = self.persist_dir / "document_metadata.json"
        self.file_index = self._load_metadata()

    def _load_metadata(self):
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.file_index, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _normalize_path(path: Path, source_dir: Path):
        return str(path.relative_to(source_dir).as_posix())

    @staticmethod
    def _hash_text(text: str):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _chunk_text(text: str, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
        text = text.replace("\r\n", "\n").strip()
        if not text:
            return []
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == length:
                break
            start += chunk_size - chunk_overlap
        return chunks

    @staticmethod
    def _extract_title(text: str, path: Path) -> str:
        text = text.strip()
        if text.startswith("---"):
            parts = text.split("---")
            if len(parts) >= 3:
                fm = parts[1]
                for line in fm.splitlines():
                    if line.lower().startswith("title:"):
                        return line.split(":", 1)[1].strip().strip('"')
        for line in text.splitlines():
            s = line.strip()
            if s:
                return s if len(s) <= 200 else s[:197] + "..."
        return path.stem

    @staticmethod
    def _normalize_value(value):
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if item)
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _parse_front_matter(text: str) -> dict:
        metadata = {}
        text = text.strip()
        if not text.startswith("---"):
            return metadata
        parts = text.split("---")
        if len(parts) < 3:
            return metadata
        fm = parts[1]
        for line in fm.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            name = key.strip().lower()
            val = value.strip().strip('"').strip("'")
            if val.startswith("[") and val.endswith("]"):
                try:
                    val = json.loads(val.replace("'", '"'))
                except Exception:
                    pass
            if name in {"author", "authors"}:
                metadata["author"] = DocumentStore._normalize_value(val)
            elif name == "publisher":
                metadata["publisher"] = DocumentStore._normalize_value(val)
            elif name in {"year", "date"}:
                metadata["year"] = DocumentStore._normalize_value(val)
            elif name in {"category", "categories"}:
                metadata["category"] = DocumentStore._normalize_value(val)
            elif name == "tier":
                try:
                    metadata["tier"] = int(val)
                except ValueError:
                    metadata["tier"] = DocumentStore._normalize_value(val)
            elif name == "title":
                metadata["title"] = DocumentStore._normalize_value(val)
        return metadata

    @staticmethod
    def _load_text_file(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _load_pdf_file(path: Path) -> str:
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
        return "\n\n".join(pages)

    def _load_file(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            return self._load_pdf_file(path)
        return self._load_text_file(path)

    def _embed(self, texts):
        return self.embedding_model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def _load_sidecar_metadata(self, source_dir: Path) -> dict:
        metadata = {}
        for filename in ("metadata.json", "metadata.yml", "metadata.yaml"):
            path = source_dir / filename
            if not path.exists():
                continue
            try:
                if path.suffix.lower() == ".json":
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
            except Exception as exc:
                print(f"Warning: failed to parse sidecar metadata {path}: {exc}")
                continue
            if isinstance(data, dict):
                if "documents" in data and isinstance(data["documents"], dict):
                    data = data["documents"]
                for key, value in data.items():
                    if not isinstance(value, dict):
                        continue
                    rel_key = str(Path(key).as_posix())
                    metadata[rel_key] = value
            break
        return metadata

    def ingest_folder(self, source_dir: str):
        source_dir = Path(source_dir)
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        sidecar_metadata = self._load_sidecar_metadata(source_dir)
        discovered_files = {}
        for path in source_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                discovered_files[self._normalize_path(path, source_dir)] = path

        removed = [key for key in self.file_index if key not in discovered_files]
        for rel_path in removed:
            entry = self.file_index.pop(rel_path, None)
            if entry and entry.get("ids"):
                self.collection.delete(ids=entry["ids"])
        if removed:
            print(f"Removed {len(removed)} deleted files from the index.")

        ingested = 0
        updated = 0
        skipped = 0

        for rel_path, path in sorted(discovered_files.items()):
            text = self._load_file(path)
            doc_hash = self._hash_text(text)
            index_entry = self.file_index.get(rel_path)
            if index_entry and index_entry.get("hash") == doc_hash:
                skipped += 1
                continue

            if index_entry and index_entry.get("ids"):
                self.collection.delete(ids=index_entry["ids"])

            chunks = self._chunk_text(text)
            if not chunks:
                print(f"Skipping empty file: {rel_path}")
                continue

            file_metadata = sidecar_metadata.get(rel_path, {})
            extracted_metadata = self._parse_front_matter(text) if path.suffix.lower() != ".pdf" else {}
            title = self._normalize_value(
                file_metadata.get("title")
                or extracted_metadata.get("title")
                or self._extract_title(text, path)
            )
            author = self._normalize_value(
                file_metadata.get("author")
                or file_metadata.get("authors")
                or extracted_metadata.get("author")
            )
            publisher = self._normalize_value(
                file_metadata.get("publisher") or extracted_metadata.get("publisher")
            )
            year = self._normalize_value(
                file_metadata.get("year")
                or file_metadata.get("date")
                or extracted_metadata.get("year")
            )
            category = self._normalize_value(
                file_metadata.get("category")
                or file_metadata.get("categories")
                or extracted_metadata.get("category")
            )
            tier_value = file_metadata.get("tier", extracted_metadata.get("tier"))
            try:
                tier = int(tier_value) if tier_value is not None and str(tier_value).strip() != "" else None
            except ValueError:
                tier = DocumentStore._normalize_value(tier_value)

            file_id = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()
            ids = [f"{file_id}-{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": rel_path,
                    "file_name": path.name,
                    "chunk_index": i,
                    "chunk_length": len(chunk),
                    "author": author,
                    "publisher": publisher,
                    "year": year,
                    "category": category,
                    "tier": tier,
                }
                for i, chunk in enumerate(chunks)
            ]
            embeddings = self._embed(chunks)
            self.collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            inserted_at = index_entry.get("inserted_at") if index_entry and index_entry.get("inserted_at") else datetime.utcnow().isoformat()
            self.file_index[rel_path] = {
                "hash": doc_hash,
                "ids": ids,
                "title": title,
                "inserted_at": inserted_at,
                "author": author,
                "publisher": publisher,
                "year": year,
                "category": category,
                "tier": tier,
            }
            ingested += 1
            if index_entry:
                updated += 1

        self.client.persist()
        self._save_metadata()

        print(f"Ingestion complete: {ingested} new/updated files, {skipped} unchanged files.")

    def list_documents(self):
        return sorted(self.file_index.keys())

    def query(self, question: str, top_k: int = 4):
        query_embedding = self._embed([question])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = []
        for doc, meta, distance in zip(
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            documents.append({"text": doc, "meta": meta, "distance": distance})
        documents.sort(key=lambda x: (int(x["meta"].get("tier", 999)) if str(x["meta"].get("tier", "")).isdigit() else 999, x["distance"]))
        return documents

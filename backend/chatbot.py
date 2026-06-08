import os
from pathlib import Path

from .config import BACKEND_DIR


class Chatbot:
    def __init__(self, store, llm):
        self.store = store
        self.llm = llm
        self.system_prompt = self._load_system_prompt()

    @staticmethod
    def _load_system_prompt() -> str:
        prompt_file = os.getenv("SYSTEM_PROMPT_FILE", str(BACKEND_DIR / "system_prompt.txt"))
        prompt_path = Path(prompt_file)
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8", errors="ignore").strip()
        return (
            "You are a helpful assistant answering user questions from documented knowledge. "
            "Use only the context provided from the documents and do not invent answers. "
            "If the answer is not contained in the context, say that you do not know."
        )

    def build_prompt(self, question: str, contexts: list[dict]) -> str:
        chunks = []
        for item in contexts:
            source = item["meta"].get("source", "unknown")
            # Strip directory path, keep only filename without extension
            source_name = Path(source).stem
            # source_name = ( item["meta"].get("title") or Path(item["meta"].get("source", "unknown")).stem )
            chunks.append(f"Source: {source_name}\n{item['text']}")
        context_section = "\n\n".join(chunks)
        return (
            f"{self.system_prompt}\n\n"
            "Context (use ALL of the following passages to build ONE "
            "unified answer, do not summarise each source separately):\n"
            f"{context_section}\n\n"
            "Question: "
            f"{question}\n\n"
            "Answer:"
        )

    def ask(self, question: str, top_k: int = 6) -> str:
        contexts = self.store.query(question, top_k=top_k)
        disclaimer = (
            "I am only a chatbot and I can make mistake. Please, always seek confirmation of what I tell you to your local parish priest."
        )
        if not contexts:
            return f"No documents are available in the vector store. Run ingest first.\n\n{disclaimer}"
        prompt = self.build_prompt(question, contexts)
        answer = self.llm.generate(prompt)
        return f"{answer}\n\n{disclaimer}"

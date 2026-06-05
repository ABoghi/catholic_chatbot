import argparse

try:
    from .config import DEFAULT_DATA_DIR
    from .document_store import DocumentStore
    from .llm_clients import create_llm_client
    from .chatbot import Chatbot
except ImportError:
    from config import DEFAULT_DATA_DIR
    from document_store import DocumentStore
    from llm_clients import create_llm_client
    from chatbot import Chatbot


def main() -> None:
    parser = argparse.ArgumentParser(description="Chroma + Ollama knowledge chatbot")
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--llm-provider", help="LLM provider to use (ollama, local, transformers)")
    parent.add_argument("--llm-model", help="LLM model name to use (e.g. llama2, phi-4-mini, gpt2)")

    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest", parents=[parent], help="Ingest PDF/Text/Markdown documents")
    ingest_parser.add_argument(
        "--source",
        default=str(DEFAULT_DATA_DIR),
        help="Source folder for documents",
    )

    query_parser = subparsers.add_parser("query", parents=[parent], help="Ask a question")
    query_parser.add_argument("question", nargs="+", help="The question to ask")
    query_parser.add_argument("--top-k", type=int, default=4, help="Number of context chunks to retrieve")

    subparsers.add_parser("interactive", parents=[parent], help="Start an interactive chat session")
    subparsers.add_parser("list-docs", parents=[parent], help="List all ingested documents")

    args = parser.parse_args()
    store = DocumentStore()
    llm = create_llm_client(getattr(args, "llm_provider", None), getattr(args, "llm_model", None))
    bot = Chatbot(store, llm)

    if args.command == "ingest":
        store.ingest_folder(args.source)
        return

    if args.command == "query":
        question = " ".join(args.question).strip()
        response = bot.ask(question, top_k=args.top_k)
        print("\n---\n")
        print(response)
        return

    if args.command == "list-docs":
        documents = store.list_documents()
        if not documents:
            print("No documents ingested yet. Run `python -m backend.main ingest --source backend/data`.")
            return
        print("Ingested documents:")
        for doc in documents:
            print(f"- {doc}")
        return

    if args.command == "interactive":
        print("Interactive chatbot ready. Type 'exit' or 'quit' to stop.")
        while True:
            question = input("\nQuestion: ").strip()
            if not question or question.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break
            answer = bot.ask(question)
            print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()

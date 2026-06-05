import json
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st

from backend.chatbot import Chatbot
from backend.config import DEFAULT_CHROMA_DIR, DEFAULT_DATA_DIR
from backend.document_store import DocumentStore
from backend.llm_clients import create_llm_client

PROJECT_ROOT = Path(__file__).resolve().parent
APP_DB = PROJECT_ROOT / "web_data.db"


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


def get_stats():
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
    return queries, donations, total_donations, donation_sum


def load_documents_table(source_dir: str = None):
    if source_dir is None:
        source_dir = DEFAULT_DATA_DIR
    else:
        source_dir = Path(source_dir)
    docs = []
    metadata_path = Path(DEFAULT_CHROMA_DIR) / "document_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            file_index = json.load(f)
        for rel, entry in file_index.items():
            docs.append(
                {
                    "path": rel,
                    "title": entry.get("title", ""),
                    "inserted_at": entry.get("inserted_at", ""),
                    "author": entry.get("author", ""),
                    "publisher": entry.get("publisher", ""),
                    "year": entry.get("year", ""),
                    "category": entry.get("category", ""),
                    "tier": entry.get("tier", ""),
                }
            )
    return docs


@st.cache_resource
def create_app(llm_provider: str | None, llm_model: str | None):
    store = DocumentStore()
    llm = create_llm_client(llm_provider, llm_model)
    return Chatbot(store, llm)


def main():
    st.set_page_config(page_title="Catholic Chatbot", layout="wide")

    st.sidebar.title("Catholic Chatbot")
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Documents", "Technical", "Stats", "About", "Disclaimer"],
    )
    st.sidebar.markdown("---")
    provider = st.sidebar.selectbox(
        "LLM provider",
        ["ollama", "local"],
        index=0,
        help="Choose whether to use Ollama or a local transformers model.",
    )
    model = st.sidebar.text_input("LLM model", value="llama2")
    st.sidebar.write("Default embedding model: all-MiniLM-L6-v2")
    st.sidebar.write("Data source: backend/data")
    st.sidebar.write("Vector store: ./chromadb")

    chatbot = create_app(provider, model)

    if page == "Home":
        st.title("Catholic Chatbot")
        st.write(
            "Ask questions about the ingested documents and receive answers backed by the vector store. "
            "Please confirm anything important with a local parish priest."
        )
        question = st.text_area("Ask a question", height=120)
        if st.button("Submit") and question.strip():
            with st.spinner("Generating answer..."):
                answer = chatbot.ask(question)
                record_query(question)
                st.subheader("Answer")
                st.write(answer)
        st.markdown("---")
        st.subheader("Contact / Donate")
        cols = st.columns(2)
        with cols[0]:
            st.write("### Send a message")
            name = st.text_input("Name", key="contact_name")
            email = st.text_input("Email", key="contact_email")
            message = st.text_area("Message", key="contact_message")
            if st.button("Send message"):
                if name.strip() and email.strip() and message.strip():
                    record_contact(name, email, message)
                    st.success("Thank you — your message has been recorded.")
                else:
                    st.error("Please fill in all contact fields.")
        with cols[1]:
            st.write("### Support the project")
            donor_name = st.text_input("Name", key="donor_name")
            amount = st.number_input("Donation amount", min_value=0.0, step=1.0, key="donation_amount")
            if st.button("Donate"):
                if donor_name.strip() and amount > 0:
                    record_donation(donor_name, amount)
                    st.success("Thanks for your support!")
                else:
                    st.error("Please enter your name and a donation amount greater than zero.")

    elif page == "Documents":
        st.title("Documents")
        st.write("List of documents currently indexed in the vector store.")
        docs = load_documents_table()
        if docs:
            st.dataframe(docs)
        else:
            st.warning("No document metadata found. Run ingestion first.")

    elif page == "Technical":
        st.title("Technical")
        st.write("Technical overview of the chatbot stack.")
        st.markdown(
            "- Vector store: ChromaDB with duckdb+parquet\n"
            "- Embedding model: all-MiniLM-L6-v2\n"
            "- LLM provider: %s\n"
            "- LLM model: %s\n"
            "- Data folder: backend/data\n"
            "- Metadata file: ./chromadb/document_metadata.json"
            % (provider, model)
        )
        st.markdown("### Model details")
        st.write(
            "Using the backend chatbot and document store code, this Streamlit frontend connects to the same data and metadata as the existing FastAPI app."
        )

    elif page == "Stats":
        st.title("Stats")
        queries, donations, total_donations, donation_sum = get_stats()
        st.metric("Total donations", f"{total_donations}")
        st.metric("Donation sum", f"${donation_sum:.2f}")
        st.markdown("### Top questions")
        if queries:
            st.dataframe(
                [
                    {"question": q, "count": c, "last_asked": d}
                    for q, c, d in queries
                ]
            )
        else:
            st.warning("No query activity yet.")
        st.markdown("### Recent donations")
        if donations:
            st.dataframe(
                [
                    {"name": name, "amount": amount, "date": date}
                    for name, amount, date in donations
                ]
            )
        else:
            st.info("No donations recorded yet.")

    elif page == "About":
        st.title("About")
        st.write(
            "This chatbot uses a vector database to answer questions from PDF and text documents. "
            "It supports Ollama or local HuggingFace models and stores document metadata for browsing."
        )
        st.markdown(
            "### Features\n"
            "- Ask questions and get context-backed answers\n"
            "- Browse ingested documents and metadata\n"
            "- Track query and donation statistics\n"
            "- Choose LLM provider and model from the sidebar\n"
        )

    elif page == "Disclaimer":
        st.title("Disclaimer")
        st.write(
            "This chatbot is for informational purposes only. It may be inaccurate or incomplete. "
            "Always confirm important religious guidance with a qualified local priest or spiritual advisor."
        )


if __name__ == "__main__":
    import json, os
    os.system("ollama serve")
    os.system("ollama pull llama2")
    init_db()
    main()

from backend.document_store import DocumentStore

store = DocumentStore()

# Replace with the exact rel_path as it appears in the index
rel_path_1 = "The Catechism of Saint Pope Pius X.pdf"

def delete_document(rel_path=rel_path_1):
    entry = store.file_index.get(rel_path)
    if not entry:
        print(f"Not found in index: {rel_path}")
    else:
        store.collection.delete(ids=entry["ids"])
        del store.file_index[rel_path]
        store._save_metadata()
        print(f"Removed {rel_path} ({len(entry['ids'])} chunk deleted)")
    print(f"Collection count: {store.collection.count()}")

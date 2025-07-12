from db.chroma_client import get_chroma
from ingestion.load_all_data import load_all_data

def get_db():
    chroma_db = get_chroma()

    collection_info=chroma_db._collection.get()
    if not collection_info["ids"]:
        print("Chroma db is empty. Loading all data...")
        load_all_data(chroma_db)
    else:
        print("Chroma db already contains data")

    return chroma_db
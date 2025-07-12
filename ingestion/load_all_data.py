from ingestion.ingest_local import ingest_local
from ingestion.ingest_web import ingest_web

def load_all_data(chroma_db):
    print("Loading all data into Chroma...")
    ingest_local(chroma_db)
    ingest_web(chroma_db)
    print("All data loaded successfully")


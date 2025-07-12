import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_DIR="data/chromadb"
DEFAULT_EMBED_MODEL=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def get_chroma(embedding_func=None, persist_dir=CHROMA_DIR):
    if embedding_func is None:
        embedding_func=HuggingFaceEmbeddings(
             model_name=DEFAULT_EMBED_MODEL
        )
    return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_func,
        )
    
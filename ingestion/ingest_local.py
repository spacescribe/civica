import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings

from db.chroma_client import get_chroma

load_dotenv()

DATA_DIR="data/raw"

def load_documents():
    documents=[]
    for filename in os.listdir(DATA_DIR):
        file_path=os.path.join(DATA_DIR, filename)
        if filename.endswith('.pdf'):
            loader=PyPDFLoader(file_path)
            doc=loader.load()
            documents.extend(doc)
        elif filename.endswith('.txt'):
            loader-TextLoader(file_path)
            doc=loader.load()
            documents.extend(doc)
    
    return documents

def main():
    print("Loading documents...")
    docs=load_documents()
    print(f"Loaded {len(docs)} documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    print("Splitting into chunks...")
    split_docs=splitter.split_documents(docs)
    print(f"Total number of chunks {len(split_docs)}")

    print("Creating Chroma vectorstore...")
    chromadb=get_chroma(create_if_missing=True)

    chromadb.add_documents(split_docs)
    # chromadb.persist()
    print("Ingestion complete!")

if __name__=="__main__":
    main()


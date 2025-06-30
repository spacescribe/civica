import trafilatura
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from db.chroma_client import get_chroma

def fetch_article(url):
    downloaded=trafilatura.fetch_url(url)
    if downloaded:
        return trafilatura.extract(downloaded)
    return None

def main():
    urls=[
        "https://www.scobserver.in/journal/supreme-court-review-top-10-judgements-of-2024/",
        "https://en.wikipedia.org/wiki/List_of_landmark_court_decisions_in_India",
        "https://www.scconline.com/blog/post/2025/01/07/top-criminal-law-cases-2024-important-decisions-high-courts-across-india/",
        "https://www.lexisnexis.in/blogs/laws-for-women-in-india/#:~:text=The%20Constitution%20along%20with%20legislations,guarantees%20these%20rights%20to%20women.",
        "https://lawchakra.in/blog/25-landmark-judgments-womens-rights/"
        "https://en.wikipedia.org/wiki/Human_rights_in_India"
        "https://www.drishtiias.com/to-the-points/Paper2/human-rights-22"
    ]

    print("Fetching articles...")
    documents=[]
    for url in urls:
        text=fetch_article(url)
        if text:
            doc = Document(
                page_content=text,
                metadata={"source": url}
            )
            documents.append(doc)
            print(f"Fetched url: {url}")
        else:
            print(f"Failed to fetch: {url}")

    if not documents:
        print("No documents fetched")
        return
    
    print("Splitting into chunks...")
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks=splitter.split_documents(documents)
    print(f"Total number of chunks {len(chunks)}")

    print("Stroing in Chroma vectorstore...")
    chromadb=get_chroma(create_if_missing=True)

    chromadb.add_documents(chunks)
    # chromadb.persist()
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
from langchain.chains import RetrievalQA
from models.groq_client import get_groq_llm
from db.chroma_client import get_chroma

def main():
    db=get_chroma()
    retriever=db.as_retriever()

    llm=get_groq_llm()

    qa=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    query=("What are the fundamental rights under the Indian consistution?")
    res=qa.invoke(query)

    print(res["result"])
    print("Sources: ")
    for doc in res["source_documents"]:
        print(doc.metadata)

    
if __name__ == "__main__":
    main()
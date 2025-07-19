from db.db_client import get_db
from models.groq_client import get_groq_llm
from rerankers.cross_encoder import rerank
from rich.console import Console

console=Console()


retriever=get_db().as_retriever(search_kwargs={"k":20})

def query(question):
    retrieved_docs=retriever.invoke(question)

    if not retrieved_docs:
        console.print("[bold red] No relevant documents found.[/bold red]")
        return
    
    reranked_docs=rerank(question, retrieved_docs)
    
    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}"
        for d in reranked_docs
    )

    console.print("\n[magenta]Sending your question to civica...[/magenta]\n")

    prompt = f"""
    You are Civica, a knowledgeable assistant for Indian law and human rights.
    Use the context below to answer the user's question concisely and accurately.
    If you don't know, say you don't know.

    Context:
    {context}

    Question: {question}

    Answer: 

    In your answer, clearly cite which [Source] you are referring to, where appropriate.

    """

    llm=get_groq_llm()
    response=llm.invoke(prompt)
    return response.content.strip(), context


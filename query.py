from db.chroma_client import get_chroma
from models.groq_client import get_groq_llm
from rich.console import Console

console=Console()

def main():
    console.print("\n[bold blue]Civica - Your own AI legal assistant[/bold blue]\n")
    question=console.input("[aqua]Your question: [/aqua]\n")

    retriever=get_chroma().as_retriever(search_kwargs={"k":5})
    docs=retriever.get_relevant_documents(question)

    if not docs:
        console.print("[bold red] No relevant documents found.[/bold red]")
        return
    
    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}"
        for d in docs
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

    """

    llm=get_groq_llm()
    response=llm.invoke(prompt)
    console.print("\n[magenta]Invoking the model...[/magenta]\n")

    console.print("[bold green]Answer: [/bold green]")
    console.print(response.content)

    console.print("\n[bold green]Sources: [/bold green]")
    for i, d in enumerate(docs, start=1):
        console.print(f"{i}. {d.metadata.get('source', "Unknown")}")

if __name__ == "__main__":
    main()

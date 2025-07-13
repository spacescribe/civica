from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import (create_history_aware_retriever, create_history_aware_retriever)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from db.db_client import get_db
from models.groq_client import get_groq_llm
from rerankers.cross_encoder import rerank
from rich.console import Console

console = Console()

base_retriever = get_db().as_retriever(search_kwargs={"k": 20})
llm = get_groq_llm()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, base_retriever, contextualize_q_prompt
)

qa_system_prompt = (
    "You are Civica, a knowledgeable assistant for Indian law and human rights.\n"
    "Use the context below to answer the user's question concisely and accurately.\n"
    "If you don't know, say you don't know.\n\n"
    "Context: {context}\n"
    "Question: {input}\n\n"
    "Anwser: \n\n"
    "Instructions:\n"
    "- List all applicable rights if relevant.\n"
    "- For each right, briefly explain it.\n"
    "- Always include the source in square brackets.\n"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt,
)

chat_history = []

def query(question):
    console.print(f"[cyan]Rephrasing user question...[/cyan]")
    retrieved_docs = history_aware_retriever.invoke({
            "chat_history": chat_history,
            "input": question
        })

    if not retrieved_docs: 
        console.print("[bold red]No relevant documents found.[/bold red]")
        return 
    
    console.print(f"[cyan]Reranking documents...[/cyan]")
    reranked_docs= rerank(question, retrieved_docs)

    console.print(f"[cyan]Generating the answer...[/cyan]")
    response = answer_chain.invoke({
        "chat_history": chat_history,
        "input": question,
        "context": reranked_docs,
    })

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.strip()))

    return response.strip()

if __name__ == "__main__":
    while True:
        user_q = input("\nYour question: ")
        if user_q.lower() in {"exit", "quit"}:
            break
        answer = query(user_q)
        console.print(f"\n[bold magenta]Answer:[/bold magenta]\n{answer}\n")

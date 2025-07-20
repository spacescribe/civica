from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from models.groq_client import get_groq_llm
from rich.console import Console

console = Console()

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
        ("human", "{input}"),
    ]
)

llm=get_groq_llm()

answer_chain = create_stuff_documents_chain(
    llm= llm,
    prompt=qa_prompt,
)

def answer_node(state: dict) -> dict:
    reranked_docs = state["reranked_docs"]
    question = state["question"]

    console.print(f"[cyan]Generating the answer...[/cyan]")
    response = answer_chain.invoke({
        "context": reranked_docs,
        "input": question
    })

    state["answer"] = response
    return state
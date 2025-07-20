from db.db_client import get_db
from rich.console import Console

console= Console()

retreiver = get_db().as_retriever(search_kwargs={"k": 20})

def retrieve_node(state: dict) -> dict:
    query = state.get("rephrased_question") or state["question"]

    console.print(f"[cyan]Retrieving docs related to  user question...[/cyan]")
    
    retrieved_docs = retreiver.invoke(query)
    state["retrieved_docs"]=retrieved_docs

    console.print(f"[cyan]Retrieving docs related to  user question complete...[/cyan]")

    return state
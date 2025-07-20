from rerankers.cross_encoder import rerank
from rich.console import Console

console = Console()

def rerank_node(state: dict) -> dict:
    query = state.get("rephrased_question") or state["question"]
    docs = state["retrieved_docs"]

    console.print(f"[cyan]Reranking retrieved docs related to  user question...[/cyan]")
    reranked_docs = rerank(query, docs)

    state["reranked_docs"]=reranked_docs

    console.print(f"[cyan]Reranking retrieved docs related to  user question complete...[/cyan]")
    return state
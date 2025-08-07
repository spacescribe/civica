from langchain_core.tools import tool
from langgraph_pipeline.nodes.rerank import rerank_node

@tool
def rerank_tool(question: str, docs: list) -> list:
    """
    Reranks the documents based on the relevance to the query
    Only use this tool after retrieve_tool.
    Expects: {'question': str, 'docs': list}
    """
    state = {
        "question": question, 
        "retrieved_docs": docs
    }

    result = rerank_node(state)
    return result["reranked_docs"]
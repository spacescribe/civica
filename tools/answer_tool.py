from langchain_core.tools import tool
from langgraph_pipeline.nodes.answer import answer_node

@tool
def answer_tool(question: str, docs: list) -> str:
    """
    Answer legal questions using provided documents about Indian law and human rights.
    This should be called only after retreive_tool or rerank_tool
    Expects: {'question': str, 'docs': list}
    """
    state = {
        "question": question, 
        "retrieved_docs": docs
    }
    result = answer_node(state)
    return result["answer"]
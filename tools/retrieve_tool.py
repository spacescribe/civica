from langchain_core.tools import tool
from langgraph_pipeline.nodes.retrieve import retrieve_node

@tool
def retrieve_tool(question: str) -> list:
    """
    Retrieves relevant documents for a question
    
    Expects: {'question': str}
    """
    state = {"question": question}
    result = retrieve_node(state)
    return result["retrieved_docs"]

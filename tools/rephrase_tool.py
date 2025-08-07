from langchain_core.tools import tool
from langgraph_pipeline.nodes.rephrase import rephrase_node

@tool
def rephrase_tool(question: str, chat_history: list)-> str:
    """
    Tool to rephrase user question using chat history
    
    Expects: {'question': str, 'chat_history': list}
    """
    state = {
        "question": question,
        "chat_history": chat_history
    }
    result = rephrase_node(state)
    return result["rephrased_question"]
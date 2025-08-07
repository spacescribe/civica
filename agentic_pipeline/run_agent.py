from tools.registry import tool_list
from models.groq_client import get_groq_llm
from langchain.agents import initialize_agent

llm = get_groq_llm()

agent = initialize_agent(
    tools=tool_list,
    llm = llm,
    agent="zero-shot-react-description",
    verbose=True,
)

def run_agentic_query(question: str, chat_history: list = None) -> str:
    """
    Runs the agent with the given user question.
    Includes chat_history if tools use it (like rephrase).
    Adds usage guidance so the agent knows to retrieve before rerank.
    """
    chat_context = ""
    if chat_history:
        chat_context = "\n".join(
            [f"{item['role']}: {item['content']}" for item in chat_history]
        )

    tool_usage_hint = """
        TOOL USAGE INSTRUCTIONS:
        - Always use 'retrieve_tool' to fetch documents before reranking.
        - Only use 'rerank_tool' if you already have documents.
        - Format input to 'rerank_tool' as: {"question": "...", "docs": [...]}
    """

    if chat_context:
        full_prompt = f"{tool_usage_hint}\n\nGiven this chat history:\n{chat_context}\n\nCurrent question: {question}"
    else:
        full_prompt = f"{tool_usage_hint}\n\nCurrent question: {question}"

    response = agent.run(full_prompt)

    return response

      
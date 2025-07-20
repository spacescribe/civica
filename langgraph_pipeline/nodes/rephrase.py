from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from models.groq_client import get_groq_llm
from rich.console import Console

console=Console()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

llm=get_groq_llm()

def rephrase_node(state: dict)-> dict:
    """
    Rephrase user question using the chat history to a standalone question.
    """

    question = state["question"]
    chat_history = state.get("chat_history", [])

    if not chat_history:
        state["rephrased_question"]=question
        return state
    
    console.print(f"[cyan]Rephrasing user question...[/cyan]")
    prompt_input = contextualize_q_prompt.format_messages(
        chat_history=chat_history,
        input = question
    )

    response = llm.invoke(prompt_input)
    rephrased = response.content.strip()

    state["rephrased_question"]=rephrased
    console.print(f"[cyan]Rephrasing user question complete...[/cyan]")
    return state
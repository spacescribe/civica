# from query import query
import json
from evaluation.llm_critique import llm_critique
from rich.console import Console
from langgraph_pipeline.build_graph import graph
from agentic_pipeline.run_agent import run_agentic_query

console=Console()

def run_chatbot_query_pipeline(question: str, chat_history: list =None):
    if chat_history is None:
        chat_history=[]

    initial_state = {
        "question": question,
        "chat_history": chat_history
    }

    final_state = graph.invoke(initial_state)
    return final_state["answer"]

def main():
    console.print("\n[bold blue]Civica - Your own AI legal assistant[/bold blue]\n")

    mode = console.input(
            "[bold magenta]Choose mode:[/bold magenta] [1] RAG (default) | [2] Agentic\n> "
        ).strip()

    chat_history = []

    while True:
        question = console.input("\n[aqua]Your question (or type 'exit' to quit): [/aqua] ")

        if question.strip().lower() in {"exit", "quit"}:
            console.print("\n[bold yellow]Goodbye! 👋[/bold yellow]")
            break
            
        if mode == "2":
            response = run_agentic_query(question, chat_history=chat_history)

        else:
            response = run_chatbot_query_pipeline(question, chat_history)

        console.print("[bold green]Answer: [/bold green]")
        console.print(response)

        # Add to chat history
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": response})

        # Run LLM critique
        critique = llm_critique(question, response)
        critique = json.loads(critique)

        console.print("\n[aqua]Critique response: [/aqua]")
        for k, v in critique.items():
            console.print(f"{k}: {v}")


if __name__ == "__main__":
    main()

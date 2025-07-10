from query import query
import json
from evaluation.llm_critique import llm_critique
from rich.console import Console

console=Console()

def main():
    console.print("\n[bold blue]Civica - Your own AI legal assistant[/bold blue]\n")
    question=console.input("[aqua]Your question: [/aqua]\n")

    response, context=query(question)

    console.print("[bold green]Answer: [/bold green]")
    console.print(response)

    critique=llm_critique(
        question,
        response,
    )

    console.print("\n[aqua]Critique response: [/aqua]")
    critique = json.loads(critique)

    print("\nEvaluation Metrics:")
    for k, v in critique.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

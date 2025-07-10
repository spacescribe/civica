from models.groq_client import get_groq_llm

def llm_critique(question, generated_answer, ref_text=None):
    llm=get_groq_llm(0)

    prompt = f"""
        You are a legal expert evaluating the answer to a legal question.
        Please analyze the answer carefully.

        Question:
        {question}

        Generated Answer:
        {generated_answer}

        Reference Answer (if available):
        {ref_text if ref_text else "N/A"}

        Instructions:
        Rate the Generated Answer on the following aspects from 1 to 5.
        Then provide a short explanation of your ratings.

        Respond strictly in this JSON format:

        {{
        "accuracy": <1-5>,
        "completeness": <1-5>,
        "faithfulness": <1-5>,
        "clarity": <1-5>,
        "comments": "<your explanation>"
        }}

    """

    response = llm.invoke(prompt)
    return response.content.strip()
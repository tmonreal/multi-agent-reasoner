import re

def split_question(llm, question: str) -> list[str]:
    """
    Split a complex question into 2–4 simpler sub-questions.

    Args:
        llm: The LLM instance to invoke for generating sub-questions.
        question (str): The original complex question.

    Returns:
        list[str]: A list of cleaned sub-questions.
    """

    prompt = f"""
Your job is to extract 2–4 sub-questions from this complex question. 
Only output direct sub-questions, each on its own line starting with a dash (-).

If the question is already simple, just return it alone.

Question: "{question}"
    """
    response = llm.invoke(prompt)
    content = getattr(response, "content", str(response))

    # Parse and clean each line
    lines = content.strip().split("\n")
    questions = []
    for line in lines:
        clean = line.strip("-•* \t").strip()
        if not clean:
            continue # Skip lines that are meta, unclear, or not actual questions
        if not re.search(r"[a-z]", clean.lower()):
            continue  # Skip all caps / header lines
        if not clean.endswith("?"):
            continue  # Only keep lines that are questions
        questions.append(clean)

    # If no valid sub-questions remain after cleaning, return the original question
    return questions if questions else [question]

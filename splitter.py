import re

def split_question(llm, question: str) -> list[str]:
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
        # Skip lines that are meta, unclear, or not actual questions
        if not clean:
            continue
        if not re.search(r"[a-z]", clean.lower()):
            continue  # skip all caps / header lines
        if not clean.endswith("?"):
            continue  # only keep lines that are questions
        questions.append(clean)

    # fallback in case filtering removed all
    return questions if questions else [question]

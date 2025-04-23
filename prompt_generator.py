def generate_role_prompt(llm, subquestion: str) -> str:
    prompt = f"""
    You are designing a one-line role prompt to guide an expert LLM answering the following sub-question:

    "{subquestion}"

    The prompt should define what kind of expert is answering, in under 25 words.

    Respond only with the prompt line, no explanation.
    """
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response)).strip()

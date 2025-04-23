def combine_answers(llm, subqa_pairs: list[tuple[str, str]]) -> str:
    combined = "\n".join(f"Q: {q}\nA: {a}" for q, a in subqa_pairs)
    prompt = f"""
    Given the following partial answers, summarize them into 3-5 sentences, avoiding repetition or excessive technicality.

    {combined}

    Final Answer:
    """
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response)).strip()
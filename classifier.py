from token_utils import count_tokens

def classify_question_complexity(llm, question: str, tracker) -> str:
    """
    Classifies a user's question as 'math-step', 'reasoning', or 'simple'.
    Logs token usage.
    """
    classification_prompt = f"""
Classify the following question as one of the following:
- 'math-step' → if it requires multi-step math or engineering-style calculation
- 'reasoning' → if it requires logical or analytical reasoning across multiple facts
- 'simple' → if it can be answered directly without breakdown

Question: {question}

Respond with only one of: math-step, reasoning, simple
    """
    response = llm.invoke(classification_prompt)
    output = getattr(response, "content", str(response)).strip().lower()
    tracker.log(classification_prompt, output, count_tokens)
    return output


def decide_decomposition_strategy(llm, question: str, tracker) -> str:
    """
    Decides whether the question should be answered via 'steps' or 'subquestions'.
    Logs token usage.
    """
    prompt = f"""
Decide the best strategy to decompose the following task:
- 'steps' → if it's best to answer step-by-step
- 'subquestions' → if it's better to divide into multiple related subquestions

Task: {question}

Respond with only one of: steps, subquestions
Answer:"""
    response = llm.invoke(prompt)
    output = getattr(response, "content", str(response)).strip().lower()
    tracker.log(prompt, output, count_tokens)
    return output


def classify_subquestion_type(llm, subq: str, tracker) -> str:
    """
    Classifies a sub-question as either 'math-step' or 'simple'.
    Logs token usage.
    """
    classification_prompt = f"""
Classify the following sub-question as one of:
- 'math-step' → if it involves numerical or multi-step calculation
- 'simple' → if it is factual or can be answered directly

Sub-question: {subq}

Respond with ONLY one of: math-step, simple
    """
    response = llm.invoke(classification_prompt)
    output = getattr(response, "content", str(response)).strip().lower()
    tracker.log(classification_prompt, output, count_tokens)
    return output

def combine_answers(llm, subqa_pairs: list[tuple[str, str]]) -> str:
    """
    Combine multiple sub-question answers into a coherent final answer.
    Args:
        llm: The language model instance used for summarization.
        subqa_pairs: A list of tuples containing sub-questions and their corresponding answers.
    
    Returns:
        A single string that summarizes the answers to the sub-questions.
    """
    combined = "\n".join(f"Q: {q}\nA: {a}" for q, a in subqa_pairs)
    prompt = f"""
Given the following partial answers, summarize them into 3-5 sentences, avoiding repetition or excessive technicality.

{combined}

Final Answer:
    """
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response)).strip()
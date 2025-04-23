from splitter import split_question
from prompt_generator import generate_role_prompt
from agent import GenericAgent, StepByStepMathAgent
from combiner import combine_answers
from llm_loader import load_llm

def classify_question_complexity(llm, question: str) -> str:
    classification_prompt = f"""
    Classify the following question as one of the following:
    - 'math-step' → if it requires multi-step math or engineering-style calculation
    - 'reasoning' → if it requires logical or analytical reasoning across multiple facts
    - 'simple' → if it can be answered directly without breakdown

    Question: {question}

    Respond with only one of: math-step, reasoning, simple
    """
    response = llm.invoke(classification_prompt)
    return getattr(response, "content", str(response)).strip().lower()

def classify_subquestion_type(llm, subq: str) -> str:
    classification_prompt = f"""
    Classify the following sub-question as one of:
    - 'math-step' → if it involves numerical or multi-step calculation
    - 'simple' → if it is factual or can be answered directly

    Sub-question: {subq}

    Respond with ONLY one of: math-step, simple
    """
    response = llm.invoke(classification_prompt)
    return getattr(response, "content", str(response)).strip().lower()

def run_reasoning_pipeline(question: str):
    llm = load_llm()
    print("\nOriginal question:", question)

    classification = classify_question_complexity(llm, question)
    print("\nGlobal classification:", classification)

    if classification == "simple":
        agent = GenericAgent(llm)
        answer = agent.answer(question, "You are a helpful expert. Answer clearly and briefly.")
        print("\n✅ Final Answer:\n")
        print(answer)
        return

    if classification == "math-step":
        agent = StepByStepMathAgent(llm)
        answer = agent.answer(question)
        print("\n✅ Final Answer:\n")
        print(answer)
        return

    # For complex reasoning, proceed with breakdown and agents
    subquestions = split_question(llm, question)
    print("\n🔍 Sub-questions:")
    for sq in subquestions:
        print("-", sq)

    context_so_far = []

    for subq in subquestions:
        role_prompt = generate_role_prompt(llm, subq)
        classification = classify_subquestion_type(llm, subq)

        context_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in context_so_far])
        full_prompt = f"""{role_prompt}

{"Here is some prior information:" if context_text else ""}

{context_text}

Now answer the following question:
{subq}
"""

        agent = StepByStepMathAgent(llm) if classification == "math-step" else GenericAgent(llm)
        answer = agent.answer(subq, full_prompt)
        context_so_far.append((subq, answer))

    print("\nPartial Answers:")
    for subq, ans in context_so_far:
        print(f"\nQ: {subq}\nA: {ans}")

    final = combine_answers(llm, context_so_far)
    print("\n✅ Final Combined Answer:\n")
    print(final)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter a complex question:\n> ")
    run_reasoning_pipeline(question)
from splitter import split_question
from prompt_generator import generate_role_prompt
from agent import GenericAgent, StepByStepAgent
from combiner import combine_answers
from llm_loader import load_llm
from token_utils import TokenTracker, count_tokens

def classify_question_complexity(llm, question: str, tracker) -> str:
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

def run_reasoning_pipeline(question: str):
    llm = load_llm()
    tracker = TokenTracker()

    print("\nOriginal question:", question)

    classification = classify_question_complexity(llm, question, tracker)
    print("\nGlobal classification:", classification)

    if classification == "simple":
        agent = GenericAgent(llm, tracker)
        answer = agent.answer(question, "You are a helpful expert. Answer clearly and briefly.")
        print("\n✅ Final Answer:\n")
        print(answer)
        tracker.print_summary()
        return

    if classification == "math-step":
        agent = StepByStepAgent(llm, tracker)
        answer = agent.answer(question)
        print("\n✅ Final Answer:\n")
        print(answer)
        tracker.print_summary()
        return

    # For complex reasoning, proceed with breakdown and agents
    strategy = decide_decomposition_strategy(llm, question, tracker)
    print("\nDecomposition strategy:", strategy)

    if strategy == "steps":
        agent = StepByStepAgent(llm, tracker)
        answer = agent.answer(question)
        print("\n✅ Final Answer:\n")
        print(answer)
        tracker.print_summary()
        return

    # Otherwise, default to subquestion breakdown
    subquestions = split_question(llm, question)
    print("\n🔍 Sub-questions:")
    for sq in subquestions:
        print("-", sq)

    context_so_far = []

    for subq in subquestions:
        role_prompt = generate_role_prompt(llm, subq)
        classification = classify_subquestion_type(llm, subq, tracker)

        context_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in context_so_far])
        full_prompt = f"""{role_prompt}

{"Here is some prior information:" if context_text else ""}

{context_text}

Now answer the following question:
{subq}
"""
        tracker.log_reasoning(context_text, count_tokens)
        agent = StepByStepAgent(llm, tracker) if classification == "math-step" else GenericAgent(llm, tracker)
        if isinstance(agent, StepByStepAgent):
            answer = agent.answer(subq) 
        else:
            answer = agent.answer(subq, full_prompt)
        #answer = agent.answer(subq, full_prompt)
        context_so_far.append((subq, answer))

    print("\nPartial Answers:")
    for subq, ans in context_so_far:
        print(f"\nQ: {subq}\nA: {ans}")

    final = combine_answers(llm, context_so_far)
    print("\n✅ Final Combined Answer:\n")
    print(final)
    tracker.print_summary()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter a complex question:\n> ")
    run_reasoning_pipeline(question)
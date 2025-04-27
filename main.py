from splitter import split_question
from prompt_generator import generate_role_prompt
from agent import GenericAgent, StepByStepAgent
from llm_loader import load_llm
from token_utils import TokenTracker, count_tokens
from classifier import classify_question_complexity, decide_decomposition_strategy, classify_subquestion_type, combine_answers

def run_reasoning_pipeline(question: str):
    """
    Main pipeline to process a user's complex question through a multi-agent reasoning system.

    Steps:
    1. Classify the overall complexity of the question (simple, math-step, or reasoning).
    2. Based on classification, select an appropriate answering agent.
    3. If necessary, decide whether to decompose into subquestions or answer step-by-step.
    4. Handle token usage tracking across reasoning stages.
    5. Combine partial answers if subquestions were generated.
    6. Print the final answer and token usage summary.

    Args:
        question (str): The complex user question input.
    """
    llm = load_llm()
    tracker = TokenTracker()

    # Step 1: Classify the global complexity of the question
    classification = classify_question_complexity(llm, question, tracker)

    # Step 2: Handle simple questions directly
    if classification == "simple":
        agent = GenericAgent(llm, tracker)
        answer = agent.answer(question, "You are a helpful expert. Answer clearly and briefly.")
        print(answer)
        tracker.print_summary()
        return
    
    # Step 3: Handle direct math-step problems
    if classification == "math-step":
        agent = StepByStepAgent(llm, tracker)
        answer = agent.answer(question)
        print(answer)
        tracker.print_summary()
        return

    # Step 4: For complex reasoning questions, decide the decomposition strategy
    strategy = decide_decomposition_strategy(llm, question, tracker)

    if strategy == "steps":
        agent = StepByStepAgent(llm, tracker)
        answer = agent.answer(question)
        print(answer)
        tracker.print_summary()
        return

    # Step 5: Otherwise, decompose into sub-questions and solve each individually
    subquestions = split_question(llm, question)

    context_so_far = []

    for subq in subquestions:
        # Generate role-based prompting
        role_prompt = generate_role_prompt(llm, subq)
        
        # Classify the type of each sub-question
        classification = classify_subquestion_type(llm, subq, tracker)

        # Build context from previous sub-answers (for better chaining of logic)
        context_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in context_so_far])
        prior_info = f"Here is some prior information:\n{context_text}\n" if context_text else ""
        full_prompt = f"""{role_prompt}

{prior_info}

Now answer the following question:
{subq}
""" 
        # Track the tokens used for intermediate reasoning context
        tracker.log_reasoning(context_text, count_tokens)

        # Select agent depending on sub-question type
        agent = StepByStepAgent(llm, tracker) if classification == "math-step" else GenericAgent(llm, tracker)

        # Get the answer for the sub-question
        if isinstance(agent, StepByStepAgent):
            answer = agent.answer(subq) 
        else:
            answer = agent.answer(subq, full_prompt)

        context_so_far.append((subq, answer))

    # Step 6: Combine all partial answers into a coherent final answer
    print("\nPartial subquestions and answers to reason the complex question:")
    for subq, ans in context_so_far:
        print(f"\nQ: {subq}\n \nA: {ans}")

    final = combine_answers(llm, context_so_far)
    print("\n✅ Final Combined Answer:\n")
    print(final)

    # Step 7: Print token usage statistics
    tracker.print_summary()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter a complex question:\n> ")
    run_reasoning_pipeline(question)
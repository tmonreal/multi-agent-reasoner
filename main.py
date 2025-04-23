"""
Orchestrates the full process:
- Load LLM
- Ask for a complex question (or read from file)
- Use splitter to break it up
- Use prompt_generator for each subquestion
- Use GenericAgent to answer each
- Use combiner to generate a final answer
"""
from splitter import split_question
from prompt_generator import generate_role_prompt
from agent import GenericAgent
from combiner import combine_answers
from llm_loader import load_llm

def run_reasoning_pipeline(question: str):
    llm = load_llm()
    print("\nOriginal question:", question)

    subquestions = split_question(llm, question)
    print("\nSub-questions:")
    for sq in subquestions:
        print("-", sq)

    agent = GenericAgent(llm)
    responses = []

    for subq in subquestions:
        print(f"\nSub-question: {subq}")
        role_prompt = generate_role_prompt(llm, subq)
        print(f"Generated role prompt: {role_prompt}")
        answer = agent.answer(subq, role_prompt)
        print(f"Answer: {answer}")
        responses.append((subq, answer))

    print("\nPartial Answers:")
    for subq, ans in responses:
        print(f"\nQ: {subq}\nA: {ans}")

    final = combine_answers(llm, responses)
    print("\n✅ Final Combined Answer:\n")
    print(final)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter a complex question:\n> ")
    run_reasoning_pipeline(question)
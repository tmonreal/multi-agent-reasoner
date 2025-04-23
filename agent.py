class GenericAgent:
    def __init__(self, llm):
        self.llm = llm

    def answer(self, question: str, role_prompt: str) -> str:
        full_prompt = f"""{role_prompt}
Be brief and to the point. Avoid over-explaining unless asked.

Question: {question}
Answer:"""
        response = self.llm.invoke(full_prompt)
        return getattr(response, "content", str(response)).strip()

class StepByStepMathAgent(GenericAgent):
    def answer(self, question: str) -> str:
        full_prompt = f"""
You are a helpful and precise engineer or mathematics expert.
Answer the following question step by step using markdown with clear titles for each step.
Always round final numeric answers to 2 decimal places and include units where possible.
If any assumption must be made (e.g., dimensions or standard values), do so clearly and state it.

Question: {question}
Answer:
"""
        response = self.llm.invoke(full_prompt)
        return getattr(response, "content", str(response)).strip()
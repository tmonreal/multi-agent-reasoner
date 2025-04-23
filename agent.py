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


from token_utils import count_tokens

class GenericAgent:
    """
    A generic agent that sends a direct prompt and question to the LLM 
    and returns a concise answer.
    """

    def __init__(self, llm, tracker=None):
        """
        Args:
            llm: The language model to interact with.
            tracker: Optional token tracker to log input/output tokens.
        """
        self.llm = llm
        self.tracker = tracker

    def answer(self, question: str, role_prompt: str) -> str:
        """
        Answers a question using a direct prompting style.
        
        Args:
            question: The user question.
            role_prompt: The role and behavior instruction for the LLM.

        Returns:
            The LLM-generated answer.
        """
        full_prompt = f"""{role_prompt}
Be brief and to the point. Avoid over-explaining unless asked.

Question: {question}
Answer:"""
        response = self.llm.invoke(full_prompt)
        output = getattr(response, "content", str(response)).strip()

        if self.tracker:
            self.tracker.log(full_prompt, output, count_tokens)

        return output


class StepByStepAgent(GenericAgent):
    """
    An agent that solves questions step-by-step using a structured prompt,
    suitable for multi-step math or reasoning tasks.
    """

    def answer(self, question: str) -> str:
        """
        Answers a question in a detailed, step-by-step manner.

        Args:
            question: The user question.

        Returns:
            The LLM-generated step-by-step answer.
        """
        full_prompt = f"""
You are a helpful and precise engineer or mathematics expert.
Answer the following question step by step using markdown with clear titles for each step.
Always round final numeric answers to 2 decimal places and include units where possible.
If any assumption must be made (e.g., dimensions or standard values), do so clearly and state it.

Question: {question}
Answer:
"""
        response = self.llm.invoke(full_prompt)
        output = getattr(response, "content", str(response)).strip()

        if self.tracker:
            self.tracker.log(full_prompt, output, count_tokens)
            self.tracker.log_reasoning(full_prompt, count_tokens)

        return output
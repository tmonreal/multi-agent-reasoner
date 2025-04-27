from transformers import AutoTokenizer

try:
    # Attempt to use the tokenizer from the official LLaMA model for accurate token counting.
    # Note: Access to 'meta-llama/Llama-4-scout-17b-16e-instruct' may be gated and require approval.
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-scout-17b-16e-instruct")
except Exception:
    # Fallback to a publicly available LLaMA-compatible tokenizer for approximate token counting.
    # 'NousResearch/Llama-2-7b-hf' is fully compatible and accessible without special permissions.
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a given text string.

    Args:
        text (str): The input text.

    Returns:
        int: The number of tokens in the text.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

class TokenTracker:
    """
    Class to track token usage across input prompts, model outputs, and internal reasoning steps.
    """
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0

    def log(self, prompt: str, output: str, count_tokens_fn):        
        """
        Log tokens used in input and output of an LLM call.

        Args:
            prompt (str): The input prompt.
            output (str): The model's output.
            count_tokens_fn (callable): The token counting function.
        """
        self.input_tokens += count_tokens_fn(prompt)
        self.output_tokens += count_tokens_fn(output)

    def log_reasoning(self, text: str, count_tokens_fn):
        """
        Log tokens used during internal reasoning steps.

        Args:
            text (str): The reasoning text.
            count_tokens_fn (callable): The token counting function.
        """
        self.reasoning_tokens += count_tokens_fn(text)

    def print_summary(self):
        """
        Print a formatted summary of token usage.
        """
        print("\n------------------------------------------")
        print("\n###🔢 **Token Usage Summary:**")
        print(f"\n - Input tokens: {self.input_tokens}")
        print(f"\n - Output tokens: {self.output_tokens}")
        print(f"\n - Reasoning tokens: {self.reasoning_tokens}")
        total = self.input_tokens + self.output_tokens + self.reasoning_tokens
        print(f"\n - **Total tokens: {total}**")
        print("\n------------------------------------------")

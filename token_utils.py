from transformers import AutoTokenizer

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-4-scout-17b-16e-instruct")
# Use a public tokenizer model just for token counting
tokenizer = AutoTokenizer.from_pretrained("gpt2")

#def count_tokens(text: str) -> int:
#    tokens = tokenizer.encode(text, add_special_tokens=False)
#    return len(tokens)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

class TokenTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0

    def log(self, prompt: str, output: str, count_tokens_fn):
        self.input_tokens += count_tokens_fn(prompt)
        self.output_tokens += count_tokens_fn(output)

    def log_reasoning(self, text: str, count_tokens_fn):
        self.reasoning_tokens += count_tokens_fn(text)

    def print_summary(self):
        print("\n🔢 Token Usage Summary:")
        print(f"📥 Input tokens: {self.input_tokens}")
        print(f"📤 Output tokens: {self.output_tokens}")
        print(f"🧠 Reasoning tokens: {self.reasoning_tokens}")
        total = self.input_tokens + self.output_tokens + self.reasoning_tokens
        print(f"💰 Total tokens: {total}")

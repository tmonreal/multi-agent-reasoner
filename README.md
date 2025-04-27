# 💬 Multi-Agent Reasoner ChatBot

## Modular Multi-Agent System for Complex Question Solving

This project implements a **modular multi-agent reasoning system** to decompose, classify, and solve **complex user questions** by dynamically selecting different expert strategies using **Groq's LLaMA 4 Scout 17B** model.

---

## 🧠 How It Works

1. User submits a question.
2. The system classifies the question:
   - **Simple** (direct answer)
   - **Math-Step** (step-by-step reasoning)
   - **Reasoning** (requires decomposition)
3. Depending on classification:
   - Simple questions are answered directly.
   - Math-step problems use a structured step-by-step agent.
   - Reasoning tasks are either decomposed into **sub-questions** or **steps**.
4. Each sub-question is answered independently, using a role-adapted prompt.
5. Partial answers are combined into a coherent final response.
6. Full token usage statistics are printed and optionally shown in the UI.

---

## 🚀 Running the App Locally

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/tmonreal/multi-agent-reasoner.git
cd multi-agent-reasoner
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create a `.env` file

```env
GROQ_API_KEY=your_groq_api_key
```

Ensure the `.env` file is listed in `.gitignore`.

---

### 3. Run the Flask web app

```bash
export FLASK_APP=app.py
flask run
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 📁 Project Structure

| File/Folder            | Description                                                            |
|------------------------|------------------------------------------------------------------------|
| `app.py`               | Flask app serving the web interface and connecting to reasoning core  |
| `main.py`              | Command-line entry point for multi-agent reasoning                    |
| `agent.py`             | Defines GenericAgent and StepByStepAgent classes                      |
| `classifier.py`        | Functions to classify questions and decide decomposition strategies   |
| `llm_loader.py`        | Loads Groq LLM instance using environment variables                   |
| `prompt_generator.py`  | Generates role-based prompts for expert answering                     |
| `splitter.py`          | Splits complex questions into simpler sub-questions                   |
| `token_utils.py`       | Token counting and tracking utilities                                 |
| `templates/index.html` | Frontend UI for the chatbot                                            |
| `static/bot.jpg`       | Bot image for the web app                                              |
| `demo/demo.mp4`        | Demo video showcasing the reasoning system                            |

---

## 🎥 Demo

### 👀 Quick Preview (GIF)

![Chatbot Demo](demo/demo-tp3.gif)

### 🎞️ Full Demo Video

🔍 [Click to download demo](demo/demo-tp3.webm)

---

## 🙌 Credits

- LLM model by [Meta's LLaMA 4 Scout 17B](https://ai.meta.com/llama/), served via [Groq](https://groq.com)
- Built with [LangChain](https://www.langchain.com) and [Flask](https://flask.palletsprojects.com)
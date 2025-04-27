import markdown
import sys
from markupsafe import Markup
from io import StringIO
from flask import Flask, render_template, request, jsonify
from main import run_reasoning_pipeline  

app = Flask(__name__)

@app.route("/")
def index():
    """
    Serves the main HTML page for the chat interface.
    """
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    """
    Handles incoming user messages via POST,
    runs the multi-agent reasoning pipeline,
    and returns the formatted response.
    """
    user_message = request.form.get("msg")
    if not user_message:
        return jsonify({"answer": "I didn't get your message."})

    try:
        # Capture printed output from run_reasoning_pipeline
        original_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        run_reasoning_pipeline(user_message)
        sys.stdout = original_stdout

        # Format output as HTML using Markdown
        response_text = captured_output.getvalue()
        html_response = Markup(markdown.markdown(response_text))
        return jsonify({"answer": html_response})

    except Exception as e:
        sys.stdout = original_stdout # Ensure stdout is always restored
        print(f"Error occurred: {e}")  # Optional: log error for debugging
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
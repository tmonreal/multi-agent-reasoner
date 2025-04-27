from flask import Flask, render_template, request, jsonify
import markdown
from markupsafe import Markup
from io import StringIO
import sys

from main import run_reasoning_pipeline  # Your core logic

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_message = request.form.get("msg")
    if not user_message:
        return jsonify({"answer": "I didn't get your message."})

    try:
        # Capture printed output from run_reasoning_pipeline
        original_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        run_reasoning_pipeline(user_message)
        sys.stdout = original_stdout

        # Get printed output
        response_text = captured_output.getvalue()
        html_response = Markup(markdown.markdown(response_text))
        return jsonify({"answer": html_response})

    except Exception as e:
        sys.stdout = original_stdout  # Restore stdout in case of error
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
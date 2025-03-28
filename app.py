from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World! (Updated) hmmmmm"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)  # Debug mode enables auto-reload

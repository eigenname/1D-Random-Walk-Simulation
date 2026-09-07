"""Local dev server for the docs/ GitHub Pages site.

Serves docs/ as static files so index.html and friends can be previewed
and iterated on before pushing. Run with `python app.py` and open
http://127.0.0.1:5000
"""
from pathlib import Path
from flask import Flask, send_from_directory

DOCS_DIR = Path(__file__).parent / "docs"
#___________________________________________________________________________________
app = Flask(__name__, static_folder=str(DOCS_DIR), static_url_path="")
#___________________________________________________________________________________
@app.route("/")
def index():
    return send_from_directory(DOCS_DIR, "index.html")
#___________________________________________________________________________________
if __name__ == "__main__":
    app.run(
        # debug=True,
        
        port=5000,
    )

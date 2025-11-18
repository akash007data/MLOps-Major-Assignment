# app.py - minimal Flask app to serve predictions from DecisionTree savedmodel.pth
from flask import Flask, request, render_template_string, redirect, url_for
import joblib
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("notebooks", "models", "savedmodel.pth")

# Load model at startup
clf = joblib.load(MODEL_PATH)

INDEX_HTML = """
<!doctype html>
<title>Olivetti Face Predictor</title>
<h2>Upload a gray-scale 64x64 image (PNG/JPG) to predict the class</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if pred is defined %}
  <h3>Predicted class: {{pred}}</h3>
{% endif %}
"""

def preprocess_image_file(file_stream):
    img = Image.open(io.BytesIO(file_stream)).convert("L")
    img = img.resize((64, 64))
    arr = np.array(img, dtype=np.float32)
    arr = arr / 255.0
    return arr.reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        content = file.read()
        x = preprocess_image_file(content)
        pred = int(clf.predict(x)[0])
        return render_template_string(INDEX_HTML, pred=pred)
    return render_template_string(INDEX_HTML)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


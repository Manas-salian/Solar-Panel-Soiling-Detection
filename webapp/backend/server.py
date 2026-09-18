"""
Solar Panel Soiling Detection - Backend API
=============================================
Serves the frontend dashboard and provides two endpoints:
  POST /api/predict  -> run the trained model on an uploaded image
  GET  /api/results  -> return metrics + links to saved plot images

Run from the project root (with your venv active):
    python webapp/backend/server.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import json
import io

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import tensorflow as tf

# Allow importing from src/ (project root's config)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config import IMG_SIZE, CLASS_NAMES, OUTPUT_DIR  # noqa: E402

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, OUTPUT_DIR)
MODEL_PATH = os.path.join(OUTPUTS_DIR, "solar_soiling_final_model.keras")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "metrics.json")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

_model = None  # lazy-loaded on first prediction request


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            return None
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    """Serve saved plot images (training curves, confusion matrix)."""
    return send_from_directory(OUTPUTS_DIR, filename)


# ---------------------------------------------------------------------------
# API: prediction
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    model = get_model()
    if model is None:
        return jsonify({
            "error": f"No trained model found. Run `python main.py` first."
        }), 400

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read the uploaded file as an image."}), 400

    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = float(model.predict(img_array, verbose=0)[0][0])
    predicted_class = CLASS_NAMES[int(prediction > 0.5)]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return jsonify({
        "label": predicted_class,
        "confidence": round(confidence * 100, 1),
    })


# ---------------------------------------------------------------------------
# API: results / metrics
# ---------------------------------------------------------------------------
@app.route("/api/results")
def results():
    response = {
        "model_available": os.path.exists(MODEL_PATH),
        "metrics": None,
        "images": {},
    }

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            response["metrics"] = json.load(f)

    for key, filename in [
        ("training_curve_initial", "training_curves_initial.png"),
        ("training_curve_finetune", "training_curves_finetune.png"),
        ("confusion_matrix", "confusion_matrix.png"),
    ]:
        if os.path.exists(os.path.join(OUTPUTS_DIR, filename)):
            response["images"][key] = f"/outputs/{filename}"

    return jsonify(response)


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model path:   {MODEL_PATH} (exists: {os.path.exists(MODEL_PATH)})")
    print("Starting server at http://localhost:5000 ...")
    app.run(debug=True, port=5000)

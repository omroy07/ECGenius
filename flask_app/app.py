"""Flask application for the ECG heart disease monitoring system.

This module exposes a small REST/HTML interface around the trained
ECG classifier:

- ``GET /``: Upload page for single-ECG inference.
- ``GET /dashboard``: Training metrics and visualisations.
- ``POST /predict``: Accept uploaded ECG files and run ``predict_single``.
- ``GET /api/training-log``: Return ``logs/training_log.csv`` as JSON.
- ``GET /api/metrics``: Return ``logs/evaluation_results.json`` as JSON.
- ``GET /health``: Simple health check endpoint.

All filesystem interactions use :class:`pathlib.Path` for
cross-platform compatibility.
"""

from __future__ import annotations

# ✅ ADD THESE IMPORTS AT TOP
import joblib
import numpy as np
from ai_assistant import ask_health_assistant

import csv
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Python path setup so we can import from the top-level ``src`` package
# when running this app directly (e.g. ``python flask_app/app.py``).
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import CONFIG, UNIFIED_LABELS  # type: ignore  # noqa: E402
from src.predict import predict_single  # type: ignore  # noqa: E402

# ==============================
# LOAD HEART MODEL + CHAT MEMORY
# ==============================
heart_model = joblib.load('D:\\MLPROJECTS\\Elderly\\health_ECG\\flask_app\\model\\heart_model.pkl')
conversation_history = []

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)


def _load_thresholds() -> List[float]:
    """Load per-label decision thresholds from evaluation metrics.

    Thresholds are read from ``logs/evaluation_results.json`` if it
    exists and contains an ``"optimal_thresholds"`` field. Otherwise a
    default threshold of ``0.5`` is used for all labels.

    Returns
    -------
    list of float
        Threshold for each label in :data:`UNIFIED_LABELS`.
    """

    default = [0.5] * len(UNIFIED_LABELS)
    metrics_path = CONFIG.BASE_DIR / "logs" / "evaluation_results.json"  # type: ignore[arg-type]

    if not metrics_path.exists():
        return default

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        values = data.get("optimal_thresholds")
        if isinstance(values, list) and len(values) == len(UNIFIED_LABELS):
            return [float(v) for v in values]
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("Failed to load thresholds from %s: %s", metrics_path, exc)

    return default


@app.route("/health", methods=["GET"])
def health() -> Any:
    """Simple health check endpoint.

    Returns a JSON object with service status and the configured
    computation device.
    """

    return jsonify({"status": "ok", "device": CONFIG.DEVICE})


@app.route("/", methods=["GET"])
def index() -> Any:
    """Render the ECG upload page."""

    return render_template("index.html")


@app.route("/dashboard", methods=["GET"])
def dashboard() -> Any:
    """Render the training metrics dashboard."""

    return render_template("dashboard.html")


@app.route("/api/training-log", methods=["GET"])
def api_training_log() -> Any:
    """Return the training log CSV as JSON.

    The response format is::

        {"epochs": [{"epoch": 1, "train_loss": ...}, ...]}
    """

    log_path = CONFIG.LOG_PATH  # type: ignore[assignment]
    if not log_path.exists():
        return jsonify({"epochs": []})

    rows: List[Dict[str, Any]] = []
    try:
        with log_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception as exc:  # pragma: no cover - IO-dependent
        logger.error("Failed to read training log from %s: %s", log_path, exc)
        return jsonify({"epochs": []}), 500

    return jsonify({"epochs": rows})


@app.route("/api/metrics", methods=["GET"])
def api_metrics() -> Any:
    """Return evaluation metrics as JSON.

    This endpoint simply exposes the contents of
    ``logs/evaluation_results.json`` if it exists; otherwise, it returns
    an empty JSON object.
    """

    metrics_path = CONFIG.BASE_DIR / "logs" / "evaluation_results.json"  # type: ignore[arg-type]
    if not metrics_path.exists():
        return jsonify({})

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - IO-dependent
        logger.error("Failed to read metrics from %s: %s", metrics_path, exc)
        return jsonify({}), 500

    return jsonify(data)


@app.route("/predict", methods=["POST"])
def predict_route() -> Any:
    """Handle ECG file uploads and run model inference.

    The form is expected to submit one or more files under the field
    name ``"files"``. Typical usage is to upload a ``.hea`` header and
    its corresponding ``.mat``/``.dat`` waveform file(s). This handler
    stores the uploads in a temporary directory, selects the most
    appropriate file to pass to :func:`predict_single`, and renders the
    result page.
    """

    files = request.files.getlist("files")
    if not files:
        return redirect(url_for("index"))

    thresholds = _load_thresholds()

    with tempfile.TemporaryDirectory(prefix="ecg_upload_") as tmpdir:
        tmp_path = Path(tmpdir)
        saved_paths: List[Path] = []

        for file_storage in files:
            filename = secure_filename(file_storage.filename or "")
            if not filename:
                continue
            dest = tmp_path / filename
            try:
                file_storage.save(dest)
                saved_paths.append(dest)
            except Exception as exc:  # pragma: no cover - IO-dependent
                logger.error("Failed to save uploaded file %s: %s", filename, exc)

        if not saved_paths:
            return redirect(url_for("index"))

        # Prefer header files, then MAT, then DAT
        header_files = [p for p in saved_paths if p.suffix.lower() == ".hea"]
        mat_files = [p for p in saved_paths if p.suffix.lower() == ".mat"]
        dat_files = [p for p in saved_paths if p.suffix.lower() == ".dat"]

        ecg_file: Path
        if header_files:
            ecg_file = header_files[0]
        elif mat_files:
            ecg_file = mat_files[0]
        elif dat_files:
            ecg_file = dat_files[0]
        else:
            # Unsupported upload combination; redirect to index for now.
            return redirect(url_for("index"))

        try:
            result = predict_single(ecg_file, CONFIG.MODEL_SAVE_PATH, thresholds)
        except Exception as exc:  # pragma: no cover - inference-dependent
            logger.error("Prediction failed for %s: %s", ecg_file, exc)
            return render_template("result.html", error=str(exc), labels=UNIFIED_LABELS)

    return render_template("result.html", result=result, labels=UNIFIED_LABELS)

# ==============================
# 🤖 AI HEALTH CHAT
# ==============================
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({"response": "Please enter a message."})

        # Store user message
        conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Get AI response
        ai_reply = ask_health_assistant(conversation_history)

        # Store AI response
        conversation_history.append({
            "role": "assistant",
            "content": ai_reply
        })

        return jsonify({"response": ai_reply})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})
    
@app.route('/predict-heart', methods=['POST'])
def predict_heart():
    try:
        data = request.get_json()

        features = np.array([
            data['age'],
            data['sex'],
            data['cp'],
            data['trestbps'],
            data['chol'],
            data['fbs'],
            data['restecg'],
            data['thalach'],
            data['exang'],
            data['oldpeak'],
            data['slope'],
            data['ca'],
            data['thal']
        ]).reshape(1, -1)

        prediction = heart_model.predict(features)[0]

        return jsonify({
            "prediction": int(prediction),
            "result": "Heart Disease Detected" if prediction == 1 else "Normal"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=5000, debug=True)

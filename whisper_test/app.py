"""
Simple Whisper ASR test server.
Usage:
    pip install flask openai-whisper
    module load Miniconda3 && source activate vllm
    python whisper_test/app.py
Then open http://localhost:5000 in your browser.
"""

import os
import tempfile
from flask import Flask, request, jsonify, send_from_directory

# Ensure local ffmpeg is visible to Whisper (in case conda env doesn't have it)
os.environ["PATH"] = "/home/yuan0165/ffmpeg-7.0.2-amd64-static:" + os.environ.get("PATH", "")

import whisper

app = Flask(__name__, static_folder=".")

# Load model once at startup (use "base" for speed; swap to "medium"/"large" for accuracy)
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
print(f"Loading Whisper model: {MODEL_SIZE} ...")
model = whisper.load_model(MODEL_SIZE)
print("Model ready.")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No audio file received"}), 400

    # Save to a temp file so Whisper can read it
    suffix = ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        audio.save(tmp_path)

    try:
        result = model.transcribe(tmp_path, language="zh")
        text = result["text"].strip()
    finally:
        os.unlink(tmp_path)

    return jsonify({"text": text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

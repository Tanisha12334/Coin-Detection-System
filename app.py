"""
=============================================================
  Automatic Coin Detection & Counting — Flask Backend
  Subject: Computer Vision and Applications
=============================================================
"""

import os, io, base64, json
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── App setup ──────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)  # allow cross-origin requests from the frontend

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Tuneable detection defaults ─────────────────────────────
DEFAULTS = dict(
    blur_kernel = 7,
    morph_kernel = 5,
    min_radius = 28,
    max_radius = 250,
    dp = 1.2,
    min_dist = 55,
    param1 = 50,
    param2 = 40,
)


# ── Helpers ─────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ndarray_to_b64(img: np.ndarray) -> str:
    """Convert OpenCV image (BGR) to base64 PNG string for JSON transport."""
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


# ── Core CV pipeline ────────────────────────────────────────
def process_image(img: np.ndarray, cfg: dict) -> dict:
    """
    Full processing pipeline.
    Returns a dict with:
      - coin_count          : int
      - steps               : dict of base64 images for each stage
      - annotated           : base64 final annotated image
      - coins               : list of {x, y, r} dicts
    """
    bk = cfg["blur_kernel"]
    mk = cfg["morph_kernel"]
    bk = bk if bk % 2 == 1 else bk + 1    # must be odd
    mk = mk if mk % 2 == 1 else mk + 1

    # ── Stage 1: Greyscale ──────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Stage 2: Gaussian blur ──────────────────────────────
    blurred = cv2.GaussianBlur(gray, (bk, bk), 0)

    # ── Stage 3: Otsu threshold ─────────────────────────────
    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Stage 4: Morphological clean-up ────────────────────
    kernel = np.ones((mk, mk), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=2)
    morph  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Stage 5: Detection (Hough + Contours merged) ───────
    coins = _detect_coins(gray, morph, cfg)

    # ── Stage 6: Annotate ───────────────────────────────────
    annotated = _annotate(img.copy(), coins)

    return {
        "coin_count": len(coins),
        "coins": [{"x": int(x), "y": int(y), "r": int(r)} for x, y, r in coins],
        "steps": {
            "original" : ndarray_to_b64(img),
            "grayscale": ndarray_to_b64(cv2.cvtColor(gray,  cv2.COLOR_GRAY2BGR)),
            "blurred"  : ndarray_to_b64(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)),
            "threshold": ndarray_to_b64(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)),
            "morphology":ndarray_to_b64(cv2.cvtColor(morph,  cv2.COLOR_GRAY2BGR)),
            "annotated" :ndarray_to_b64(annotated),
        }
    }


def _detect_coins(gray: np.ndarray, morph: np.ndarray, cfg: dict) -> list:
    MIN_R, MAX_R = cfg["min_radius"], cfg["max_radius"]
    coins = []

   
    # Method B: Contour-based (handles touching coins)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (np.pi * MIN_R**2 < area < np.pi * MAX_R**2 * 4):
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        cx, cy, r = int(cx), int(cy), int(r)
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / perim**2
        if circularity >= 0.55 and MIN_R <= r <= MAX_R:
            coins.append((cx, cy, r))

    return _deduplicate(coins)


def _deduplicate(coins: list) -> list:
    unique = []
    for (x1, y1, r1) in coins:
        dup = any(
            np.hypot(x1 - x2, y1 - y2) < (r1 + r2) / 2
            for (x2, y2, r2) in unique
        )
        if not dup:
            unique.append((x1, y1, r1))
    return unique


def _annotate(img: np.ndarray, coins: list) -> np.ndarray:
    COLORS = {
        "circle": (0, 230, 118),
        "dot"   : (255, 82,  82),
        "label" : (255, 214,   0),
        "banner": (18,  18,  30),
        "text"  : (0,  230, 118),
    }

    for idx, (x, y, r) in enumerate(coins, start=1):
        cv2.circle(img, (x, y), r, COLORS["circle"], 3)
        cv2.circle(img, (x, y), 5, COLORS["dot"], -1)
        cv2.putText(img, str(idx), (x - 11, y + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    COLORS["label"], 2, cv2.LINE_AA)

    # Banner
    bh = 60
    banner = np.full((bh, img.shape[1], 3), COLORS["banner"], np.uint8)
    label  = f"Coins Detected: {len(coins)}"
    fs     = min(1.1, img.shape[1] / 640)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    tx = (img.shape[1] - tw) // 2
    cv2.putText(banner, label, (tx, (bh + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, COLORS["text"], 2, cv2.LINE_AA)
    return np.vstack([banner, img])


# ── Routes ──────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Read image bytes directly (no disk write needed)
    buf = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Resize if too large (keeps processing fast)
    h, w = img.shape[:2]
    if max(h, w) > 1800:
        scale = 1800 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Collect config from form or use defaults
    cfg = {k: float(request.form.get(k, v)) for k, v in DEFAULTS.items()}
    cfg["blur_kernel"]  = int(cfg["blur_kernel"])
    cfg["morph_kernel"] = int(cfg["morph_kernel"])
    cfg["min_radius"]   = int(cfg["min_radius"])
    cfg["max_radius"]   = int(cfg["max_radius"])

    result = process_image(img, cfg)
    return jsonify(result)


if __name__ == "__main__":
    print("\n  🪙  Coin Detector running at  http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)

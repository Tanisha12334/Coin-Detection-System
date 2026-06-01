# 🪙 CoinVision — Automatic Coin Detection & Counting
### Subject: Computer Vision and Applications
> **Stack:** Python · OpenCV · Flask · HTML/CSS/JavaScript

---

## 📁 Folder Structure

```
coin_app/
│
├── app.py                  ← Flask backend (CV pipeline + API)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── frontend/
│   └── index.html          ← Complete frontend (HTML + CSS + JS)
│
└── uploads/                ← Auto-created; temporary upload buffer
```

---

## ⚡ Quick Setup (Step by Step)

### Step 1 — Clone / Download the project
Place all files in a folder called `coin_app/`.

### Step 2 — Create a virtual environment (recommended)
```bash
cd coin_app
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install flask flask-cors opencv-python numpy werkzeug
```

### Step 4 — Run the server
```bash
python app.py
```

You should see:
```
🪙  Coin Detector running at  http://127.0.0.1:5000
```

### Step 5 — Open in browser
Visit: **http://127.0.0.1:5000**

---

## 🎯 How to Use

1. **Drag and drop** a coin image (or click "Browse File")
2. Preview appears instantly
3. Optionally expand **Detection Parameters** to tune sensitivity
4. Click **🔍 Detect Coins**
5. View:
   - Total coin count
   - Statistics (avg radius, largest, smallest coin)
   - Full 6-stage processing pipeline images
   - Final annotated image with circles drawn around each coin
6. Click **⬇ Download Result** to save the output

---

## ⚙️ Detection Parameters (in UI)

| Param | Default | Effect |
|-------|---------|--------|
| Blur Kernel | 7 | Higher = smoother (helps noisy images) |
| Morph Kernel | 5 | Controls morphological clean-up size |
| Min Radius | 20px | Ignore circles smaller than this |
| Max Radius | 250px | Ignore circles larger than this |
| Sensitivity | 28 | **Lower → detects more**; Higher → more strict |
| Min Dist | 40px | Minimum gap between coin centres |

**If count is wrong:**
- Too many detected → raise Sensitivity slider
- Too few detected → lower Sensitivity slider

---

## 🔄 CV Pipeline Explained

| Stage | Operation | Purpose |
|-------|-----------|---------|
| 1 | Grayscale | Reduce 3-channel to 1 for faster processing |
| 2 | Gaussian Blur | Remove high-frequency noise |
| 3 | Otsu Threshold | Automatic binary segmentation (no manual threshold) |
| 4 | Morphological Opening | Remove tiny noise specks |
| 5 | Morphological Closing | Fill holes inside coin blobs |
| 6 | Hough Circles | Detect round patterns via accumulator voting |
| 7 | Contour + Circularity | Catch touching / overlapping coins |
| 8 | Deduplication | Merge results from both methods |

---

## ❌ Common Errors & Fixes

### `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### `Address already in use` (port 5000)
```bash
# Kill whatever is using port 5000, then restart
# Or change port in app.py: app.run(port=5001)
```

### `CORS error` in browser console
Make sure `flask-cors` is installed and Flask is running on port **5000**.  
The frontend fetch URL is `/detect` (relative) — this works when Flask serves the HTML.

### Page loads but Detect button does nothing
Open browser DevTools (F12) → Console tab → look for errors.  
Most common: Flask not running, or running on a different port.

### Wrong coin count
Use the **Sensitivity** slider in the UI. Lower = more coins detected, Higher = fewer.  
For very small coins, reduce **Min Radius**. For very large coins, raise **Max Radius**.

### Image uploads fail (>16MB)
Resize your image before uploading, or increase `MAX_CONTENT_LENGTH` in `app.py`.

---

## 🖼️ Best Images for Testing

- Coins on a **plain white or dark background**
- **Good lighting**, minimal shadows
- **No overlapping** (for highest accuracy)
- Resolution: **500px – 1500px** ideal

You can search "coins on white background" and download any image.

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| CV Library | OpenCV 4.x |
| Web Framework | Flask 3.x |
| CORS | Flask-CORS |
| Frontend | Vanilla HTML5 / CSS3 / JavaScript |
| Fonts | Syne + DM Mono (Google Fonts) |

---

## 📊 API Reference

### `POST /detect`

**Form Data:**
| Field | Type | Description |
|-------|------|-------------|
| `image` | file | Image file (jpg/png/webp/bmp) |
| `blur_kernel` | int | Gaussian blur kernel (odd) |
| `morph_kernel` | int | Morphological kernel |
| `min_radius` | int | Min coin radius (px) |
| `max_radius` | int | Max coin radius (px) |
| `param2` | int | Hough accumulator threshold |
| `min_dist` | int | Min distance between centres |

**Response (JSON):**
```json
{
  "coin_count": 7,
  "coins": [
    {"x": 120, "y": 95, "r": 42},
    ...
  ],
  "steps": {
    "original":   "<base64 png>",
    "grayscale":  "<base64 png>",
    "blurred":    "<base64 png>",
    "threshold":  "<base64 png>",
    "morphology": "<base64 png>",
    "annotated":  "<base64 png>"
  }
}
```

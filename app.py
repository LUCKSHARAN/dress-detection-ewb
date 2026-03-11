import os
import base64
from collections import deque

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

# ── HuggingFace Spaces writable config dir ────────────────────────
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/ultralytics')

# ── App setup ─────────────────────────────────────────────────────
app = Flask(__name__)

# ── Model loading ─────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
model = YOLO(MODEL_PATH)
print(f"[INFO] Model loaded from: {MODEL_PATH}")

# ── Dress code logic ──────────────────────────────────────────────
formal_mandatory = {"formal_shirt_tucked", "formal_pant", "formal_shoes"}
formal_optional  = {"belt", "blazer"}
informal_items   = {"formal_shirt_untucked", "tshirt", "informal_pant",
                    "informal_shoes", "cap", "band"}

BUFFER_SIZE = 12
decision_buffer = deque(maxlen=BUFFER_SIZE)

# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data      = request.get_json(force=True)
        img_b64   = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Resize — same as original OpenCV script
        frame = cv2.resize(frame, (960, 540))
        h, w  = frame.shape[:2]

        # YOLO inference
        results = model(frame, conf=0.25, verbose=False)[0]

        detected_classes = set()
        boxes_out        = []

        for box in results.boxes:
            cls_id     = int(box.cls[0])
            cls_name   = model.names[cls_id]
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            detected_classes.add(cls_name)
            boxes_out.append({
                "x1":    x1 / w,   # normalized 0-1
                "y1":    y1 / h,
                "x2":    x2 / w,
                "y2":    y2 / h,
                "label": cls_name,
                "conf":  round(conf_score, 2),
            })

        # Rule 1: any informal item → INFORMAL
        if detected_classes.intersection(informal_items):
            decision = "INFORMAL"
        # Rule 2: all mandatory formal items present → FORMAL
        elif formal_mandatory.issubset(detected_classes):
            decision = "FORMAL"
        else:
            decision = "INFORMAL"

        # Temporal smoothing — 12 frame buffer
        decision_buffer.append(decision)
        final_verdict = max(set(decision_buffer), key=decision_buffer.count)

        return jsonify({
            "verdict":  final_verdict,
            "detected": list(detected_classes),
            "boxes":    boxes_out,
        })

    except Exception as e:
        print(f"[ERROR] /detect: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # HuggingFace Spaces requires port 7860
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, threaded=True, host="0.0.0.0", port=port)
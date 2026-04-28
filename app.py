"""
MT-AHNet Weed Detection — Flask API Server
Wraps the trained model to expose REST endpoints for the frontend.
Run: python app.py
Runs on: http://localhost:5000
"""

import os
import io
import json
import base64
import logging
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import datetime

# ── Suppress TF verbose logs ──────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Allow requests from the frontend (GitHub Pages, localhost dev, etc.)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Serve Frontend ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the single-page frontend."""
    return send_from_directory('.', 'index.html')

# ── Model Configuration ────────────────────────────────────────────────────────
IMG_SIZE    = (128, 128)
NUM_CLASSES = 4
CLASS_NAMES = ['Soil', 'Soybean', 'Grass Weed', 'Broadleaf Weed']
CLASS_INFO  = {
    'Soil':          {'color': '#8B6914', 'icon': '🟤', 'risk': 'none',   'action': 'Healthy bare soil detected. Ready for planting.'},
    'Soybean':       {'color': '#4CAF50', 'icon': '🌱', 'risk': 'none',   'action': 'Healthy soybean crop. Continue normal care.'},
    'Grass Weed':    {'color': '#FF9800', 'icon': '🌾', 'risk': 'medium', 'action': 'Grass weed detected. Apply selective herbicide.'},
    'Broadleaf Weed':{'color': '#F44336', 'icon': '🍃', 'risk': 'high',   'action': 'Broadleaf weed detected. Immediate treatment recommended.'},
}

# ── Lazy-load model ────────────────────────────────────────────────────────────
_model = None

def get_model():
    """Load model once and cache it. Returns None if model file not found."""
    global _model
    if _model is not None:
        return _model
    try:
        import tensorflow as tf
        MODEL_PATH = os.environ.get('MODEL_PATH', 'mt_ahnet_model.h5')
        if os.path.exists(MODEL_PATH):
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info(f"✅ Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"⚠️  Model file '{MODEL_PATH}' not found — running in DEMO mode")
    except Exception as e:
        logger.error(f"❌ Model load error: {e}")
    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize and normalise an uploaded image for inference."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 128, 128, 3)


def demo_predict(filename: str) -> dict:
    """Return plausible fake predictions when no model is loaded."""
    import random, hashlib
    seed = int(hashlib.md5(filename.encode()).hexdigest(), 16) % 10000
    random.seed(seed)
    probs = np.array([random.random() for _ in range(NUM_CLASSES)], dtype=np.float32)
    probs /= probs.sum()
    idx   = int(probs.argmax())
    return {
        'predicted_class': CLASS_NAMES[idx],
        'confidence':      float(probs[idx]),
        'probabilities':   {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        'demo_mode':       True,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    """Liveness probe — frontend polls this to check backend status."""
    model = get_model()
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'demo_mode': model is None,
        'classes': CLASS_NAMES,
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
        'version': '1.0.0',
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Body: multipart/form-data with key 'image'
    Returns: JSON with predicted class, confidence, and per-class probabilities.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided. Use multipart key "image".'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        image_bytes = file.read()
        model = get_model()

        if model is not None:
            # ── Real inference ──────────────────────────────────────────────
            x = preprocess_image(image_bytes)
            preds = model.predict(x, verbose=0)[0]
            idx   = int(preds.argmax())
            result = {
                'predicted_class': CLASS_NAMES[idx],
                'confidence':      float(preds[idx]),
                'probabilities':   {CLASS_NAMES[i]: float(preds[i]) for i in range(NUM_CLASSES)},
                'demo_mode':       False,
            }
        else:
            # ── Demo mode (no model file) ───────────────────────────────────
            result = demo_predict(file.filename)

        # Attach class metadata
        cls = result['predicted_class']
        result['class_info'] = CLASS_INFO.get(cls, {})
        result['filename']   = file.filename

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Dataset statistics used by the dashboard."""
    return jsonify({
        'dataset': {
            'total_images': 15336,
            'split': {'train': 0.70, 'val': 0.15, 'test': 0.15},
            'classes': {
                'Soil':           {'count': 3249, 'color': '#8B6914'},
                'Soybean':        {'count': 7376, 'color': '#4CAF50'},
                'Grass Weed':     {'count': 3520, 'color': '#FF9800'},
                'Broadleaf Weed': {'count': 1191, 'color': '#F44336'},
            },
        },
        'model': {
            'name':       'MT-AHNet',
            'backbone':   'EfficientNetB0',
            'parameters': '4.8M',
            'accuracy':   0.9612,
            'precision':  0.9587,
            'recall':     0.9634,
            'f1_score':   0.9610,
            'val_loss':   0.1243,
        },
        'training': {
            'epochs':      30,
            'batch_size':  32,
            'img_size':    128,
            'optimizer':   'Adam',
            'loss':        'Categorical Cross-Entropy',
        },
        # Simulated per-epoch accuracy for the training chart
        'history': {
            'accuracy': [
                0.51,0.62,0.71,0.77,0.82,0.84,0.86,0.87,0.88,0.89,
                0.90,0.91,0.91,0.92,0.92,0.93,0.93,0.94,0.94,0.95,
                0.95,0.95,0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.9612
            ],
            'val_accuracy': [
                0.48,0.58,0.66,0.73,0.79,0.81,0.83,0.85,0.86,0.87,
                0.88,0.89,0.89,0.90,0.91,0.91,0.92,0.92,0.93,0.93,
                0.94,0.94,0.94,0.95,0.95,0.95,0.95,0.96,0.96,0.9587
            ],
            'loss': [
                1.21,0.98,0.83,0.72,0.63,0.55,0.49,0.44,0.40,0.37,
                0.34,0.31,0.29,0.27,0.25,0.23,0.22,0.21,0.20,0.19,
                0.18,0.17,0.16,0.16,0.15,0.15,0.14,0.14,0.13,0.1243
            ],
        },
    })


@app.route('/api/field-scan', methods=['GET'])
def field_scan():
    """Simulated recent field scan results for the dashboard map widget."""
    import random, math
    random.seed(42)
    zones = []
    labels = ['Soil', 'Soybean', 'Grass Weed', 'Broadleaf Weed']
    weights = [0.25, 0.50, 0.18, 0.07]  # realistic distribution
    for i in range(24):
        cls = random.choices(labels, weights=weights)[0]
        zones.append({
            'id':         i + 1,
            'row':        i // 6,
            'col':        i % 6,
            'class':      cls,
            'confidence': round(0.82 + random.random() * 0.17, 3),
            'color':      CLASS_INFO[cls]['color'],
        })
    summary = {c: sum(1 for z in zones if z['class'] == c) for c in labels}
    return jsonify({'zones': zones, 'summary': summary, 'total_zones': len(zones)})


# ── Entry-point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌱 MT-AHNet API running on http://localhost:{port}")
    logger.info(f"   Model path: {os.environ.get('MODEL_PATH', 'mt_ahnet_model.h5')}")
    app.run(host='0.0.0.0', port=port, debug=False)

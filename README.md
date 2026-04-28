# 🌱 MT-AHNet — Weed Detection in Soybean Crops

> **Multi-Task Attention-Augmented Hybrid Network** · EfficientNetB0 + Attention Gates  
> Minor Project · SRMIST-KTR · 21CSP302L

A full-stack agricultural AI application that classifies soybean field images into four categories:  
**Soil · Soybean · Grass Weed · Broadleaf Weed** — with 96.1% test accuracy.

---

## 📁 Project Structure

```
weed-detection-app/
├── backend/
│   ├── app.py              ← Flask REST API (wraps the ML model)
│   ├── requirements.txt    ← Python dependencies
│   └── mt_ahnet_model.h5   ← (You supply this — your trained model)
├── frontend/
│   └── index.html          ← Complete single-file frontend
└── README.md
```

---

## 🖥️ Running Locally

### Step 1 — Set up the Backend

**Prerequisites:** Python 3.9+

```bash
cd backend

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Point to your trained model weights
export MODEL_PATH=path/to/mt_ahnet_model.h5

# Start the server
python app.py
```

The API will run at **http://localhost:5000**

> **No model file?** The server automatically enters **Demo Mode** — it returns
> realistic fake predictions so you can explore the UI without the weights.

### Step 2 — Run the Frontend

The frontend is a single HTML file. Open it in any way:

```bash
# Option A: just open in browser
open frontend/index.html

# Option B: serve via Python (avoids any CORS edge-cases)
cd frontend
python -m http.server 3000
# then visit http://localhost:3000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Backend liveness + model status |
| POST | `/api/predict` | Upload image → classification result |
| GET | `/api/stats` | Model metrics + training history |
| GET | `/api/field-scan` | Simulated 24-zone field scan |

---

## 🚀 Deployment

### Why not GitHub Pages alone?

GitHub Pages serves **static files only** — it cannot run a Python Flask server.  
You need to deploy the backend to a separate service. Below are two recommended approaches.

---

### Option A: Render (Free tier, recommended)

#### Deploy Backend to Render

1. Push your code to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/weed-detection-app.git
   git push -u origin main
   ```

2. Go to [render.com](https://render.com) → **New → Web Service**

3. Connect your GitHub repo, then set:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Environment Variable:** `MODEL_PATH = mt_ahnet_model.h5`

4. Click **Deploy**. Note the URL, e.g. `https://weed-detection-api.onrender.com`

#### Deploy Frontend to GitHub Pages

1. Edit `frontend/index.html` — find this line near the top of the `<script>` section:
   ```js
   const API_BASE = window.API_BASE || 'http://localhost:5000';
   ```
   Replace with your Render URL:
   ```js
   const API_BASE = window.API_BASE || 'https://weed-detection-api.onrender.com';
   ```

2. Push the updated file to GitHub.

3. Go to your repo → **Settings → Pages → Source: Deploy from branch → `/` (root) or `/frontend`**

4. Your site will be live at `https://YOUR_USERNAME.github.io/weed-detection-app/`

---

### Option B: Railway

#### Backend on Railway

```bash
# Install Railway CLI
npm i -g @railway/cli
railway login

cd backend
railway init
railway up
```
Note the Railway URL and update `API_BASE` in `index.html` as above.

#### Frontend stays on GitHub Pages (same steps as Option A).

---

### Option C: Local Network Demo

Run both on the same machine and share via local IP:

```bash
# Backend
cd backend && python app.py         # http://0.0.0.0:5000

# Frontend (edit API_BASE to your LAN IP first)
cd frontend && python -m http.server 3000
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | MT-AHNet (EfficientNetB0 + Attention Gates) |
| Input size | 128 × 128 RGB |
| Classes | Soil, Soybean, Grass Weed, Broadleaf Weed |
| Parameters | ~4.8M |
| Test Accuracy | **96.12%** |
| F1 Score | **96.10%** |
| Training | Two-phase (frozen backbone → fine-tune) |
| Loss | Weighted Categorical Cross-Entropy |

### Dataset

- **Source:** Kaggle — [Weed Detection in Soybean Crops](https://www.kaggle.com/datasets/fpeccia/weed-detection-in-soybean-crops)
- **Total images:** 15,336
  - Soil: 3,249 · Soybean: 7,376 · Grass: 3,520 · Broadleaf: 1,191
- **Split:** 70% train / 15% val / 15% test

---

## ✨ Frontend Features

- **🔬 Live image classification** — upload any crop field image
- **🎯 Confidence bars** — per-class probability breakdown
- **📊 Model dashboard** — accuracy, precision, recall, F1, training curves
- **🫘 Soybean growth tracker** — phenology stages + weed risk by stage
- **🗺️ Field zone map** — 24-zone interactive overlay with hover tooltips
- **📱 Fully responsive** — works on mobile, tablet, desktop
- **🔌 Graceful offline** — demo mode if backend is unreachable

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| "Backend Offline" in nav | Start `python backend/app.py` and refresh |
| CORS error in console | Ensure `flask-cors` is installed; it's already configured in `app.py` |
| Model not loading | Check `MODEL_PATH` env var points to your `.h5` file |
| Render cold start slow | Free Render instances spin down after 15 min; first request may take ~30s |
| GitHub Pages 404 | Make sure GitHub Pages is pointed to the correct folder |

---

## 📜 License

MIT — Free for academic and research use.  
Built for SRMIST-KTR Minor Project 21CSP302L.

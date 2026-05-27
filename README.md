# DeepScan — Deepfake Image Detection System

A full-stack web application for detecting deepfake facial images, built on the
methodology from:

> *"Optimized Preprocessing with Deep Learning for Robust Low-Quality Deepfake
> Image Detection"* — Anand et al., 2024

The pipeline achieves **97.1% accuracy / AUC 0.991** on the Kaggle 140k Real &
Fake Faces dataset using:
- **MTCNN** (facenet-pytorch) for face detection & cropping
- **ViT-B/16** pretrained on ImageNet-21k (via TIMM) for binary classification
- Two-phase fine-tuning with cosine-annealing LR and class-weighted loss

-------------

## 🚀 Live Demo & Deployment

| Service | URL | Platform |
|---------|-----|----------|
| **Frontend Web App** | [https://vipin-gupta-07.github.io/deepfake-detector/](https://vipin-gupta-07.github.io/deepfake-detector/) | GitHub Pages |
| **Backend API**      | [https://vipingupta04-deepfake-detector.hf.space/](https://vipingupta04-deepfake-detector.hf.space/) | Hugging Face Spaces |

---------------------

## Directory Structure

```
deepfake-detector/
├── backend/
│   ├── main.py             # FastAPI application & /predict endpoint
│   ├── model.py            # ViT-B/16 architecture + inference helper
│   ├── preprocessing.py    # MTCNN face detection + image pipeline
│   ├── requirements.txt    # Python dependencies
│   └── models/
│       └── best_model.pth  # ← Place your fine-tuned weights here
└── frontend/
    └── index.html          # Single-file React-free UI (HTML/CSS/JS)
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| pip | 23+ |
| CUDA (optional) | 11.8 / 12.x |
| A modern browser | Chrome / Firefox / Safari |

---

## 1 — Clone / Download

```bash
git clone <https://github.com/Vipin-Gupta-07/deepfake-detector.git>
cd deepfake-detector
```

---

## 2 — Backend Setup

### 2.1 Create a virtual environment

```bash
cd backend
python -m venv .venv

# Activate
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 2.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users** — PyTorch in `requirements.txt` uses the CPU build by default.
> For CUDA 12.x replace the `torch` and `torchvision` lines with the appropriate
> wheel, or install via:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2.3 Place the fine-tuned weights

Copy your trained `best_model.pth` into `backend/models/`:

```
backend/
└── models/
    └── best_model.pth   ← here
```

The file is expected to contain either:
- a plain `state_dict` (from `torch.save(model.state_dict(), ...)`)
- a checkpoint dict with key `"model_state_dict"` (from
  `torch.save({"model_state_dict": model.state_dict(), ...}, ...)`)

> **No weights yet?**  The server still starts and runs predictions using the
> ImageNet-21k pretrained backbone only.  Results will not reflect deepfake
> training; a warning is logged on startup.

### 2.4 Training your own model (optional)

Key hyperparameters from the paper:

| Parameter | Value |
|-----------|-------|
| Backbone | `vit_base_patch16_224` (ImageNet-21k via TIMM) |
| Phase 1 LR | 1e-3 (head only, 10 epochs) |
| Phase 2 LR | 3e-5 (full model, 30 epochs, cosine annealing) |
| Optimizer | AdamW, weight_decay=0.01 |
| Loss | BCELoss with class weights (real:1.0, fake:~1.4) |
| Augmentation | HorizontalFlip, ±10° rotation, JPEG Q60–80 |
| Input | 224×224 MTCNN-cropped face |

Minimal training snippet:

```python
import timm, torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from backend.model import DeepfakeViT

model = DeepfakeViT(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Phase 1: head only
for p in model.backbone.parameters():
    p.requires_grad = False

optim = AdamW(model.head.parameters(), lr=1e-3, weight_decay=0.01)
pos_weight = torch.tensor([1.4]).to(device)   # class imbalance correction
criterion = nn.BCELoss(weight=pos_weight)

# ... training loop ...

# Phase 2: full fine-tune
for p in model.backbone.parameters():
    p.requires_grad = True

optim = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scheduler = CosineAnnealingLR(optim, T_max=30)

# ... fine-tuning loop ...

torch.save(model.state_dict(), "backend/models/best_model.pth")
```

### 2.5 Start the backend server

```bash
# From the backend/ directory (venv must be active)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:
```
INFO  | Uvicorn running on http://0.0.0.0:8000
INFO  | FacePreprocessor initialised on device=cpu, target_size=224
INFO  | Loading fine-tuned weights from: .../models/best_model.pth
INFO  | Model ready on device=cpu
INFO  | === API ready ===
```

Interactive API docs available at → **http://localhost:8000/docs**

---

## 3 — Frontend Setup

No build step required.  The frontend is a single static HTML file.

```bash
# Simply open it in your browser:
open frontend/index.html        # macOS
xdg-open frontend/index.html    # Linux
start frontend/index.html       # Windows
```

Or serve it with any static file server to avoid CORS issues with local
`file://` URLs on some browsers:

```bash
# Python built-in server (from the frontend/ directory)
cd ../frontend
python -m http.server 3000
# Then visit http://localhost:3000
```

---

## 4 — Using the Application

### Option A: Using the Live Deployed App (Default)
1. Open the live frontend at **https://vipin-gupta-07.github.io/deepfake-detector/**.
2. Drag-and-drop or click to upload a facial image (JPEG/PNG/WebP/BMP, ≤ 15 MB).
3. The frontend automatically sends the image to the live API backend at `https://vipingupta04-deepfake-detector.hf.space/predict`.
4. Results appear instantly with the classification metrics.

### Option B: Running the Entire Stack Locally
1. Start the local backend server (as described in **2.5**).
2. Open [frontend/index.html](file:///c:/Users/vipin/Downloads/deepfake-detector/frontend/index.html) in your editor and change the `API_BASE` variable back to `"http://localhost:8000"` (it is currently pointed to the Hugging Face Space).
3. Open **http://localhost:3000** (or double-click the `index.html` file to open it directly).
4. Drag-and-drop or click to upload an image to analyze it on your local server.

### Results Analysis
For both options, results appear with:
- **REAL / FAKE** verdict
- Confidence percentage
- Real vs. Fake probability bars
- Face detection status (MTCNN found face vs. fallback)
- Server latency in milliseconds

---

## 5 — API Reference

### `GET /`
Health check.

**Response:**
```json
{ "status": "ok", "model": "ViT-B/16 deepfake detector", "device": "cpu" }
```

### `POST /predict`
Classify an uploaded image.

**Request:** `multipart/form-data` with field `file` (image/jpeg, png, webp, bmp)

**Response:**
```json
{
  "label":       "Fake",
  "confidence":  94.3,
  "real_prob":   5.7,
  "fake_prob":   94.3,
  "face_found":  true,
  "latency_ms":  312.4
}
```

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | `"Real"` or `"Fake"` |
| `confidence` | float | Probability of the predicted class (%) |
| `real_prob` | float | Probability of Real (%) |
| `fake_prob` | float | Probability of Fake (%) |
| `face_found` | bool | Whether MTCNN detected a face |
| `latency_ms` | float | End-to-end processing time |

**Error responses:**
| Status | Cause |
|--------|-------|
| 400 | Empty file uploaded |
| 413 | File exceeds 15 MB |
| 415 | Unsupported image type |
| 422 | Image could not be decoded |
| 500 | Internal model error |

---

## 6 — Production Deployment Notes

- Replace `allow_origins=["*"]` in `main.py` with your actual frontend domain.
- Use Gunicorn with Uvicorn workers for multi-process serving:
  ```bash
  gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
  ```
- Containerise with Docker and mount `models/best_model.pth` as a volume.
- For high-throughput scenarios, consider batching requests or using TorchServe.

---


# ⛑️ AI HardHat Detection

A **YOLOv8-based computer vision model** with a **Streamlit web app** that checks whether workers on a construction site are wearing safety helmets.

---

## Features
- Detects three classes: **helmet**, **head** (no helmet) and **person**
- Streamlit UI with drag-and-drop **batch upload** and one-click sample images
- Per-image compliance verdict (compliant / non-compliant) plus a summary dashboard
- Adjustable confidence, IoU and inference-size thresholds
- CPU or GPU inference, with automatic fallback when the installed PyTorch build doesn't support the GPU

## Tech Stack
- Python 3.11
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) (`yolov8n`)
- PyTorch, OpenCV
- Streamlit
- Trained on an NVIDIA RTX 5060
- Dataset: *Hard Hat Workers* (Roboflow, public domain) — 1,413 training / 416 validation images

---

## Getting Started

### 1. Clone and install
```bash
git clone https://github.com/BaguSubash1/AI_HardHat_Detection.git
cd AI_HardHat_Detection
pip install -r requirements.txt
```

### 2. Add the model weights
Weights (`*.pt`) are excluded from git. Place your trained model at:

```
Hard Hat Workers.v1i.yolov8/runs/detect/train2/weights/best.pt
```

or paste a different path into **Settings → Advanced → Model weights** in the app.

### 3. Run the app
```bash
streamlit run app.py
```
If the `streamlit` command isn't found, use `python -m streamlit run app.py`.

Then open http://localhost:8501, upload one or more site photos (JPG, PNG, BMP, WEBP) and review the results.

---

## Using the App

| Setting | What it does |
|---|---|
| Confidence threshold | Minimum score for a detection to be shown. Higher means fewer, more certain boxes. |
| IoU (NMS) threshold | Overlap above which duplicate boxes are merged. |
| Inference size | Image size fed to the model (320–640). Larger can be more accurate but slower. |
| Device | CPU or GPU. GPU is only offered by default if your PyTorch build supports your card. |

**Verdicts**
- ✅ **Compliant**: at least one helmet and no uncovered heads
- ❌ **Non-compliant**: one or more uncovered heads
- ⚠️ **No detection**: no heads or helmets found above the threshold

The summary row reports totals for images, helmets, uncovered heads and people, plus a **compliance %** (helmets ÷ (helmets + uncovered heads)).

---

## 🧠 Model Training
Run from inside `Hard Hat Workers.v1i.yolov8/`:

```bash
# Train
yolo detect train data=data.yaml model=yolov8n.pt epochs=10 imgsz=416

# Validate
yolo detect val model=runs/detect/train2/weights/best.pt
```

### Results (10 epochs, `train2`)
| Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|
| 0.956 | 0.605 | 0.645 | 0.438 |

Precision is high but recall is modest, so the model rarely raises false alarms but can miss some workers. Training longer or using a larger model (`yolov8s`/`yolov8m`) should help. Curves and the confusion matrix are in `runs/detect/train2/`.

---

## Project Structure
```
AI_HardHat_Detection/
├── app.py                          # Streamlit app
├── requirements.txt
├── README.md
└── Hard Hat Workers.v1i.yolov8/    # Roboflow dataset + training outputs
    ├── data.yaml                   # classes: head, helmet, person
    ├── train/ valid/ test/
    └── runs/detect/train2/         # metrics, plots, weights/best.pt
```

## Troubleshooting
- **`requirements.txt` not found**: `cd` into `AI_HardHat_Detection` first.
- **GPU detected but unused**: some newer GPUs (e.g. RTX 50-series, `sm_120`) need a PyTorch build with matching kernels; the app falls back to CPU. Install a newer PyTorch nightly/CUDA build to use the GPU.
- **numpy / opencv version warning**: harmless; `ultralytics==8.3.0` pins numpy 1.x.

## License
Dataset: public domain (Roboflow). Add a license for the code if you plan to share it.

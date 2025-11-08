# AI HardHat Detection

A **YOLOv8-based computer vision model** that detects whether workers are wearing safety helmets on construction sites.

---

## Project Overview
This project uses the **Ultralytics YOLOv8** framework to detect:
-  Workers wearing hard hats  
-  Workers **without** hard hats  

Built and trained using:
- Python 3.11  
- Ultralytics YOLOv8  
- NVIDIA RTX 5060 GPU  
- Custom labeled dataset (1,413 training, 416 validation images)

---

## 🧠 Model Training
```bash
# Train model
yolo detect train data=data.yaml model=yolov8n.pt epochs=10 imgsz=416

# Validate model
yolo detect val model=runs/detect/train2/weights/best.pt

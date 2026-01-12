# YOLOv8 Violence Detection – Inference Module (Batch)

This module implements **violence / fight detection** using a **YOLOv8 model** as part of the Master's Thesis  
**“Security System with Real-Time Inference”**.

It is designed to be integrated with the **security system controller**, receiving **single frames or batches of images** and detecting the presence of **violent actions**.

The module exposes a **FastAPI-based inference service**, optimized for real-time and batch processing.

---

## 1. Module Overview

This inference module provides:

- Violence / fight detection using YOLOv8 (Ultralytics)
- Single-image and batch inference
- REST API interface for real-time integration
- Lightweight YOLOv8-nano model for fast execution
- Confidence-based filtering of detections

This module **does not perform training**, only inference.

---

## 2. Directory Structure

```text
.
├── violenceDetection.py     # Violence detection inference API (YOLOv8)
├── Yolo_nano_weights.pt     # Trained YOLOv8 weights (required)
└── requirements.txt         # Python dependencies
```

⚠️ The file `Yolo_nano_weights.pt` must be located in the same directory as `violenceDetection.py`.

---

## 3. Environment Setup

⚠️ Python version: **3.8**

```bash
pip install -r requirements.txt
```

---

## 4. Running the Inference API

```bash
python violenceDetection.py --port 8009
```

---

## 5. Available API Endpoints

- `GET /` – Health check  
- `POST /detect` – Single image violence detection  
- `POST /detect_batch` – Batch violence detection  

---

## 6. Model and Dataset

- Model: YOLOv8-nano
- Detection class: Violence / Fight (class id = 1)

Dataset reference:
```text
https://huggingface.co/Musawer14/fight_detection_yolov8
```

---

## 7. Detection Logic

- Only detections with `class_id == 1` and `confidence > 0.5` are returned
- Non-violence detections are ignored

---

## 8. Building Executable (Windows)

```bash
pyinstaller --onefile --hidden-import ultralytics --hidden-import cv2 --add-data "C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EnvironmentCategory\violance_detection\Fight-Violence-detection-yolov8\Yolo_nano_weights.pt;." violenceDetection.py
```

---

## 9. Notes

- Inference only
- Python 3.8 recommended
- Model weights must be bundled with the executable

# DeepFace Emotion Recognition – Inference Module (Batch)

This module implements **facial emotion recognition** using the **DeepFace** framework as part of the Master's Thesis  
**“Security System with Real-Time Inference”**.

It is designed to be integrated with the **security system controller**, receiving **single frames or batches of images** and extracting the **dominant facial emotion**.

The module exposes a **FastAPI-based inference service** and relies on **TensorFlow** as the backend.

---

## 1. Module Overview

This inference module provides:

- Facial emotion recognition using DeepFace
- Single-image and batch inference
- REST API interface for real-time integration
- Automatic download and caching of DeepFace model weights
- TensorFlow-based execution (CPU)

This module **does not perform training**, only inference.

---

## 2. Directory Structure

```text
.
├── emotion_deepface.py   # Emotion inference API (DeepFace)
└── requirements.txt      # Python dependencies
```

---

## 3. Environment Setup

⚠️ **Python version**  
This module was developed and packaged using **Python 3.8**.  
Do not forget to create the virtual environment using Python 3.8.

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Running the Inference API

```bash
python emotion_deepface.py --port 8013
```

---

## 6. Building Executable (Windows)

```bash
venv\Scripts\python.exe -m PyInstaller --clean --onefile --name emotion_deepface ^
 --collect-all tensorflow ^
 --collect-all deepface ^
 --add-data "venv\Lib\site-packages\tensorflow;tensorflow" ^
 --add-data "venv\Lib\site-packages\cv2\data;cv2\data" ^
 --add-binary "C:\Windows\System32\msvcp140_1.dll;." ^
 --add-binary "C:\Windows\System32\vcruntime140.dll;." ^
 emotion_deepface.py
```

---

## 7. Notes

- CPU-based inference
- First run downloads DeepFace models
- Python 3.8 required

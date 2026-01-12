# FER Emotion Recognition – Inference Module (Batch)

This module implements **facial emotion recognition** using the **FER (Facial Expression Recognition)** library as part of the Master's Thesis  
**“Security System with Real-Time Inference”**.

It is designed to be integrated with the **security system controller**, receiving **single frames or batches of images** and extracting the **dominant facial emotion** with an associated confidence score.

The module exposes a **FastAPI-based inference service**, optimized for batch processing with parallel execution.

---

## 1. Module Overview

This inference module provides:

- Facial emotion recognition using FER (with MTCNN face detection)
- Single-image and batch inference endpoints
- Parallel batch processing using thread pools
- REST API interface for real-time integration
- Confidence scores per detected emotion

This module **does not perform training**, only inference.

---

## 2. Directory Structure

```text
.
├── emotion_fer.py      # Emotion inference API (FER)
└── requirements.txt    # Python dependencies
```

---

## 3. Environment Setup

⚠️ **Python version**  
This module was built and packaged using **Python 3.8**.  
Do not forget to create the virtual environment with Python 3.8.

### Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment (Windows):

```bash
venv\Scripts\activate
```

### Install dependencies

All required dependencies are defined in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

## 4. Running the Inference API

### Option A – Pass port as argument

```bash
python emotion_fer.py --port 8012
```

### Option B – Interactive mode

```bash
python emotion_fer.py
```

You will be prompted to enter the port manually.

The API will be available at:

```
http://0.0.0.0:<PORT>
```

---

## 5. Available API Endpoints

### GET /
Health check endpoint.

---

### POST /detect-emotions
Single image emotion recognition.

---

### POST /detect_batch
Batch emotion recognition (recommended for controller integration).

---

## 6. Batch Response Format

```json
{
  "status": "success",
  "emotions": [
    { "emotion": "neutral", "confidence": 0.81 },
    { "emotion": "angry", "confidence": 0.76 }
  ],
  "faces_areas": [18342, 16510]
}
```

---

## 7. Example Batch Request (curl)

```bash
curl -X POST "http://localhost:8012/detect_batch" \
  -F "files=@frame1.jpg" \
  -F "files=@frame2.jpg" \
  -F "files=@frame3.jpg"
```

---

## 8. Emotion Detection Logic

- FER detects all faces in each frame
- Dominant emotion is selected from the first detected face
- Face areas are used to select the most relevant frame

---

## 9. Building Executable (Windows)

Executables are generated using **PyInstaller**.

### Command used (older)

```bash
pyinstaller --onefile --hidden-import=fer --hidden-import=tensorflow --hidden-import=torch --hidden-import=torchvision --hidden-import=fastapi --hidden-import=uvicorn --add-data "C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EmotionCategory\FER\venv\Lib\site-packages\facenet_pytorch\data;facenet_pytorch/data" --add-data "C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EmotionCategory\FER\venv\Lib\site-packages\fer\data;fer/data" emotion_fer.py
```

### Command used (most recent)

```bash
"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EmotionCategory\FER\venv\Scripts\python.exe" -m PyInstaller --clean --onefile --name emotion_fer --collect-all fer --hidden-import=tensorflow --hidden-import=torch --hidden-import=torchvision --hidden-import=fastapi --hidden-import=uvicorn --add-data "C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EmotionCategory\FER\venv\Lib\site-packages\facenet_pytorch\data;facenet_pytorch/data" --add-data "C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EmotionCategory\FER\venv\Lib\site-packages\fer\data;fer/data" --add-binary "C:\Windows\System32\msvcp140_1.dll;." --add-binary "C:\Windows\System32\vcruntime140.dll;." emotion_fer.py
```

### Notes

- The FER / facenet_pytorch data folders are bundled via `--add-data`
- Required Windows runtime DLLs are included via `--add-binary`
- The generated executable will be placed in the `dist/` directory
- Build artifacts (`build/`, `dist/`, `.spec`) should not be versioned
- This build is intended for **Windows environments**
- **Python 3.8** was used to create the venv used for packaging

---

## 10. Tests / Validation Notes

- Only tested emotions: **angry** and **fear**
- Test datasets used:

```text
https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset
https://www.kaggle.com/datasets/msambare/fer2013
```

---

## 11. Notes for System Integration

- Designed for real-time security systems
- Batch inference reduces overhead
- Emotion thresholds handled by controller
- CPU-based inference (TensorFlow backend)

---

## 12. Reproducibility Notes

- Inference only
- FER models downloaded automatically
- Only one instance per port

# EfficientNet Weapon Classification – Inference Module (Batch)

This module implements **weapon classification** using a **PyTorch EfficientNet-like model** as part of the Master's Thesis  
**“Security System with Real-Time Inference”**.

It is designed to be integrated with the **security system controller**, receiving **batches of images** and classifying each one as:

- `weapon`
- `normal`

The module exposes a **FastAPI-based inference service**, optimized for batch image processing.

---

## 1. Module Overview

This inference module provides:

- Image classification (weapon vs normal)
- Batch inference endpoint for security controllers
- GPU support (CUDA, if available)
- Softmax confidence scores per image
- REST API interface for real-time integration

This module **does not perform training**, only inference.

---

## 2. Directory Structure

```text
.
├── efficient_net.py     # Inference API
├── b0_global.pt         # Trained PyTorch model (required)
└── requirements.txt     # Python dependencies
```

⚠️ **Important**  
The model file `b0_global.pt` must be located in the **same directory** as `efficient_net.py`.

---

## 3. Environment Setup

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
python efficient_net.py --port 8010
```

### Option B – Interactive mode

```bash
python efficient_net.py
```

The API will be available at:

```
http://0.0.0.0:<PORT>
```

---

## 5. Available API Endpoints

### GET /
Health check endpoint.

---

### POST /detect
Single image classification.

---

### POST /detect_batch
Batch image classification (recommended for controller integration).

---

## 6. Class Mapping

```python
CLASS_NAMES = ["weapon", "normal"]
```

| Class ID | Label   |
|---------:|---------|
| 0        | weapon  |
| 1        | normal  |

---

## 7. Batch Response Example

```json
{
  "status": "success",
  "detections": [
    [
      {
        "label": "weapon",
        "confidence": 0.99
      }
    ],
    [
      {
        "label": "normal",
        "confidence": 0.87
      }
    ]
  ]
}
```

---

## 8. Example Batch Request (curl)

```bash
curl -X POST "http://localhost:8010/detect_batch" \
  -F "files=@frame1.jpg" \
  -F "files=@frame2.jpg" \
  -F "files=@frame3.jpg"
```

---

## 9. Building Executable (Windows)

Executables are generated using **PyInstaller**.

Run the following command **exactly as shown below**:

```bash
python -m PyInstaller --onefile --name EfficientNetAPI --collect-all torch --collect-all torchvision --collect-submodules PIL --hidden-import uvicorn --hidden-import fastapi --add-data "C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\EfficientNet\EfficientNet-for-Gun-detection\b0_global.pt;." "C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\EfficientNet\EfficientNet-for-Gun-detection\efficient_net.py"
```

### Notes

- The `b0_global.pt` file is bundled with the executable using `--add-data`
- The generated executable will be placed in the `dist/` directory
- Build artifacts (`build/`, `dist/`, `.spec`) should not be versioned
- This build is intended for **Windows environments**

---

## 10. Notes for System Integration

- Designed for real-time security systems
- Batch inference reduces API overhead
- Confidence thresholds should be applied by the controller
- CUDA is automatically used if available

---

## 11. Reproducibility Notes

- Inference only (no training)
- Model weights must be provided externally
- Only one model instance should run per port

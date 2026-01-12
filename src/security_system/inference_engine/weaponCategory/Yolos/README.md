
# YOLOv8 Weapon Detection – Inference Module

This module implements weapon detection using YOLOv8 models as part of the Master's Thesis
**"Security System with Real-Time Inference"**.

It includes:
- Model inference
- A dedicated service for each model, exposing an API to communicate with the security system controller
- API endpoints for image-based detection
- Benchmarking utilities


## 1. Environment Setup

### Create a virtual environment

python -m venv venv

Activate the virtual environment (Windows)
venv\Script\activate

Install dependencies
python -r requirements.txt

## 2. Running the API and Testing Models

The API exposes an endpoint for weapon detection via image upload.

Example request (YOLOv8 – best model)
    Para o modelo best:
    curl -X POST "http://127.0.0.1:8000/detect" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/absolute/path/to/image.jpg"
    Notes:
        Ensure the API is running on the correct port for the selected model.

        Replace /absolute/path/to/image.jpg with the full path to the test image.

        Only one model should be running at a time.


## 3. Building Executables (Windows)

Executables are generated using PyInstaller

    python -m PyInstaller --onefile --hidden-import ultralytics --hidden-import cv2 --add-data "C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\Yolos\YoloV8\models\best2.pt;models" weaponDetectionBest2.py

    python -m PyInstaller --onefile --hidden-import ultralytics --hidden-import cv2 --add-data "C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\Yolos\YoloV8\models\best.pt;models" weaponDetectionBest.py

    python -m PyInstaller --onefile --hidden-import ultralytics --hidden-import cv2 --add-data "C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\Yolos\YoloV8\models\model.pt;models" weaponDetectionModel.py

Note:
    The models/ directory must contain the corresponding .pt weights.

    Generated executables are placed in the dist/ directory (not versioned).



## 4. YOLOv8 Models
    Model Repositories
        Model Repositories:
            modelo best.pt  -- https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8/blob/main/runs/detect/Normal/weights/last.pt
            modelo best2.pt -- https://github.com/BecayeSoft/Guns-Detection-YOLOv8/blob/main/runs/detect/train18/weights/best.pt
            modelo model.pt -- https://github.com/GingerBrains/object-detection/tree/main
## 5. Training Datasets
        The models were trained using the following datasets:
            modelo best.pt -- https://universe.roboflow.com/joao-assalim-xmovq/weapon-2/dataset/2
            modelo best2.pt -- https://universe.roboflow.com/yolo-xkggu/guns-mms73/dataset/3
            modelo model.pt -- https://universe.roboflow.com/fend-tech/weapon-detection-dinou

            
## 6. Notes for Reproducibility

    This repository focuses on inference and evaluation, not model training.

    Model weights are obtained from external sources and must be downloaded manually.

    Build artifacts (build/, dist/, *.exe) are intentionally excluded from version control.
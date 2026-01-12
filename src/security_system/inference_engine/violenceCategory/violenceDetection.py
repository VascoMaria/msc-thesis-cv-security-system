from fastapi import FastAPI, UploadFile, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO
import uvicorn
import os
import io
import argparse
import sys


def get_model_path():
    # Se for um executável PyInstaller, pega do _MEIPASS
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(__file__)
    return os.path.join(base, "Yolo_nano_weights.pt")

# Carrega o modelo usando a função acima
model = YOLO(get_model_path())


# Inicializa a aplicação FastAPI
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "Modelo de detecção de violência ativo!"}

@app.post("/detect")
async def detect_violence(file: UploadFile):
    try:
        # Lê o conteúdo do ficheiro e converte para imagem OpenCV
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Aplica o modelo YOLO ao frame
        results = model(frame)

        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            if class_id == 1 and confidence > 0.5:  # Detecta apenas "violência"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": "violence",
                    "confidence": confidence,
                    #"bounding_box": {
                    #    "x1": x1,
                    #    "y1": y1,
                    #    "x2": x2,
                    #    "y2": y2
                    #}
                })

        return {"status": "success", "detections": detections}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



from typing import List
from fastapi import File, UploadFile
import numpy as np
import cv2
import time

# Função auxiliar para batch
async def detect_violence_batch_files(files: List[UploadFile]):
    """
    Decodifica uploads em frames, executa batch e retorna detecções por frame.
    """
    # Decodificar frames
    frames = []
    for file in files:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Imagem inválida ou corrompida.")
        frames.append(frame)

    # Medir tempo
    start = time.time()
    # inference em batch (stream=False garante batch)
    results_batch = model(frames, stream=False)
    elapsed = round(time.time() - start, 3)

    # Processar resultados
    all_detections = []
    for result in results_batch:
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            # somente classe "violence" (id=1)
            if cls_id == 1 and conf > 0.5:
                detections.append({
                    "label": "violence",
                    "confidence": conf
                })
        all_detections.append(detections)

    return elapsed, all_detections

@app.post("/detect_batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    Endpoint batch para detecção de violência. Recebe uma lista de imagens
    e retorna detecções em batch, além do tempo total de inferência.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Nenhuma imagem enviada.")
    elapsed, detections = await detect_violence_batch_files(files)
    return {
        "status": "success",
        "processing_time": elapsed,
        "detections": detections
    }




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inicia o Modelo de Violence-detection environment")
    parser.add_argument('--port', type=int, help='Porta para o Modelo de Violence-detection environment (ex: 8009)', default=None)
    args = parser.parse_args()

    if args.port:
        port = args.port
    else:
        port = int(input("Digite a porta para iniciar o Modelo de Violence-detection environment (ex: 8009): "))

    print("Iniciando o Modelo de Violence-detection environment...")
    uvicorn.run(app, host="0.0.0.0", port=port)

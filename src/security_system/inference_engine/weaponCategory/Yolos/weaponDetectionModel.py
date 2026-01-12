from fastapi import FastAPI, UploadFile, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO
import uvicorn
import sys
import os
import argparse

# Função para localizar arquivos no executável ou no script
def get_model_path():
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller extrai os arquivos para a pasta _MEIPASS
        return os.path.join(sys._MEIPASS, "models", "model.pt")
    else:
        # Caminho normal para o script
        return os.path.join(os.path.dirname(__file__),  "models", "model.pt")

model_path = get_model_path()
weapon_model = YOLO(model_path)


def detect_weapons(frame):
    """
    Detecta armas em uma imagem e retorna o rótulo, confiança e bounding boxes.
    """
    results = []
    weapon_results = weapon_model(frame)
    for result in weapon_results:
        for box in result.boxes:
            confidence = box.conf.item()
            label = weapon_model.names[int(box.cls)]
            if confidence > 0.5:
                results.append({
                    "label": label,
                    "confidence": confidence
                })
    return results

app = FastAPI()

@app.get("/")
def health_check():
    """Verifica se o serviço está ativo."""
    return {"status": "Modelo está ativo!"}

@app.post("/detect")
async def detect(file: UploadFile):
    """
    Recebe um frame e retorna a detecção do modelo.
    """
    try:
        # Lê o conteúdo do arquivo recebido
        file_content = await file.read()
        nparr = np.frombuffer(file_content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Processa o frame com o modelo
        results = detect_weapons(frame)

        # Retorna os resultados
        return {"status": "success", "detections": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from typing import List
from fastapi import File
import time

def detect_weapons_batch(frames):
    """
    Detecta armas em múltiplos frames usando o modelo Model.pt em batch.
    Também imprime o tempo total de inferência.
    """
    start_time = time.time()

    all_results = []
    # stream=False garante inferência em batch
    weapon_results = weapon_model(frames, stream=False)
    for result in weapon_results:
        frame_results = []
        for box in result.boxes:
            confidence = box.conf.item()
            label = weapon_model.names[int(box.cls)]
            if confidence > 0.5:
                frame_results.append({
                    "label": label,
                    "confidence": confidence
                })
        all_results.append(frame_results)

    total_time = time.time() - start_time
    print(f"\n🔍 Tempo total de inferência para {len(frames)} frames (Model): {total_time:.3f} segundos\n")

    return all_results

@app.post("/detect_batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    Recebe múltiplos frames e retorna as detecções em batch pelo modelo Model.pt.
    """
    try:
        frames = []
        for file in files:
            content = await file.read()
            nparr = np.frombuffer(content, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise HTTPException(status_code=400, detail="Imagem inválida ou corrompida.")
            frames.append(frame)

        results = detect_weapons_batch(frames)

        return {"status": "success", "detections": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inicia o Modelo de armas model.pt")
    parser.add_argument('--port', type=int, help='Porta para o Modelo de armas model.pt (ex: 8006)', default=None)
    args = parser.parse_args()

    if args.port:
        port = args.port
    else:
        port = int(input("Digite a porta para iniciar o Modelo de armas model.pt (ex: 8006): "))

    print("Iniciando o Modelo de armas model.pt ...")
    uvicorn.run(app, host="0.0.0.0", port=port)



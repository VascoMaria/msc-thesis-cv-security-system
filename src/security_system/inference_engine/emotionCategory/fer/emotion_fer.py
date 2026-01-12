from fastapi import FastAPI, UploadFile, HTTPException,File
import cv2
import numpy as np
from fer import FER
from fastapi.encoders import jsonable_encoder
import uvicorn
import argparse
from typing import List, Dict, Any, Optional
import os
import time
from fastapi import APIRouter


# Inicializa o aplicativo FastAPI
app = FastAPI()

# Inicializa o detector de emoções
detector = FER(mtcnn=True)  # Usa MTCNN para detecção mais precisa (desative para mais velocidade)

@app.get("/")
def health_check():
    """Verifica se o serviço está ativo."""
    return {"status": "Serviço de reconhecimento de emoções ativo!"}


def detect_emotions(frame):
    """
    Detecta emoções em um frame utilizando FER.
    """
    try:
        # Analisar emoções com FER
        results = detector.detect_emotions(frame)

        if not results:
            return {"emotion": "No Face", "confidence": 0.0}

        # Obtém a emoção dominante e a confiança
        result = results[0]
        emotion, confidence = max(result["emotions"].items(), key=lambda x: x[1])

        return {"emotion": emotion, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}


@app.post("/detect-emotions")
async def detect_emotions_api(file: UploadFile):
    """
    Recebe um frame e retorna as emoções detectadas.
    """
    try:
        # Ler o conteúdo do arquivo enviado
        file_content = await file.read()
        nparr = np.frombuffer(file_content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Processar o frame com a função detect_emotions
        results = detect_emotions(frame)

        # Serializar os resultados para evitar problemas com FastAPI
        results_serialized = jsonable_encoder(results)

        # Retornar os resultados
        return {"status": "success", "emotions": results_serialized}

    except Exception as e:
        # Levantar exceção com mensagem de erro em caso de falha
        raise HTTPException(status_code=500, detail=str(e))
    
from concurrent.futures import ThreadPoolExecutor
import asyncio


# =========================
# Helpers
# =========================
def _areas_from_fer_results(results: List[Dict[str, Any]]) -> List[int]:
    """
    results: lista de dicts do FER: {"box":[x,y,w,h], "emotions":{...}}
    """
    areas: List[int] = []
    for r in results or []:
        box = r.get("box") or [0, 0, 0, 0]
        if len(box) >= 4:
            w, h = int(box[2] or 0), int(box[3] or 0)
            a = max(w, 0) * max(h, 0)
            if a > 0:
                areas.append(int(a))
    areas.sort(reverse=True)
    return areas

def _dominant_from_first_face(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"emotion": "No Face", "confidence": 0.0}
    r0 = results[0]
    emo_map = r0.get("emotions") or {}
    if not emo_map:
        return {"emotion": "No Face", "confidence": 0.0}
    emotion, conf = max(emo_map.items(), key=lambda x: x[1])
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    return {"emotion": emotion, "confidence": conf}


# Função síncrona que detecta emoções em uma imagem
def detect_emotion_fer_sync(frame):
    """
    Processa UMA imagem com FER e devolve:
      - emotions (compat)  -> dict {emotion, confidence}
      - areas_sorted       -> list[int] áreas de todas as caras, desc
    """
    try:
        results = detector.detect_emotions(frame)  # lista de faces
        emotions_payload = _dominant_from_first_face(results)
        areas_sorted = _areas_from_fer_results(results)
        return emotions_payload, areas_sorted
    except Exception as e:
        return {"error": str(e)}, []

@app.post("/detect_batch")
async def detect_emotions_batch_parallel(files: List[UploadFile] = File(...)):
    """
    Recebe várias imagens:
      - devolve emotions (lista compat por frame)
      - devolve faces_areas [best_user_area, best_intruder_area] do 'melhor' frame
    """
    loop = asyncio.get_event_loop()
    max_workers = int(os.getenv("BATCH_MAX_WORKERS", "4"))
    executor = ThreadPoolExecutor(max_workers=max_workers)

    # Ler frames
    frames = []
    for upload in files:
        content = await upload.read()
        arr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Uma das imagens está corrompida ou inválida.")
        frames.append(frame)

    start_time = time.time()
    tasks = [loop.run_in_executor(executor, detect_emotion_fer_sync, f) for f in frames]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    print(f"\n🕒 Tempo de inferência FER para {len(frames)} imagens: {total_time:.3f}s\n")

    # Emotions (compat) + pares (user_area, intruder_area) por frame
    emotions_only = []
    pairs = []
    for emo, areas in results:
        emotions_only.append(emo)
        u = areas[0] if len(areas) > 0 else 0
        v = areas[1] if len(areas) > 1 else 0
        pairs.append((u, v))

    # Escolher o "melhor" frame:
    # 1) preferir frames com 2 caras (v>0)
    # 2) entre esses, menor diferença (u - v)
    # 3) empate -> maior soma (u + v)
    # 4) se nenhum tiver 2 caras, escolher o de maior u
    best_idx = None
    best_diff = None
    best_sum = None
    for i, (u, v) in enumerate(pairs):
        if v <= 0:
            continue
        diff = u - v
        s = u + v
        if (best_idx is None) or (diff < best_diff) or (diff == best_diff and s > best_sum):
            best_idx, best_diff, best_sum = i, diff, s
    if best_idx is None:
        best_idx = max(range(len(pairs)), key=lambda i: pairs[i][0]) if pairs else 0

    best_user_area, best_intruder_area = pairs[best_idx] if pairs else (0, 0)

    return {
        "status": "success",
        "emotions": emotions_only,
        "faces_areas": [best_user_area, best_intruder_area]  # par do frame “melhor”
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inicia o Modelo de emoções com FER")
    parser.add_argument('--port', type=int, help='Porta para o Modelo de emoções com FER (ex: 8012)', default=None)
    args = parser.parse_args()

    if args.port:
        port = args.port
    else:
        port = int(input("Digite a porta para iniciar o Modelo de emoções com FER (ex: 8012): "))

    print("Iniciando o Modelo de emoções com FER...")
    uvicorn.run(app, host="0.0.0.0", port=port)

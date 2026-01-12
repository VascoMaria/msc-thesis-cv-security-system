from fastapi import FastAPI, UploadFile, HTTPException
import cv2
import numpy as np
from deepface import DeepFace
from fastapi.encoders import jsonable_encoder
import uvicorn
import os
import argparse
from typing import List
from fastapi import File

# Função para determinar o caminho do arquivo Haar Cascade
def get_haarcascade_path(filename):
    # Caminho específico para os arquivos Haar Cascade no OpenCV
    haarcascade_base_path = cv2.data.haarcascades
    haarcascade_path = os.path.join(haarcascade_base_path, filename)
    # Verificar se o arquivo existe
    if not os.path.exists(haarcascade_path):
        raise FileNotFoundError(f"Arquivo Haar Cascade não encontrado: {haarcascade_path}")
    return haarcascade_path

# Caminhos para os arquivos Haar Cascade
HAARCASCADE_FACE = get_haarcascade_path("haarcascade_frontalface_default.xml")
HAARCASCADE_EYE = get_haarcascade_path("haarcascade_eye.xml")

# Inicializa o aplicativo FastAPI
app = FastAPI()

@app.get("/")
def health_check():
    """Verifica se o serviço está ativo."""
    return {"status": "Serviço de emoções está ativo!"}


def detect_emotions(frame):
    """
    Detecta emoções em um frame utilizando o DeepFace.
    """
    try:
        # Analisar emoções com DeepFace
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Obter a emoção dominante e a confiança
        emotion = analysis[0]['dominant_emotion']
        confidence = float(analysis[0]['emotion'][emotion])  

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
    
import os
import cv2
import numpy as np
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from deepface import DeepFace
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

app = FastAPI()

# ---------------- helpers ----------------
def _to_list(analysis):
    if isinstance(analysis, list):
        return analysis
    return [analysis] if isinstance(analysis, dict) else []

def _areas_sorted(results):
    areas = []
    for r in results:
        region = (r.get("region") or {})
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))
        a = max(w, 0) * max(h, 0)
        if a > 0:
            areas.append(int(a))
    areas.sort(reverse=True)  # maior -> menor
    return areas

# ------------ worker síncrono (1 chamada ao DeepFace) ------------
def detect_emotion_sync(frame):
    try:
        analysis = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend=os.getenv("DETECTOR_BACKEND", "opencv")
        )
        results = _to_list(analysis)

        # Emotions (compat): usa o primeiro item, tal como o teu original
        if results:
            first = results[0]
            emo = first.get('dominant_emotion')
            conf = float((first.get('emotion') or {}).get(emo, 0.0))
            emotions_payload = {"emotion": emo, "confidence": conf}
        else:
            emotions_payload = {"emotion": None, "confidence": 0.0}

        # Áreas de todas as caras (ordenadas desc)
        areas = _areas_sorted(results)

        # devolve em formato simples para o batch
        return emotions_payload, areas

    except Exception as e:
        # não rebenta o batch
        return {"error": str(e)}, []

@app.post("/detect_batch")
async def detect_emotions_batch(files: List[UploadFile] = File(...)):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=int(os.getenv("BATCH_MAX_WORKERS", "4")))

    frames = []
    for file in files:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Uma das imagens está corrompida ou inválida.")
        frames.append(frame)

    start_time = time.time()
    tasks = [loop.run_in_executor(executor, detect_emotion_sync, f) for f in frames]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    print(f"\n🕒 Tempo total de inferência para {len(frames)} imagens: {total_time:.3f} s\n")

    # separar emotions + pares (user_area, intruder_area) por frame
    emotions_only = []
    pairs = []  # [(user_area, intruder_area), ...] alinhado com inputs
    for emo, areas in results:
        emotions_only.append(emo)
        user_area = areas[0] if len(areas) > 0 else 0
        intruder_area = areas[1] if len(areas) > 1 else 0
        pairs.append((user_area, intruder_area))

    # escolher o "melhor" frame:
    # 1) preferir frames com 2 caras (intruder_area > 0)
    # 2) entre esses, escolher o de menor diferença (user - intruder)
    # 3) empate -> maior soma (user + intruder)
    # 4) se nenhum tiver 2 caras, escolher o de MAIOR user_area
    best_idx = None
    best_diff = None
    best_sum = None
    for i, (u, v) in enumerate(pairs):
        if v <= 0:
            continue  # só consideramos aqui os que têm 2 caras
        diff = u - v  # u>=v por construção
        s = u + v
        if (best_idx is None
            or diff < best_diff
            or (diff == best_diff and s > best_sum)):
            best_idx, best_diff, best_sum = i, diff, s

    if best_idx is None:
        # nenhum frame com 2 caras: escolhe o de maior user_area
        best_idx = max(range(len(pairs)), key=lambda i: pairs[i][0]) if pairs else 0

    best_user_area, best_intruder_area = pairs[best_idx] if pairs else (0, 0)

    return {
        "status": "success",
        "emotions": emotions_only,           # exatamente como dantes (lista por frame)
        "faces_areas": [best_user_area, best_intruder_area]  # NOVO: só do frame “melhor”
    }






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inicia o Modelo de emoções com DeepFace")
    parser.add_argument('--port', type=int, help='Porta para o Modelo de emoções com DeepFace (ex: 8001)', default=None)
    args = parser.parse_args()

    if args.port:
        port = args.port
    else:
        port = int(input("Digite a porta para iniciar o Modelo de emoções com DeepFace (ex: 8001): "))

    print("Iniciando o Modelo de emoções com DeepFace...")
    uvicorn.run(app, host="0.0.0.0", port=port)

    print(f"Caminho do Haarcascade (face): {HAARCASCADE_FACE}")
    print(f"Caminho do Haarcascade (eye): {HAARCASCADE_EYE}")

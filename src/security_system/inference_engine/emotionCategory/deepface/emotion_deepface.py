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
    
import asyncio
from fastapi import File, UploadFile
from concurrent.futures import ThreadPoolExecutor
import time
# Função síncrona que roda DeepFace para 1 imagem
def detect_emotion_sync(frame):
    try:
        analysis = DeepFace.analyze(
            frame, actions=['emotion'], enforce_detection=False
        )

        # DeepFace pode devolver lista ou dict
        if isinstance(analysis, dict):
            analysis = [analysis]

        # filtra só itens válidos
        valid = [a for a in analysis if "dominant_emotion" in a and "emotion" in a]

        if not valid:
            return {"emotion": "No Face", "confidence": 0.0}

        first = valid[0]
        emo = first["dominant_emotion"]
        conf = float((first.get("emotion") or {}).get(emo, 0.0))
        return {"emotion": emo, "confidence": conf}

    except Exception as e:
        return {"emotion": "No Face", "confidence": 0.0, "error": str(e)}


@app.post("/detect_batch")
async def detect_emotions_batch(files: List[UploadFile] = File(...)):
   
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor()

    frames = []
    for file in files:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"status": "error", "detail": "Uma das imagens está corrompida ou inválida."}
        frames.append(frame)
  
    start_time = time.time()

    tasks = [loop.run_in_executor(executor, detect_emotion_sync, f) for f in frames]
    results = await asyncio.gather(*tasks)

    
    total_time = time.time() - start_time
    print(f"\n🕒 Tempo total de inferência para 5 imagens: {total_time:.3f} segundos\n")
    print(results)
    return {
        "status": "success",
        "emotions": results
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

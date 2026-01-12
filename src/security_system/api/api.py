import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, HTTPException
import asyncio
from CONTROLADOR.controlador_batch import process_frame
from COMMON.common_batch import check_camera_status
import numpy as np
import cv2
import uvicorn
from COMMON.logging_config import logger
import argparse
from collections import deque
from datetime import datetime, timezone

app = FastAPI()

# fila assíncrona para armazenar apenas o último frame recebido
ayncio_loop = asyncio.get_event_loop()
frame_queue = asyncio.Queue(maxsize=1)

#novidade


# module‐level state
last_alarm = {"timestamp": None, "detections": []}

# buffer para armazenar muitos frames recebidos
all_frames = deque(maxlen=100)
all_futures = deque(maxlen=100)

# controle de processamento
processing_lock = asyncio.Lock()

# inicia com o loop de processamento 
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(frame_processing_loop())

@app.get("/")
def health_check():
    """Verifica se a API está ativa."""
    return {"status": "API está funcionando!"}

@app.post("/process_frame")
async def process_frame_endpoint(frame: UploadFile):
    try:
        frame_content = await frame.read()
        np_arr = np.frombuffer(frame_content, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return {
                "status": "discarded",
                "alarm": False,
                "processing_time": 0.0
            }

        camera_status = check_camera_status(image)
        if camera_status != "Camera Normal":
            return {
                "status": "discarded",
                "alarm": False,
                "processing_time": 0.0,
                "message": f"Frame descartado — estado da câmera = '{camera_status}'"
            }

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Se o buffer estiver cheio, descarta o mais antigo e avisa o cliente antigo
        if len(all_frames) == all_frames.maxlen:
            old_future = all_futures.popleft()
            all_frames.popleft()
            old_future.set_result({
                "status": "discarded",
                "alarm": False,
                "processing_time": 0.0,
                "detections": [],
                "message": "Frame descartado — buffer cheio (substituído por um mais recente)"
            })


        # Adiciona novo frame e future ao buffer
        all_frames.append(frame_content)
        all_futures.append(future)

        return await future

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# loop de processamento de frames em segundo plano
async def frame_processing_loop():
    while True:
        await asyncio.sleep(0.1)  # dá tempo para acumular frames

        #logger.info(f"🔍 Tamanho atual do buffer: {len(all_frames)}")
        
        if len(all_frames) < 1 or processing_lock.locked():
            continue  # espera ter frames e não estar ocupado

        async with processing_lock:
            logger.info("🔄 Processando os 5 frames espaçados temporalmente...")

            N = 10
            total = len(all_frames)

            if total < N:
                indices = list(range(total))
            else:
                indices = [int(i * total / N) for i in range(N)]

            frames_to_process = [all_frames[i] for i in indices]
            futures_to_set = [all_futures[i] for i in indices]
            
            all_frames.clear()
            all_futures.clear()
            
            try:
                results = await process_frame(frames_to_process)
                

                # envia os resultados individuais
                for future, result in zip(futures_to_set, results):

                     # If this batch flagged an alarm, record it:
                    if result.get("alarm"):
                        last_alarm["timestamp"]  = datetime.now(timezone.utc).isoformat()
                        last_alarm["detections"] = result.get("detections", [])


                    future.set_result({
                        "status": result["status"],
                        "alarm": result["alarm"],
                        "processing_time": result["processing_time"],
                        "detections": result["detections"],
                        "extra_info": result.get("extra_info", [])
                    })

            except Exception as e:
                logger.error("Erro ao processar batch: %s", e)
                for future in futures_to_set:
                    future.set_result({
                        "status": "error",
                        "alarm": False,
                        "processing_time": 0.0,
                        "detections": [],
                        "extra_info": result.get("extra_info", []),
                        "error": str(e)
                    })

            
@app.get("/status")
def get_status():
    if last_alarm["timestamp"] is None:
        return {"message": "No alarms have occurred yet."}
    return {
        "message":    "Alarm detected",
        "timestamp":  last_alarm["timestamp"],
        "detections": last_alarm["detections"],
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inicia a API de processamento de frames"
    )
    parser.add_argument(
        "--port", type=int,
        help="Porta onde a API irá escutar (ex: 8050)",
        default=None
    )
    args = parser.parse_args()

    if args.port:
        port = args.port
    else:
        try:
            port = int(input("Digite a porta para iniciar a API (ex: 8050): "))
        except Exception:
            print("Porta inválida, usando 8050 por padrão.")
            port = 8050

    logger.info(f"🚀 Iniciando a API FastAPI na porta {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

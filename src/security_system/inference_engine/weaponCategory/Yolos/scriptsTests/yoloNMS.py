import time
import os
import cv2
import pandas as pd
from ultralytics import YOLO

# Caminho do modelo
model_path = os.path.join(os.path.dirname(__file__), "../models/best.pt")
weapon_model = YOLO(model_path)

# Imagem de teste
input_image_path = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EvaluateModels\Distribuicao\assaltoArma.png"
frame = cv2.imread(input_image_path)
if frame is None:
    raise FileNotFoundError(f"Imagem não encontrada: {input_image_path}")

# Função para medir tempo de inferência
def measure_inference(model, frame, nms: bool):
    start = time.time()
    # Executa inference com ou sem NMS
    results = model(frame, nms=nms)
    _ = list(results)  # força computação
    return time.time() - start

# ==================== WARM-UP ====================
# Descarta a primeira inferência para aquecer o modelo
_ = measure_inference(weapon_model, frame, nms=True)

# ==================== MEDIÇÕES ====================
timings = []
# 10 vezes com NMS
for i in range(1, 11):
    t = measure_inference(weapon_model, frame, nms=True)
    timings.append({'mode': 'with_nms', 'run': i, 'time_s': t})
    print(f"Run {i} with_nms: {t:.4f} s")
# 10 vezes sem NMS
for i in range(1, 11):
    t = measure_inference(weapon_model, frame, nms=False)
    timings.append({'mode': 'without_nms', 'run': i, 'time_s': t})
    print(f"Run {i} without_nms: {t:.4f} s")

# ==================== SALVAR EM EXCEL ====================
df = pd.DataFrame(timings)
output_excel = os.path.join(os.path.dirname(__file__), 'yolo_timing.xlsx')
df.to_excel(output_excel, index=False)
print(f"Tempos salvos em: {output_excel}")

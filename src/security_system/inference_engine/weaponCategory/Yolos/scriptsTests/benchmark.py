import os
import time
import psutil
import torch
import cv2
import pandas as pd
import glob
from ultralytics import YOLO
import sys 

# Diretório onde os modelos estão armazenados
model_paths = [
    r"C:\Vasco\Tese\Projeto\cv_kiosk_safety\models\best.pt",
    r"C:\Vasco\Tese\Projeto\cv_kiosk_safety\models\best2.pt",
    r"C:\Vasco\Tese\Projeto\cv_kiosk_safety\models\model.pt"
]

# Caminhos do dataset
dataset_path = r"C:\Vasco\Tese\Projeto\cv_kiosk_safety\datasets\Weapons\weapon detection cctv v3 dataset.v1-weapon_detection_in_cctv.yolov8\valid"
image_folder = os.path.join(dataset_path, "images")

# Pasta de saída
output_folder = r"C:\Vasco\Tese\Projeto\cv_kiosk_safety\output"
os.makedirs(output_folder, exist_ok=True)
excel_path = os.path.join(output_folder, "benchmark_resultswithoutCharger.xlsx")

# Lista para armazenar os resultados do benchmark
benchmark_results = []

# Função para medir a memória usada
def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # Em MB

# Função para processar um modelo e avaliar o desempenho
def benchmark_model(model_path, test_images):
    # Carregar o modelo
    model = YOLO(model_path)

    # Medir tamanho do modelo no disco
    model_size = os.path.getsize(model_path) / (1024 ** 2)  # Em MB

    # Medir tempo de inferência e uso de memória
    start_time = time.time()
    start_mem = get_memory_usage()

    images_processed = 0  # Contador de imagens processadas

    with torch.no_grad():
        for image_path in test_images:
            img = cv2.imread(image_path)
            if img is None:
                continue  # Ignora imagens inválidas
            model(image_path)
            images_processed += 1  # Incrementa o contador

    end_time = time.time()
    end_mem = get_memory_usage()

    # Calcular métricas
    total_time = end_time - start_time
    avg_inference_time = total_time / images_processed if images_processed > 0 else 0  # Evita divisão por zero
    memory_usage = end_mem - start_mem  # Memória usada durante inferência
    fps = images_processed / total_time if total_time > 0 else 0  # FPS (frames por segundo)

    # Guardar resultados
    return {
        "Modelo": os.path.basename(model_path),
        "Tamanho do Modelo (MB)": round(model_size, 2),
        "Tempo Médio de Inferência (s)": round(avg_inference_time, 4),
        "Memória Usada (MB)": round(memory_usage, 2),
        "FPS (Frames por Segundo)": round(fps, 2),
        "Imagens Processadas": images_processed  # Nova métrica
    }

# Lista de imagens para teste
test_images = glob.glob(os.path.join(image_folder, "*.jpg"))  # Testa apenas 10 imagens

# Executar benchmarks para cada modelo
for model_path in model_paths:
    print(f"Uso da CPU antes do modelo: {psutil.cpu_percent()}%")
    sys.stdout.write(f"Processando {model_path}...\n")
    results = benchmark_model(model_path, test_images)
    sys.stdout.write(f"Processando resultados {model_path}...\n")
    benchmark_results.append(results)

# Criar DataFrame e guardar em Excel
df_results = pd.DataFrame(benchmark_results)
df_results.to_excel(excel_path, index=False)

print(f"Benchmark concluído! Resultados guardados em {excel_path}")

import os
import cv2
import pandas as pd
import glob
import subprocess
import sys
from ultralytics import YOLO

# Definir nome do modelo atual (modifique conforme necessário)
model_name = "Best"

# Caminhos do dataset
dataset_path = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\Datasets\Weapons\weapon detection cctv v3 dataset.v1-weapon_detection_in_cctv.yolov8\valid"
image_folder = os.path.join(dataset_path, "images")
label_folder = os.path.join(dataset_path, "labels")

# Caminho do modelo YOLO
model_path = os.path.join(os.path.dirname(__file__), "../models/best.pt")
weapon_model = YOLO(model_path)

# Caminho do arquivo Excel
output_folder = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\Yolos\YoloV8\TestesYolo\Test"
os.makedirs(output_folder, exist_ok=True)
image_list_path = os.path.join(output_folder, "image_list.xlsx")  # Arquivo inicial com as imagens
detections_path = os.path.join(output_folder, "detectionsComparisonCCTV.xlsx")  # Arquivo com resultados

# Criar um arquivo Excel com a lista de imagens (somente se não existir)
def create_image_list():
    if not os.path.exists(image_list_path):
        image_files = [os.path.basename(img) for img in glob.glob(os.path.join(image_folder, "*.jpg"))]
        df = pd.DataFrame({"Imagem": image_files})
        df.to_excel(image_list_path, index=False)
        print(f"✅ Arquivo de imagens criado: {image_list_path}")

# Carregar a lista de imagens existente
def load_image_list():
    if os.path.exists(image_list_path):
        return pd.read_excel(image_list_path)["Imagem"].tolist()
    else:
        raise FileNotFoundError(f"Arquivo de imagens não encontrado: {image_list_path}")

# Função para calcular IoU (Interseção sobre União)
def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

# Processar dataset e adicionar resultados ao Excel
def process_dataset():
    image_list = load_image_list()  # Garantir que estamos usando as imagens da lista
    detections = []

    for filename in image_list:
        image_path = os.path.join(image_folder, filename)
        if not os.path.exists(image_path):
            print(f"⚠️ Imagem não encontrada: {image_path}")
            continue

        # Fazer inferência com YOLO
        results = YOLO(model_path)(image_path)
        max_confidence = 0
        detected = False

        # Processar detecções
        for result in results:
            for box in result.boxes:
                confidence = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if confidence > max_confidence:
                    max_confidence = confidence

                # Verificar interseção (IoU) com ground truth
                for label in labels:
                    if label['class_id'] == 1:  
                        gt_box = [int(coord * frame.shape[1]) if i % 2 == 0 else int(coord * frame.shape[0]) for i, coord in enumerate(label['bbox'])]
                        iou = calculate_iou([x1, y1, x2, y2], gt_box)

                        if iou > 0:  
                            detected = True
                            detections.append({
                                "Imagem": filename,
                                f"Confiança - {model_name}": round(confidence, 4),
                                f"Interseção (%) - {model_name}": round(iou * 100, 2),
                                f"Tem Bounding Box - {model_name}": has_gt_bounding_box
                            })

        # Se não houver detecção, registrar com 0
        if not detected:
            detections.append({
                "Imagem": filename,
                f"Confiança - {model_name}": round(max_confidence, 4),
                f"Interseção (%) - {model_name}": 0,
                f"Tem Bounding Box - {model_name}": has_gt_bounding_box
            })

    save_results_to_excel(detections)

# Salvar os resultados no Excel corretamente
def save_results_to_excel(data):
    df_new = pd.DataFrame(data)

    if os.path.exists(detections_path):
        with pd.ExcelWriter(detections_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
            df_new.to_excel(writer, sheet_name=model_name, index=False)
    else:
        with pd.ExcelWriter(detections_path, engine='openpyxl') as writer:
            df_new.to_excel(writer, sheet_name=model_name, index=False)
    
    print(f"✅ Resultados salvos na aba '{model_name}' em: {detections_path}")


# Chamar o script `metricas.py` e passar o nome do modelo como argumento
def call_metricas_script(model_name):
    metricas_script = os.path.join(os.path.dirname(__file__), "metricas.py")
    try:
        subprocess.run([sys.executable, metricas_script, model_name], check=True)
        print(f"📊 Métricas calculadas para o modelo {model_name} e salvas com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao calcular métricas para {model_name}: {e}")

# Executar o código
if __name__ == "__main__":
    create_image_list()  # Criar a lista de imagens apenas se ainda não existir
    process_dataset()  # Processar e avaliar as imagens

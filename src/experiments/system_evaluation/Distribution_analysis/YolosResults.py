import time
import cv2
import json
import numpy as np
import os
from ultralytics import YOLO

# Hardcoded paths: ajuste conforme necessário
# Em Windows, use raw string (prefix r) ou escape backslashes:
WEIGHTS_PATH = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EnvironmentCategory\violance_detection\Fight-Violence-detection-yolov8\Yolo_nano_weights.pt"
INPUT_IMAGE_PATH = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EvaluateModels\Distribuicao\neutral\neutral.jpg"
OUTPUT_JSON = 'WeaponmodelNeutral.json'
# Definir pasta ou caminho de saída para imagem com bounding boxes
# Exemplo: grava no mesmo diretório com sufixo
input_basename = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
OUTPUT_IMAGE_PATH = f"{input_basename}_model.png"


def load_yolo_model():
    """Carrega modelo YOLOv8 a partir de pesos hardcoded."""
    model = YOLO(WEIGHTS_PATH)
    return model


def process_frame_yolo(model, frame: np.ndarray):
    """Processa um único frame com YOLO, retorna dicionário detalhado e lista de boxes para plot."""
    h, w = frame.shape[:2]
    start = time.time()
    results = model(frame)
    inference_time = time.time() - start

    detections = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf.item())
            class_id = int(box.cls.item())
            class_name = model.names[class_id] if hasattr(model, 'names') else str(class_id)
            # Extrair coords em pixels: lidar com possíveis formatos
            arr = box.xyxy.cpu().numpy()
            arr = arr.flatten()
            if arr.shape[0] < 4:
                print(f"Aviso: coords inesperadas, pulando detecção: {arr}")
                continue
            x1, y1, x2, y2 = arr[:4].tolist()
            xyxy_norm = [x1 / w, y1 / h, x2 / w, y2 / h]
            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': conf,
                'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'bbox_norm': {'x1': xyxy_norm[0], 'y1': xyxy_norm[1], 'x2': xyxy_norm[2], 'y2': xyxy_norm[3]}
            })
    confidences = [d['confidence'] for d in detections]
    num = len(confidences)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    max_conf = float(np.max(confidences)) if confidences else 0.0
    return {
        'inference_time': inference_time,
        'num_detections': num,
        'avg_confidence': avg_conf,
        'max_confidence': max_conf,
        'detections': detections
    }


def draw_and_save_boxes(frame: np.ndarray, detections: list, output_path: str):
    """Desenha bounding boxes na imagem e salva em output_path."""
    # Copiar frame para não alterar original
    img = frame.copy()
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        class_name = det['class_name']
        conf = det['confidence']
        # Desenhar retângulo
        color = (0, 0, 255)
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        label = f"{class_name}: {conf:.2f}"
        
        ((text_w, text_h), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    success = cv2.imwrite(output_path, img)
    if not success:
        print(f"Erro ao salvar imagem com boxes em '{output_path}'")
    else:
        print(f"Imagem com boxes salva em '{output_path}'")

# Execução para um único frame sem argparse
def main():
    model = load_yolo_model()
    frame = cv2.imread(INPUT_IMAGE_PATH)
    if frame is None:
        print(f"Erro: não foi possível ler imagem '{INPUT_IMAGE_PATH}'")
        return
    res = process_frame_yolo(model, frame)
    print(f"Processado '{INPUT_IMAGE_PATH}': {res['num_detections']} detections, tempo {res['inference_time']:.3f}s")
    # Desenhar e salvar a imagem com bounding boxes
    draw_and_save_boxes(frame, res['detections'], OUTPUT_IMAGE_PATH)
    # Salvar resultado em JSON
    output_data = {INPUT_IMAGE_PATH: res}
    try:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Resultados salvos em '{OUTPUT_JSON}'")
    except Exception as e:
        print(f"Erro ao salvar JSON: {e}")

if __name__ == '__main__':
    main()

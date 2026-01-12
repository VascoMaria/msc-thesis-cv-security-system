import time
import cv2
import json
import os
import numpy as np
from fer import FER
from PIL import Image, ImageDraw, ImageFont

# Hardcoded paths: ajustar conforme necessário
INPUT_IMAGE_PATH = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EvaluateModels\Distribuicao\neutral\neutral.jpg"
OUTPUT_JSON = 'fer_results_neutral.json'
# Annotated output image path (same basename with suffix)
base_name = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
OUTPUT_IMAGE_PATH = f"{base_name}_fer_netral.png"

# Inicializa detector FER
# mtcnn=True para detecção de rosto mais precisa, mas pode ser mais lento
detector = FER(mtcnn=True)
# Função para processar uma única imagem e retornar info detalhada
def process_image_fer(image_path):
    # Resolve extensão se necessário
    def resolve_path(path):
        if os.path.isfile(path): return path
        base, _ = os.path.splitext(path)
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            p = base + ext
            if os.path.isfile(p): return p
        return None

    real_path = resolve_path(image_path)
    if real_path is None:
        raise FileNotFoundError(f"Imagem não encontrada nem em alternativas para: {image_path}")
    # Ler imagem com OpenCV para FER (BGR)
    frame = cv2.imread(real_path)
    if frame is None:
        raise IOError(f"Erro ao ler imagem: {real_path}")
    # Para JSON: armazenar absolute path
    abs_path = os.path.abspath(real_path)
    # Medir tempo de inferência
    start = time.time()
    results = detector.detect_emotions(frame)
    inference_time = time.time() - start
    # results: lista de dicts: {'box': (x, y, w, h), 'emotions': {emotion: score, ...}}
    faces = []
    for res in results:
        box = res.get('box', None)
        emotions = res.get('emotions', {})
        # Top emotion
        if emotions:
            top_emotion, top_conf = max(emotions.items(), key=lambda x: x[1])
        else:
            top_emotion, top_conf = None, 0.0
        faces.append({
            'box': {'x': int(box[0]), 'y': int(box[1]), 'w': int(box[2]), 'h': int(box[3])} if box is not None else None,
            'emotions': emotions,
            'top_emotion': top_emotion,
            'top_confidence': float(top_conf)
        })
    return {
        'image_path': abs_path,
        'inference_time': inference_time,
        'faces': faces
    }

# Função para desenhar boxes e labels e salvar imagem anotada
def draw_and_save(frame_path, faces, output_path):
    # Usa PIL para desenhar
    pil_img = Image.open(frame_path).convert('RGB')
    draw = ImageDraw.Draw(pil_img)
    # Fonte padrão
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()
    for face in faces:
        box = face.get('box')
        top = face.get('top_emotion')
        conf = face.get('top_confidence')
        if box is None:
            continue
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        # desenhar retângulo
        draw.rectangle([x, y, x+w, y+h], outline=(255, 0, 0), width=2)
        label = f"{top}: {conf:.2f}" if top is not None else ""
        # Calcular tamanho do texto com textbbox ou font.getsize
        try:
            bbox = draw.textbbox((x, y), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = font.getsize(label)
        # desenhar fundo texto
        draw.rectangle([x, y - text_h - 4, x + text_w, y], fill=(255,0,0))
        draw.text((x, y - text_h - 2), label, fill=(255,255,255), font=font)
    # Salvar
    pil_img.save(output_path)
    print(f"Imagem anotada salva em '{output_path}'")

# Execução principal
if __name__ == '__main__':
    try:
        result = process_image_fer(INPUT_IMAGE_PATH)
    except Exception as e:
        print(f"Erro: {e}")
        exit(1)
    # Imprime resumo
    num_faces = len(result['faces'])
    print(f"Processado '{result['image_path']}': {num_faces} face(s) detectada(s), tempo {result['inference_time']:.3f}s")
    # Salvar JSON
    try:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Resultados salvos em '{OUTPUT_JSON}'")
    except Exception as e:
        print(f"Erro ao salvar JSON: {e}")
    # Desenhar e salvar imagem anotada
    real_path = result['image_path']
    draw_and_save(real_path, result['faces'], OUTPUT_IMAGE_PATH)

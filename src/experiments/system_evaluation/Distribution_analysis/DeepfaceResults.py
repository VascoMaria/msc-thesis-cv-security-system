import os
import json
import time
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont

# Hardcoded paths: ajuste conforme necessário
INPUT_IMAGE_PATH = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EvaluateModels\Distribuicao\neutral\neutral.jpg"
OUTPUT_JSON = 'deepface_neutral.json'
# Annotated output image path
base_name = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
OUTPUT_IMAGE_PATH = f"{base_name}_deepface_neutral.png"

# Função para resolver extensão de imagem
def resolve_path(path):
    if os.path.isfile(path): return path
    base, _ = os.path.splitext(path)
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        p = base + ext
        if os.path.isfile(p): return p
    return None

# Função para processar imagem e detectar emoções usando DeepFace
def process_image_deepface(image_path):
    real_path = resolve_path(image_path)
    if real_path is None:
        raise FileNotFoundError(f"Imagem não encontrada nem em alternativas para: {image_path}")
    abs_path = os.path.abspath(real_path)
    # Ler imagem com OpenCV e converter para RGB
    frame_bgr = cv2.imread(real_path)
    if frame_bgr is None:
        raise IOError(f"Erro ao ler imagem: {real_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Medir tempo de inferência
    start = time.time()
    try:
        # DeepFace.analyze detecta e retorna lista/dict
        analysis = DeepFace.analyze(img_path = real_path, actions=['emotion'], enforce_detection=True)
    except Exception:
        # Retry with enforce_detection=False
        analysis = DeepFace.analyze(img_path = real_path, actions=['emotion'], enforce_detection=False)
    inference_time = time.time() - start
    faces = []
    # analysis pode ser dict (uma face) ou list de dicts
    analyses = analysis if isinstance(analysis, list) else [analysis]
    for res in analyses:
        # Região detectada
        region = res.get('region', None)
        emotions = res.get('emotion', {})
        if region:
            x, y, w, h = int(region.get('x',0)), int(region.get('y',0)), int(region.get('w',0)), int(region.get('h',0))
        else:
            # se não houver region, usar toda imagem
            h_img, w_img = frame_bgr.shape[:2]
            x, y, w, h = 0, 0, w_img, h_img
        # Top emotion
        if emotions:
            top_emotion, top_conf = max(emotions.items(), key=lambda x: x[1])
        else:
            top_emotion, top_conf = None, 0.0
        faces.append({
            'box': {'x': x, 'y': y, 'w': w, 'h': h},
            'emotions': emotions,
            'top_emotion': top_emotion,
            'top_confidence': float(top_conf)
        })
    return {
        'image_path': abs_path,
        'inference_time': inference_time,
        'faces': faces
    }

# Função para desenhar e salvar imagem anotada
def draw_and_save(frame_path, faces, output_path):
    pil_img = Image.open(frame_path).convert('RGB')
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()
    for face in faces:
        box = face.get('box')
        top = face.get('top_emotion')
        conf = face.get('top_confidence')
        emotions = face.get('emotions', {})
        if box is None:
            continue
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        # desenhar retângulo
        draw.rectangle([x, y, x+w, y+h], outline=(255, 0, 0), width=2)
        # label com top emotion e confiança
        label = f"{top}: {conf:.2f}" if top is not None else ""
        # Calcular tamanho do texto
        try:
            bbox = draw.textbbox((x, y), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = font.getsize(label)
        # desenhar fundo texto
        draw.rectangle([x, y - text_h - 4, x + text_w, y], fill=(255,0,0))
        draw.text((x, y - text_h - 2), label, fill=(255,255,255), font=font)
        # opcional: desenhar barra de emoções
        # desenhar texto adicional abaixo da caixa
        offset_y = y + h + 5
        for emo, score in emotions.items():
            txt = f"{emo}: {score:.2f}"
            try:
                bbox2 = draw.textbbox((x, offset_y), txt, font=font)
                tw = bbox2[2] - bbox2[0]
                th = bbox2[3] - bbox2[1]
            except AttributeError:
                tw, th = font.getsize(txt)
            draw.text((x, offset_y), txt, fill=(255,255,255), font=font)
            offset_y += th + 2
    pil_img.save(output_path)
    print(f"Imagem anotada salva em '{output_path}'")

# Execução principal
if __name__ == '__main__':
    try:
        result = process_image_deepface(INPUT_IMAGE_PATH)
    except Exception as e:
        print(f"Erro: {e}")
        exit(1)
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
    draw_and_save(result['image_path'], result['faces'], OUTPUT_IMAGE_PATH)

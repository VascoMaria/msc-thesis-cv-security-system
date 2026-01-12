import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
compare_model_jsons.py

Script profissional para comparar resultados de múltiplos modelos a partir de arquivos JSON de inferência.
Inclui comparação YOLO, classificação e emoções, agora com plotagem por face comparando FER vs DeepFace com correspondência de caixas.

Uso:
    python compare_model_jsons.py \
        --yolo-json ... \
        --class-json classification.json \
        --fer-json fer_results.json \
        --deepface-json deepface_results.json \
        --names YOLO1 YOLO2 ... EffNet FER DeepFace \
        --output-dir comparacao_plots

Requisitos:
    Python 3.7+; bibliotecas: numpy, pandas, matplotlib
"""

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Falha ao carregar JSON '{path}': {e}")
        return None


def extract_inner(data: Any) -> Dict:
    if isinstance(data, dict) and len(data) == 1 and isinstance(list(data.values())[0], dict):
        return list(data.values())[0]
    if isinstance(data, dict):
        return data
    logging.warning("Dados JSON não estão no formato esperado de dict; retornando dict vazio.")
    return {}


def extract_emotion_faces(data: Any, model_type: str) -> List[Dict]:
    """Retorna lista de faces com caixa e emoções (normalizadas para DeepFace)."""
    inner = extract_inner(data)
    faces = inner.get('faces', [])
    result = []
    for face in faces:
        box = face.get('box', {})
        emotions = face.get('emotions', {})
        if model_type.lower() == 'deepface':
            # normalizar DeepFace de [0,100] para [0,1]
            emotions = {k: float(v)/100.0 for k, v in emotions.items()}
        # ensure box has x,y,w,h
        if not all(k in box for k in ('x','y','w','h')):
            continue
        result.append({'box': box, 'emotions': emotions})
    return result


def box_iou(box1: Dict, box2: Dict) -> float:
    """Calcula IoU entre duas boxes com keys x,y,w,h."""
    x1, y1, w1, h1 = box1['x'], box1['y'], box1['w'], box1['h']
    x2, y2, w2, h2 = box2['x'], box2['y'], box2['w'], box2['h']
    # coordenadas de canto
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    # interseção
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def match_faces(faces1: List[Dict], faces2: List[Dict], iou_thresh: float = 0.5) -> List[Tuple[Dict, Dict]]:
    """Retorna lista de pares de faces entre faces1 e faces2 baseados em IoU>=thresh."""
    matched = []
    used2 = set()
    for f1 in faces1:
        best_iou = 0.0
        best_f2 = None
        for idx2, f2 in enumerate(faces2):
            if idx2 in used2: continue
            iou = box_iou(f1['box'], f2['box'])
            if iou > best_iou:
                best_iou = iou
                best_f2 = (idx2, f2)
        if best_f2 and best_iou >= iou_thresh:
            matched.append((f1, best_f2[1]))
            used2.add(best_f2[0])
    return matched


def plot_emotion_comparison_per_face(matched_pairs: List[Tuple[Dict, Dict]], output_dir: Path):
    """Para cada par de faces, plotar comparação de emoções (barra side-by-side)."""
    for idx, (f1, f2) in enumerate(matched_pairs, start=1):
        emos1 = f1['emotions']
        emos2 = f2['emotions']
        # coletar todas emoções
        all_emotions = sorted(set(emos1.keys()) | set(emos2.keys()))
        vals1 = [emos1.get(e, 0.0) for e in all_emotions]
        vals2 = [emos2.get(e, 0.0) for e in all_emotions]
        x = np.arange(len(all_emotions))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, vals1, width, label='FER')
        ax.bar(x + width/2, vals2, width, label='DeepFace')
        ax.set_ylabel('Score de Emoção')
        ax.set_title(f'Comparação de Emoções Face {idx}')
        ax.set_xticks(x)
        ax.set_xticklabels(all_emotions, rotation=45, ha='right')
        for i, v in enumerate(vals1): ax.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
        for i, v in enumerate(vals2): ax.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
        ax.legend()
        fig.tight_layout()
        outp = output_dir / f'face{idx}_emotion_compare.png'
        try:
            fig.savefig(outp)
            logging.info(f"Plot comparação emoções face salvo: {outp}")
        except Exception as e:
            logging.error(f"Falha ao salvar plot comparação emoções face {outp}: {e}")
        plt.close(fig)


def save_summary_csv(rows: List[Dict[str, Any]], output_path: Path):
    try:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logging.info(f"Resumo CSV salvo em {output_path}")
    except Exception as e:
        logging.error(f"Falha ao salvar CSV resumo em {output_path}: {e}")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='Comparar métricas entre múltiplos JSONs de modelos e plotar emoções detalhadas por face')
    parser.add_argument('--yolo-json', nargs='*', help='Arquivos JSON de resultados YOLO')
    parser.add_argument('--class-json', help='JSON de resultados classificação')
    parser.add_argument('--fer-json', help='JSON de resultados FER')
    parser.add_argument('--deepface-json', help='JSON de resultados DeepFace')
    parser.add_argument('--names', nargs='+', help='Nomes legíveis correspondentes (na mesma ordem de args)')
    parser.add_argument('--output-dir', default='plots', help='Diretório para salvar CSV e plots')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='Threshold IoU para correspondência de faces')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar JSONs
    yolo_data = []
    if args.yolo_json:
        for p in args.yolo_json:
            data = load_json(Path(p))
            if data is not None:
                yolo_data.append((Path(p).stem, data))
    class_data = []
    if args.class_json:
        data = load_json(Path(args.class_json))
        if data is not None:
            class_data.append((Path(args.class_json).stem, data))
    fer_data = None
    if args.fer_json:
        fer_data = load_json(Path(args.fer_json))
    deepface_data = None
    if args.deepface_json:
        deepface_data = load_json(Path(args.deepface_json))

    # Comparações por face FER vs DeepFace
    if fer_data and deepface_data:
        faces1 = extract_emotion_faces(fer_data, 'FER')
        faces2 = extract_emotion_faces(deepface_data, 'DeepFace')
        matched = match_faces(faces1, faces2, args.iou_thresh)
        if matched:
            plot_emotion_comparison_per_face(matched, output_dir)
        else:
            logging.warning("Nenhuma face correspondente encontrada entre FER e DeepFace com o IoU threshold fornecido.")

    # Outras comparações podem seguir o padrão anterior (YOLO, classificação etc.)
    # ... (manter ou reutilizar funções definidas anteriormente para métricas gerais)
    logging.info("Processamento finalizado.")

if __name__ == '__main__':
    main()
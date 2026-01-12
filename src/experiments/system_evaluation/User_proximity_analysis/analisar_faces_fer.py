import os
import cv2
import numpy as np
from fer import FER
from openpyxl import Workbook

# ====== CONFIG ======
DATASET_BASE = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EvaluateModels\HefestoPhotos\Areas"  # pasta que contém "positivos" e "negativos"
OUTPUT_EXCEL = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EvaluateModels\HefestoPhotos\Areas\faces_ratios.xlsx"
PASTAS = ["positivos", "negativos"]

# Ativa MTCNN (mais preciso, mais lento)
detector = FER(mtcnn=True)


def get_faces_areas(frame):
    results = detector.detect_emotions(frame)
    areas = []
    for r in results or []:
        box = r.get("box") or [0, 0, 0, 0]
        if len(box) >= 4:
            w, h = int(box[2] or 0), int(box[3] or 0)
            area = max(w, 0) * max(h, 0)
            if area > 0:
                areas.append(area)
    areas.sort(reverse=True)
    return areas


def process_pasta(label: str, pasta_path: str, ws):
    ratios = []
    for fname in os.listdir(pasta_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(pasta_path, fname)
        frame = cv2.imread(path)
        if frame is None:
            print(f"[AVISO] Não consegui ler {fname}")
            continue

        areas = get_faces_areas(frame)
        if not areas:
            print(f"[{label}/{fname}] Nenhuma cara detectada")
            ws.append([fname, label, 0, 0, 0.0])
            continue

        user_area = areas[0]
        intruder_area = areas[1] if len(areas) > 1 else 0
        ratio = (intruder_area / user_area * 100) if user_area > 0 else 0

        ratios.append(ratio)

        print(f"[{label}/{fname}] user={user_area} intruder={intruder_area} ratio={ratio:.1f}%")
        ws.append([fname, label, user_area, intruder_area, round(ratio, 1)])

    return ratios


def main():
    wb = Workbook()
    ws = wb.active
    ws.title = "Resultados"
    ws.append(["Frame", "Label", "UserArea", "IntruderArea", "Ratio(%)"])

    resultados = {}
    for pasta in PASTAS:
        pasta_path = os.path.join(DATASET_BASE, pasta)
        if not os.path.exists(pasta_path):
            print(f"❌ Pasta não encontrada: {pasta_path}")
            continue
        print(f"\n>>> Processando pasta: {pasta}")
        ratios = process_pasta(pasta, pasta_path, ws)
        resultados[pasta] = ratios

        if ratios:
            print(f"\n--- Estatísticas {pasta} ---")
            print(f"Frames: {len(ratios)}")
            print(f"Média: {np.mean(ratios):.1f}%")
            print(f"Mediana: {np.median(ratios):.1f}%")
            print(f"Máximo: {np.max(ratios):.1f}%")
            print(f"Mínimo: {np.min(ratios):.1f}%")

    wb.save(OUTPUT_EXCEL)
    print(f"\n✅ Resultados guardados em: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()

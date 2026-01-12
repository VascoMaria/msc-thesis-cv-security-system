import os
import cv2
from fer import FER
from openpyxl import Workbook

# ====== CONFIG ======
DATASET_BASE = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EvaluateModels\HefestoPhotos\Areas"
OUTPUT_EXCEL = os.path.join(DATASET_BASE, "classificacao_faces.xlsx")
PASTAS = ["positivos", "negativos"]

# Limiar em percentagem
THRESHOLD = 31.0

# Ativar MTCNN no FER
detector = FER(mtcnn=True)


def get_faces_areas(frame):
    """Deteta faces e devolve lista de áreas, ordenada da maior para a menor."""
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


def main():
    wb = Workbook()
    ws = wb.active
    ws.title = "Classificacao"
    ws.append(["Frame", "GroundTruth", "Resultado", "Ratio(%)"])

    for pasta in PASTAS:
        pasta_path = os.path.join(DATASET_BASE, pasta)
        if not os.path.exists(pasta_path):
            print(f"❌ Pasta não encontrada: {pasta_path}")
            continue

        for fname in os.listdir(pasta_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(pasta_path, fname)
            frame = cv2.imread(path)
            if frame is None:
                print(f"[AVISO] Não consegui ler {fname}")
                continue

            areas = get_faces_areas(frame)
            if len(areas) < 2:
                # Só consideramos imagens com duas caras detetadas
                continue

            user_area, intruder_area = areas[0], areas[1]
            ratio = (intruder_area / user_area) * 100 if user_area > 0 else 0

            # Classificação pelo limiar
            resultado = "positivo" if ratio >= THRESHOLD else "negativo"

            # Ground truth = nome da pasta
            groundtruth = pasta

            print(f"[{fname}] GT={groundtruth}, ratio={ratio:.1f}%, resultado={resultado}")
            ws.append([fname, groundtruth, resultado, round(ratio, 1)])

    wb.save(OUTPUT_EXCEL)
    print(f"\n✅ Resultados guardados em: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()

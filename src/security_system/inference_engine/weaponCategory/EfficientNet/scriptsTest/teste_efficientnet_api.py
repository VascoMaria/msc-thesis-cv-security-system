import os
import glob
import requests
import pandas as pd
from PIL import Image
import json

def load_labels(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            labels.append({'class_id': cls, 'bbox_norm': [x_c, y_c, w, h]})
    return labels

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0

def main():
    # Configurações iniciais
    dataset_path = input("Caminho do dataset (pasta 'valid'): ").strip()
    image_folder = os.path.join(dataset_path, "images")
    label_folder = os.path.join(dataset_path, "labels")
    server_url = input("URL do servidor de detecção (ex: http://localhost:8010/detect): ").strip()
    selected_classes = list(map(int, input("IDs das classes a considerar (vírgula-separados): ").split(',')))

    image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) +                   glob.glob(os.path.join(image_folder, "*.png"))
    detections = []

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + ".txt")

        # Carrega ground truth
        labels = load_labels(label_path)
        # Reconverte bboxes para pixels
        img = Image.open(img_path)
        W, H = img.size
        gt_boxes = []
        for lbl in labels:
            if lbl['class_id'] in selected_classes:
                x_c, y_c, w, h = lbl['bbox_norm']
                x1 = int((x_c - w/2) * W)
                y1 = int((y_c - h/2) * H)
                x2 = int((x_c + w/2) * W)
                y2 = int((y_c + h/2) * H)
                gt_boxes.append([x1, y1, x2, y2])
        has_gt = 1 if gt_boxes else 0

        # Envia ao endpoint
        ext = os.path.splitext(filename)[1].lower()
        mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        with open(img_path, "rb") as f:
            files = {"file": (filename, f, mime)}
            resp = requests.post(server_url, files=files)
        resp.raise_for_status()
        result = resp.json()

        det = result.get("detections", [])[0]
        conf = det.get("confidence", 0.0)
        label = det.get("label", "")
        box = det.get("bounding_box")

        if box:
            pred_box = [box["x1"], box["y1"], box["x2"], box["y2"]]
            best_iou = max(calculate_iou(pred_box, gt) for gt in gt_boxes) if gt_boxes else 0.0
        else:
            best_iou = 0.0

        detections.append({
            "Imagem": filename,
            "Confiança - EfficientNet": round(conf, 4),
            "Interseção (%) - EfficientNet": round(best_iou * 100, 2),
            "Label - EfficientNet": label,
            "Tem Bounding Box": has_gt
        })

    # Salva resultados
    df = pd.DataFrame(detections)
    # Colunas de FP e FN
    fp_col = "FP - EfficientNet"
    fn_col = "FN - EfficientNet"
    df[fp_col] = 0
    df[fn_col] = 0
    for idx, row in df.iterrows():
        has_gt = row["Tem Bounding Box"]
        conf = row["Confiança - EfficientNet"]
        if has_gt and conf == 0:
            df.at[idx, fn_col] = 1
        if not has_gt and conf > 0:
            df.at[idx, fp_col] = 1

    output_file = os.path.join(os.getcwd(), "resultado_efficientnet_api.xlsx")
    df.to_excel(output_file, index=False)
    print(f"✅ Resultados salvos em: {output_file}")

if __name__ == "__main__":
    main()

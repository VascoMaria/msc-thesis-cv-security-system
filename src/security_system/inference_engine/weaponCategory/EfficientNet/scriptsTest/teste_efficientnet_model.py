import os
import glob
import cv2
import requests
import argparse
import pandas as pd

# --- CONFIGURAÇÃO FIXA ---
dataset_path = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\Datasets\Weapons\silah.v3i.yolov8\valid\split_1"
api_url      = "http://localhost:8010/detect"
timeout_secs = 35  # tempo máximo por imagem
output_excel = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\Datasets\Weapons\silah.v3i.yolov8\valid\split_1_results.xlsx"

# -------- Funções auxiliares --------

def load_labels(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cid = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            x1 = x_c - w/2; y1 = y_c - h/2
            x2 = x_c + w/2; y2 = y_c + h/2
            labels.append({'class_id': cid, 'bbox': [x1, y1, x2, y2]})
    return labels

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter/union if union>0 else 0

def call_api(image_path):
    with open(image_path, 'rb') as f:
        resp = requests.post(api_url, files={'file': f}, timeout=timeout_secs)
    resp.raise_for_status()
    return resp.json().get('detections', [])

# -------- Loop principal --------

def process_dataset(selected_classes):
    image_folder = os.path.join(dataset_path, "images")
    label_folder = os.path.join(dataset_path, "labels")

    rows = []
    total_timeouts = 0

    for img_path in glob.glob(os.path.join(image_folder, "*.jpg")):
        fn = os.path.basename(img_path)
        # GT
        gt = load_labels(os.path.join(label_folder, fn.replace('.jpg','.txt')))
        has_gt = any(l['class_id'] in selected_classes for l in gt)

        # chama API
        try:
            dets = call_api(img_path)
        except requests.exceptions.Timeout:
            print(f"[{fn}] ⏱ Timeout, saltando.")
            total_timeouts += 1
            continue
        except Exception as e:
            print(f"[{fn}] ❌ Erro: {e}, saltando.")
            continue

        # prepara caixas GT em px
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        gt_boxes_px = [
            [int(b['bbox'][0]*w), int(b['bbox'][1]*h),
             int(b['bbox'][2]*w), int(b['bbox'][3]*h)]
            for b in gt if b['class_id'] in selected_classes
        ]

        # escolhe melhor deteção
        best_conf, best_iou = 0.0, 0.0
        best_label = "none"
        for d in dets:
            conf = float(d['confidence'])
            bbox = d.get('bounding_box')
            if bbox and gt_boxes_px:
                box = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
                for gt_box in gt_boxes_px:
                    iou = calculate_iou(box, gt_box)
                    if iou > 0 and (conf > best_conf or (conf == best_conf and iou > best_iou)):
                        best_conf, best_iou, best_label = conf, iou, d['label']
            else:
                if conf > best_conf:
                    best_conf, best_label = conf, d['label']

        falta = 1 if has_gt and best_conf == 0 else 0
        no_det = 1 if best_conf == 0 else 0

        model_name = "EfficientNet"
        # regista linha
        rows.append({
                "Imagem": fn,
                f"Confiança - {model_name}": round(best_conf, 4),
                f"Interseção (%) - {model_name}": 0,
                f"Tem Bounding Box - {model_name}": int(has_gt)
    })


        # print
        print(f"[{fn}] {best_label:<10} | Conf {best_conf:.4f} | IoU {best_iou*100:6.2f}% | FN {falta}")

    # DataFrame e Excel
    df = pd.DataFrame(rows)
    df.to_excel(output_excel, index=False)
    print(f"\n✅ Excel gravado em: {output_excel}")
    print(f"Total imagens processadas: {len(rows)} (+{total_timeouts} timeouts)")

# -------- Entrypoint --------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--classes", required=True,
                   help="IDs das classes alarmantes, ex: 1")
    args = p.parse_args()
    sel = list(map(int, args.classes.split(",")))
    process_dataset(sel)


# -------- Entrypoint --------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Avalia EfficientNet via API e imprime resultados no terminal"
    )
    p.add_argument("--classes", required=True,
                   help="IDs das classes alarmantes, separados por vírgula")
    args = p.parse_args()
    sel_classes = list(map(int, args.classes.split(",")))
    process_dataset(sel_classes)
